import numpy as np
from pathlib import Path
import optimization_utils as opt
import prompt_utils as utils
from tqdm import tqdm
import argparse
import pickle
import json
import pandas as pd
from collections import defaultdict

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error

import embedding_utils as eu

import torch

if __name__ == "__main__":
    args = eu.parse_arguments()
    
    if args.dataset == '':
        datasets = ['adult', 'housing']
    else:
        datasets = [args.dataset]
    
    if args.seed == -1:
        seeds = range(3)
    else:
        seeds = [args.seed]
    
    eval_methods = ['lr']

    data_path = Path(f'./{args.data_dir}')
    result_path = Path(f'./{args.result_dir}')
    
    result_df = defaultdict(dict)
    for dataset in datasets:
        for seed in seeds:
            model_dir_name = f'{dataset}_{seed}_dim_{args.hidden_dim}_iter_{args.iterations}'
                
            if args.M != -1:
                model_dir_name += f'_M_{args.M}'

            result_dir = result_path / model_dir_name
            print(result_dir)
            embedding_dir = result_dir / 'embeddings'
            
            with open(result_dir / 'new_cols.json', 'r') as f:
                new_cols = json.load(f)
            
            utils.set_seed(seed)
            print(f'Evaluating embedding for Dataset {dataset} Seed {seed}')

            train_inputs = np.load(embedding_dir / f'train_inputs.npy')
            train_embeddings = np.load(embedding_dir / f'train_embeddings.npy')
            train_labels = np.load(embedding_dir / f'train_labels.npy')

            test_inputs = np.load(embedding_dir / f'test_inputs.npy')
            test_embeddings = np.load(embedding_dir / f'test_embeddings.npy')
            test_labels = np.load(embedding_dir / f'test_labels.npy')

            if len(new_cols) == 0:
                train_embeddings = train_inputs
                test_embeddings = test_inputs

            eval_s = ""
            for m in eval_methods:
                if dataset in eu.REGRESSION_DATASETS:
                    clf = Ridge(random_state=seed)
                elif dataset not in eu.REGRESSION_DATASETS:
                    clf = LogisticRegression(random_state=seed)
                else:
                    assert(0)

                clf.fit(train_embeddings, train_labels)
                if dataset in eu.REGRESSION_DATASETS:
                    test_preds = clf.predict(test_embeddings)
                    val = mean_squared_error(test_labels, test_preds, squared=False)
                    metric_name = 'RMSE'
                else:
                    total_labels = sorted(list(set(train_labels) | set(test_labels)))
                    
                    test_pred_probs = clf.predict_proba(test_embeddings)
                    if len(total_labels) > 2:
                        val = roc_auc_score(test_labels, test_pred_probs, multi_class='ovr', labels=total_labels)
                    else:
                        val = roc_auc_score(test_labels, test_pred_probs[:,1])
                    val *= 100
                    metric_name = 'AUC'

                eval_s += f"{m} {metric_name}: {val}\n"

            eval_file_name = f'model_eval'

            with open(result_dir / eval_file_name, 'w') as eval_f:
                print(eval_s, file=eval_f, flush=True)
            
            result_df[f'Seed {seed}'][dataset] = val
            
    result_df = pd.DataFrame.from_dict(result_df)
    result_df['Mean'] = result_df.mean(axis=1)
    result_df['StdDev'] = result_df.std(axis=1)
    print(result_df)