import pandas as pd
import numpy as np
from pathlib import Path
import copy
import optimization_utils as opt
import prompt_utils as utils
import embedding_utils as eu
from tqdm import tqdm
import argparse
import math
import pickle
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
    
if __name__ == "__main__":
    args = eu.parse_arguments()
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    
    if args.dataset == '':
        datasets = ['adult', 'housing']
    else:
        datasets = [args.dataset]
    
    if args.seed == -1:
        seeds = range(3)
    else:
        seeds = [args.seed]

    data_path = Path(f'./{args.data_dir}')
    df_path = Path('./dataframe')
        
    proxy_path = Path(f'./{args.proxy_dir}')
    result_path = Path(f'./{args.result_dir}')
    
    for dataset in datasets:
        for seed in seeds:
            utils.set_seed(seed)
            
            ind_train_df, ind_test_df = eu.get_indicator_df(
                dataset=dataset,
                seed=seed,
                n_clusters=args.n_clusters,
                proxy_path=proxy_path,
                df_path=df_path,
            )
            
            df, X_train, X_test, y_train, y_test, target_attr, label_list, is_cat = utils.get_dataset(dataset, seed)
            
            X_train.reset_index(drop=True, inplace=True)
            X_test.reset_index(drop=True, inplace=True)
            
            X_train, X_test = opt.fill_missing(X_train, X_test)

            train_new_col = set(ind_train_df.columns) - set(X_train.columns)
            test_new_col = set(ind_test_df.columns) - set(X_test.columns)
            assert(train_new_col == test_new_col)
                        
            train_ind = ind_train_df.drop(columns=X_train.columns, errors='ignore')
            test_ind = ind_test_df.drop(columns=X_test.columns, errors='ignore')
                        
            new_col = list(train_ind.columns)
            
            ind_dim = train_ind.shape[1]
            if args.M == -1:
                M = ind_dim
            else:
                M = min(args.M, ind_dim)

                train_ind = train_ind.iloc[:, :M]
                test_ind = test_ind.iloc[:, :M]
                
            model_dir_name = f'{dataset}_{seed}_dim_{args.hidden_dim}_iter_{args.iterations}'
                
            if args.M != -1:
                model_dir_name += f'_M_{args.M}'
                
            model_save_path = result_path / model_dir_name
            model_save_path.mkdir(exist_ok=True, parents=True)
            print(f'Path: {model_save_path}')
                
            with open(model_save_path / 'new_cols.json', 'w') as f:
                json.dump(new_col, f)
            
            X_train, enc, scaler = opt.process(X_train, is_cat)
            X_test, _, _ = opt.process(X_test, is_cat, enc, scaler)
            
            with open(model_save_path / f'enc.pkl', 'wb') as f:
                pickle.dump(enc, f, pickle.HIGHEST_PROTOCOL)
                
            with open(model_save_path / f'scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f, pickle.HIGHEST_PROTOCOL)
            
            if len(new_col) == 0:
                continue

            if dataset in eu.REGRESSION_DATASETS:
                y_train_num = y_train
                y_test_num = y_test
            else:
                y_train_num = np.array([label_list.index(k) for k in y_train])
                y_test_num = np.array([label_list.index(k) for k in y_test])
            
            
            train_loader = eu.get_loader(X_train, y_train_num, train_ind, batch_size=args.batch_size, shuffle=True)
            test_loader = eu.get_loader(X_test, y_test_num, test_ind, batch_size=args.batch_size, shuffle=False)
            
            n_cat = sum(is_cat)
            n_num = len(is_cat) - n_cat

            model, criterion = eu.get_model(
                in_features=X_train.shape[1],
                hidden_dim=args.hidden_dim,
                proj_dim=args.proj_dim,
                M=M,
            )
            model = model.cuda()

            criterion = criterion.cuda()
            
            print(f'Dataset {dataset} Seed {seed} Training')

            start = time.time()
            model = eu.train(
                model,
                train_loader,
                criterion,
                seed=seed,
                learning_rate=args.learning_rate,
                iters=args.iterations,
                model_save_path=model_save_path,
            )
            end = time.time()
            training_time = end - start
            
            torch.save(model.state_dict(), model_save_path / f'model_last.pt')
            with open(model_save_path / 'training_time', 'w') as f:
                print(training_time, file=f)