import numpy as np
from pathlib import Path
import optimization_utils as opt
import prompt_utils as utils
from tqdm import tqdm
import argparse
import pickle
import json
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import embedding_utils as eu

import torch

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
    result_path = Path(f'./{args.result_dir}')
    
    for dataset in datasets:
        for seed in seeds:
            model_dir_name = f'{dataset}_{seed}_dim_{args.hidden_dim}_iter_{args.iterations}'
                
            if args.M != -1:
                model_dir_name += f'_M_{args.M}'
                
            result_dir = result_path / model_dir_name
            print(result_dir)
            
            embedding_dir = result_dir / 'embeddings'

            utils.set_seed(seed)

            print(f'Extracting embedding for Dataset {dataset} Seed {seed}')
            df, X_train, X_test, y_train, y_test, target_attr, label_list, is_cat = utils.get_dataset(dataset, seed)

            X_train.reset_index(drop=True, inplace=True)
            X_test.reset_index(drop=True, inplace=True)

            X_train, X_test = opt.fill_missing(X_train, X_test)

            with open(result_dir / 'enc.pkl', 'rb') as f:
                enc = pickle.load(f)

            with open(result_dir / 'scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)

            X_train, _, _ = opt.process(X_train, is_cat, enc, scaler)
            X_test, _, _ = opt.process(X_test, is_cat, enc, scaler)

            if dataset in eu.REGRESSION_DATASETS:
                y_train_num = y_train
                y_test_num = y_test
            else:
                y_train_num = np.array([label_list.index(k) for k in y_train])
                y_test_num = np.array([label_list.index(k) for k in y_test])

            train_loader = eu.get_loader(X_train, y_train_num, indicators=None, batch_size=args.batch_size, shuffle=False)
            test_loader = eu.get_loader(X_test, y_test_num, indicators=None, batch_size=args.batch_size, shuffle=False)

            last_model_path = result_dir / 'model_last.pt'

            with open(result_dir / 'new_cols.json', 'r') as f:
                new_cols = json.load(f)

            if len(new_cols) > 0:
                df_path = Path('./dataframe')

                ind_train_df = pd.read_csv(df_path / f'{dataset}_{seed}_train.csv')
                ind_test_df = pd.read_csv(df_path / f'{dataset}_{seed}_test.csv')

                train_ind = ind_train_df[new_cols]
                test_ind = ind_test_df[new_cols]

                ind_dim = train_ind.shape[1]
                if args.M == -1:
                    M = ind_dim
                else:
                    M = min(args.M, ind_dim)

                    train_ind = train_ind.iloc[:, :M]
                    test_ind = test_ind.iloc[:, :M]

                model, _ = eu.get_model(
                    in_features=X_train.shape[1],
                    hidden_dim=args.hidden_dim,
                    proj_dim=args.proj_dim,
                    M=M,
                )
                model.load_state_dict(torch.load(last_model_path, map_location='cpu'))

                model = model.cuda()
            else:
                model = None

            embedding_dir.mkdir(exist_ok=True, parents=True)

            train_inputs, train_embeddings, train_labels = eu.extract_embedding(model, train_loader)
            test_inputs, test_embeddings, test_labels = eu.extract_embedding(model, test_loader)

            np.save(embedding_dir / f'train_inputs.npy', train_inputs)
            np.save(embedding_dir / f'train_embeddings.npy', train_embeddings)
            np.save(embedding_dir / f'train_labels.npy', train_labels)

            np.save(embedding_dir / f'test_inputs.npy', test_inputs)
            np.save(embedding_dir / f'test_embeddings.npy', test_embeddings)
            np.save(embedding_dir / f'test_labels.npy', test_labels)

            if len(new_cols) > 0:
                for n in [0, 5, 10, 50, 100, 500]:
                    intermediate_model_path = result_dir / f'model_iter_{n}.pt'

                    model.load_state_dict(torch.load(intermediate_model_path, map_location='cpu'))
                    model = model.cuda()

                    _, train_embeddings, _ = eu.extract_embedding(model, train_loader)
                    _, test_embeddings, _ = eu.extract_embedding(model, test_loader)

                    np.save(embedding_dir / f'iter_{n}_train_embeddings.npy', train_embeddings)
                    np.save(embedding_dir / f'iter_{n}_test_embeddings.npy', test_embeddings)