import os
import argparse
import numpy as np
import pandas as pd

import prompt_utils as utils
import optimization_utils as opt

from tqdm import tqdm 
from pathlib import Path
from scipy.stats import chi2_contingency
from embedding_utils import discretize_cols

import warnings
warnings.filterwarnings("ignore")

def pandas_entropy(column):
    vc = pd.Series(column).value_counts(normalize=True, sort=False)
    return -(vc * np.log(vc)/np.log(np.e)).sum()

def contains_numpy_array(series):
    return any(isinstance(x, np.ndarray) for x in series)

def contains_pandas_array(series):
    return any(isinstance(x, pd.DataFrame) for x in series)

def cramers_V(var1, var2) :
    crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))
    statistics = chi2_contingency(crosstab)[0]
    obs_count = np.sum(crosstab)
    mini = min(crosstab.shape)-1 
    return (statistics/(obs_count*mini))


parser = argparse.ArgumentParser()
parser.add_argument(
    '--rule_dir', type=str, default='diversity_5rules_40trials',
    help='The directory name to load features'
)
parser.add_argument(
    '--select_k', type=int, default=20,
     help='The number of selected features'
)

args = parser.parse_args()

_RULE_DIR = args.rule_dir
_SELECT_K = args.select_k


for _DATA in [
    'adult', 'insurance', 'heart', 'car', 'communities', 'credit-g', 'diabetes', 'bank', 'myocardial', 
    'junglechess', 'housing', 'solution-mix', 'forest-fires', 'eucalyptus', 'balance-scale', 'vehicle',
    'blood', 'sequence-type', 'tic-tac-toe', 'bike', 'crab', 'wine'
]:
    for _SEED in [0, 1, 2]:
        print(_DATA, _SEED)
        if os.path.isfile(f"./dataframe/{_DATA}_{_SEED}_test.csv"):
            continue

        utils.set_seed(_SEED)
        df, X_train, X_test, _, _, _, _, _ = utils.get_dataset(_DATA, _SEED)

        # Dataset for baselines
        X_train_org, X_test_org = opt.fill_missing(X_train, X_test)

        # Dataset after feature expansion
        X_train_org_added, X_test_org_added = opt.add_data_features(
            _DATA, _SEED, _RULE_DIR, X_train_org, X_test_org
        )
        
        # Remove array features
        columns_to_drop = [col for col in X_train_org_added.columns if contains_numpy_array(X_train_org_added[col]) or contains_pandas_array(X_train_org_added[col])]
        X_train_org_added = X_train_org_added.drop(columns_to_drop, axis=1)
        X_test_org_added = X_test_org_added.drop(columns_to_drop, axis=1)

        # Remove NaN features or singular features
        X_train_filtered, X_test_filtered = opt.filter_features(X_train_org_added, X_test_org_added)

        X_train_all = pd.concat([X_train_org, X_train_filtered], axis=1)
        X_test_all = pd.concat([X_test_org, X_test_filtered], axis=1)

        # Remove low entropy features
        X_train_disc, X_test_disc = discretize_cols(X_train_all, X_test_all, 0, 10)
        filtered_columns = X_train_filtered.columns[X_train_disc[X_train_filtered.columns].apply(lambda x: pandas_entropy(x)) > 0.7]
        filtered_columns_all = X_train_org.columns.append(filtered_columns)
        X_train_disc = X_train_disc[filtered_columns_all]
        X_test_disc = X_test_disc[filtered_columns_all]

        # Reduce redundancy
        rows = []
        for var1 in tqdm(X_train_disc.columns):
            col = []
            for var2 in X_train_disc.columns:
                cramers = cramers_V(X_train_disc[var1], X_train_disc[var2])
                col.append(cramers) 
            rows.append(col)
        cramers_results = np.array(rows)
        corr_results = pd.DataFrame(cramers_results, columns=X_train_disc.columns, index=X_train_disc.columns)

        corr_results = corr_results[filtered_columns]        
        selected_features = X_train_org.columns.tolist()

        for k in range(_SELECT_K):
            if len(corr_results.columns) < k:
                break

            current_corr_results = corr_results.iloc[corr_results.index.isin(selected_features)]
            current_corr_results = current_corr_results[corr_results.columns[corr_results.columns.isin(selected_features) == False]]
            if current_corr_results.max(axis=0).min() < 1.0:
                minimum_redundancy_feature = current_corr_results.columns[current_corr_results.max(axis=0).argmin()]
                selected_features.append(minimum_redundancy_feature)
            else:
                break

        X_train_addall = X_train_all[selected_features[:_SELECT_K+len(X_train_org.columns)]]
        X_test_addall = X_test_all[selected_features[:_SELECT_K+len(X_train_org.columns)]]

        X_train_addall.to_csv(f"./dataframe/{_DATA}_{_SEED}_train.csv", index=False)
        X_test_addall.to_csv(f"./dataframe/{_DATA}_{_SEED}_test.csv", index=False)