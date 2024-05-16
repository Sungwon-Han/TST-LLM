# Utility function for getting data & prompting & query
import os
import time
import json
import openai
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from optimization_utils import is_categorical

REGRESSION_DATASETS = ['insurance', 'housing', 'solution-mix', 'forest-fires', 'bike', 'crab', 'wine']

TASK_DICT = {
    'balance-scale': "Which direction does the balance scale tip to? Right, left, or balanced?",
    'eucalyptus': "How good is this Eucalyptus species for soil conservation in the specified location? None, low, average, good, or best?",
    'vehicle': "What kind of vehicle is the given silhouette information about? Bus, opel, saab, or van?",
    'forest-fires': "Estimate the burned area of forest fires from given information.",
    'tic-tac-toe': "Will the first player (player x) win the game? Positive or negative?",
    'blood': "Did the person donate blood? Yes or no?",
    'credit-g': "Does this person receive a credit? Yes or no?",
    'diabetes': "Does this patient have diabetes? Yes or no?",
    'heart': "Does the coronary angiography of this patient show a heart disease? Yes or no?",
    'adult': "Does this person earn more than 50000 dollars per year? Yes or no?",
    'bank': "Does this client subscribe to a term deposit? Yes or no?",
    'car': "How would you rate the decision to buy this car? Unacceptable, acceptable, good or very good?",
    'communities': "How high will the rate of violent crimes per 100K population be in this area. Low, medium, or high?",
    'myocardial': "Does the myocardial infarction complications data of this patient show chronic heart failure? Yes or no?",
    'junglechess': "Which player wins this two pieces endgame of Jungle Chess? Black or white?",
    'housing': "Estimate the house price from given information.",
    'insurance': "Estimate the individual medical cost of this patient billed by health insurance",
    'solution-mix': "Given the volumes and concentrations of four solutions, calculate the percent concentration of the mixed solution after mixing them.",
    'sequence-type': "What is the type of following sequence? Arithmetic, geometric, fibonacci, or collatz?",
    'wine': "Estimate the wine quality on a scale from 0 to 10 from given information.",
    'bike': "Estimate the count of total rental bikes from given information.",
    'crab': "Estimate the age of the crab from given information.",
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    
def get_dataset(data_name, seed, shot='full'):
    file_name = f"./data/{data_name}.csv"
    df = pd.read_csv(file_name)
    default_target_attribute = df.columns[-1]
    
    categorical_indicator = [is_categorical(df.iloc[:, i]) for i in range(df.shape[1])][:-1]
    attribute_names = df.columns[:-1].tolist()

    X = df.convert_dtypes()
    y = df[default_target_attribute].to_numpy()
    label_list = np.unique(y).tolist()
    if data_name in REGRESSION_DATASETS:
        stratify = None      
    else:
        stratify = y
        
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(default_target_attribute, axis=1),
        y,
        test_size=0.2,
        random_state=seed,
        stratify=stratify,
    )
        
    if shot != 'full':
        X_new_train = X_train.copy()
        X_new_train[default_target_attribute] = y_train
        
        if data_name in REGRESSION_DATASETS:
             X_balanced = X_new_train.sample(shot, random_state=seed)
        else:
            sampled_list = []
            total_shot_count = 0
            remainder = shot % len(np.unique(y_train))
            for _, grouped in X_new_train.groupby(default_target_attribute):
                sample_num = shot // len(np.unique(y_train))
                if remainder > 0:
                    sample_num += 1
                    remainder -= 1
                grouped = grouped.sample(sample_num, random_state=seed)
                sampled_list.append(grouped)
            X_balanced = pd.concat(sampled_list)
        X_train = X_balanced.drop([default_target_attribute], axis=1)
        y_train = X_balanced[default_target_attribute].to_numpy()

    return df, X_train, X_test, y_train, y_test, default_target_attribute, label_list, categorical_indicator


def query_gpt(text_list, api_key, max_tokens=30, temperature=0, max_try_num=10, model="gpt-3.5-turbo", verbose=True):
    openai.api_key = api_key
    result_list = []
    for prompt in tqdm(text_list, disable=(verbose==False)):
        curr_try_num = 0
        while curr_try_num < max_try_num:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role":"user", "content":prompt}],
                    temperature = temperature,
                    max_tokens = max_tokens,
                    top_p = 1,
                    request_timeout=100
                )
                result = response["choices"][0]["message"]["content"]
                result_list.append(result)
                break
            except openai.error.InvalidRequestError as e:
                return [-1]
            except Exception as e:
                print(e)
                curr_try_num += 1
                if curr_try_num >= max_try_num:
                    result_list.append(-1)
                time.sleep(10)
    return result_list


def serialize(row):
    target_str = f""
    for attr_idx, attr_name in enumerate(list(row.index)):
        if attr_idx < len(list(row.index)) - 1:
            target_str += " is ".join([attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()])
            target_str += ". "
        else:
            if len(attr_name.strip()) < 2:
                continue
            target_str += " is ".join([attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()])
            target_str += "."
    return target_str


def fill_in_templates(fill_in_dict, template_str):
    for key, value in fill_in_dict.items():
        if key in template_str:
            template_str = template_str.replace(key, value)
    return template_str    


def get_prompt_for_asking(data_name, df_x, label_list, file_name, meta_file_name, is_cat, num_col=10, num_query=5, meta_data={}):
    with open(file_name, "r") as f:
        prompt_type_str = f.read()

    try:
        with open(meta_file_name, "r") as f:
            filed_meta_data = json.load(f)
    except:
        filed_meta_data = {}
        
    meta_data = dict(meta_data, **filed_meta_data)
    
    task_desc = f"{TASK_DICT[data_name]}\n"    
    df_incontext = df_x.copy()
                
    template_list = []
    current_query_num = 0
    end_flag = False
    while True:     
        if current_query_num >= num_query:
            break
                        
        # Feature bagging
        if len(df_incontext.columns) >= 20:
            total_column_list = []
            for i in range(len(df_incontext.columns) // 10):
                column_list = df_incontext.columns.tolist()
                random.shuffle(column_list)
                total_column_list.append(column_list[i*10:(i+1)*10])
        else:
            total_column_list = [df_incontext.columns.tolist()]
            
        for selected_column in total_column_list:
            if current_query_num >= num_query:
                break
                
            # Sample bagging
            threshold = 10   
            if len(df_incontext) > threshold:
                df_incontext = df_incontext.sample(threshold)
                
            feature_name_list = []
            sel_cat_idx = [df_incontext.columns.tolist().index(col_name) for col_name in selected_column]
            is_cat_sel = np.array(is_cat)[sel_cat_idx]
            
            for cidx, cname in enumerate(selected_column):
                if is_cat_sel[cidx] == True:
                    clist = df_x[cname].unique().tolist()
                    clist = [str(c) for c in clist]

                    if len(clist) > 20:
                        clist_str = f"{clist[0]}, {clist[1]}, ..., {clist[-1]}"
                    else:
                        clist_str = ", ".join(clist)
                    desc = meta_data[cname] if cname in meta_data.keys() else ""
                    feature_name_list.append(f"- {cname}: {desc} (categorical variable with categories [{clist_str}])")
                else:
                    min_val, max_val = df_x[cname].min(), df_x[cname].max()
                    desc = meta_data[cname] if cname in meta_data.keys() else ""
                    feature_name_list.append(f"- {cname}: {desc} (numerical variable within range [{min_val}, {max_val}])")

            feature_desc = "\n".join(feature_name_list)
            
            in_context_desc = ""  
            df_current = df_incontext.copy()
            df_current = df_current.sample(frac=1)

            for icl_idx, icl_row in df_current.iterrows():
                icl_row = icl_row[selected_column]
                in_context_desc += serialize(icl_row)
                in_context_desc += "\n"

            fill_in_dict = {
                "[TASK]": task_desc, 
                "[EXAMPLES]": in_context_desc,
                "[FEATURES]": feature_desc,
                "[NUM]": str(num_col)
            }
            template = fill_in_templates(fill_in_dict, prompt_type_str)
            template_list.append(template)
            current_query_num += 1
        
    return template_list, feature_desc


def get_prompt_for_asking_with_diversity(data_name, df_x, prev_modules_list, label_list, file_name, 
                                         meta_file_name, is_cat, total_column_list, current_query_num, num_col=10):
    with open(file_name, "r") as f:
        prompt_type_str = f.read()

    try:
        with open(meta_file_name, "r") as f:
            meta_data = json.load(f)
    except:
        meta_data = {}
    
    task_desc = f"{TASK_DICT[data_name]}\n"    
    df_incontext = df_x.copy()
    selected_column = total_column_list[current_query_num % len(total_column_list)]
                
    # Sample bagging
    threshold = 10   
    if len(df_incontext) > threshold:
        df_incontext = df_incontext.sample(threshold)                    

    feature_name_list = []
    sel_cat_idx = [df_incontext.columns.tolist().index(col_name) for col_name in selected_column]
    is_cat_sel = np.array(is_cat)[sel_cat_idx]

    for cidx, cname in enumerate(selected_column):
        if is_cat_sel[cidx] == True:
            clist = df_x[cname].unique().tolist()
            clist = [str(c) for c in clist]
            if len(clist) > 20:
                clist_str = f"{clist[0]}, {clist[1]}, ..., {clist[-1]}"
            else:
                clist_str = ", ".join(clist)
            desc = meta_data[cname] if cname in meta_data.keys() else ""
            feature_name_list.append(f"- {cname}: {desc} (categorical variable with categories [{clist_str}])")
        else:
            desc = meta_data[cname] if cname in meta_data.keys() else ""
            feature_name_list.append(f"- {cname}: {desc} (numerical variable)")

    feature_desc = "\n".join(feature_name_list)

    in_context_desc = ""  
    df_current = df_incontext.copy()
    df_current = df_current.sample(frac=1)

    for icl_idx, icl_row in df_current.iterrows():
        icl_row = icl_row[selected_column]
        in_context_desc += serialize(icl_row)
        in_context_desc += "\n"
        
    num = 1
    example_features_desc = ""
    for prev_module_name, prev_module_desc in prev_modules_list:
        example_features_desc += f"{num} | {prev_module_name} | {prev_module_desc}\n"
        num += 1
    example_features_desc = example_features_desc.strip()

    fill_in_dict = {
        "[TASK]": task_desc, 
        "[EXAMPLES]": in_context_desc,
        "[FEATURES]": feature_desc,
        "[EXAMPLE FEATURES]": example_features_desc,
        "[NUM]": str(num_col)
    }
    template = fill_in_templates(fill_in_dict, prompt_type_str)
    return template, feature_desc


def get_prompt_for_generating_function(parsed_col_desc_list, feature_desc, file_name):
    with open(file_name, "r") as f:
        prompt_type_str = f.read()
    
    template_list = []
    for col_desc in parsed_col_desc_list:    
        fill_in_dict = {
            "[DESCRIPTIONS]": col_desc,
            "[FEATURES]": feature_desc
        }
        template = fill_in_templates(fill_in_dict, prompt_type_str)
        template_list.append(template)
        
    return template_list