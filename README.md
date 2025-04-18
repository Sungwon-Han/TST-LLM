# LLM-Guided Self-Supervised Tabular Learning With Task-Specific Pre-text Tasks #

This repo is the PyTorch code of our paper, which will be published on TMLR'25.

## Required packages ##
- python == 3.10.4
- torch == 1.11.0
- numpy == 1.22.3
- scikit-learn == 1.1.2
- scipy == 1.10.1
- pandas == 1.4.2
- openai == 0.28.0
- tqdm == 4.64.0
- faiss == 1.8.0
  
## To run the code ##
Run extract_columns.ipynb for discovering features via LLM. Need to put your own OpenAI API keys for running.  
The generation results from LLM will be saved in a directory 'LLM_results/diversity_5rules_40trials'.

Then, execute select_column.py for feature selection.
```
python select_column.py --select_k [number of features to select] --rule_dir [path for the saved discovered features]
```

Follow the below commands to train the encoder with selected features and evaluate with linear model.
1. Training
```
python train_embeddings.py --M [number of discovered features to use in training] 
```
2. Embedding Extraction
```
python extract_embeddings.py --M [number of discovered features to use in training]
```
3. Evaluation
```
python eval_embeddings.py --M [number of discovered features to use in training]
```
