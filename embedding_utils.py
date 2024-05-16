import argparse
import copy
import numpy as np
import math
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import silhouette_score

import torch
import torch.nn as nn
import torch.nn.functional as F

import optimization_utils as opt

import faiss

REGRESSION_DATASETS = ['forest-fires', 'housing', 'insurance', 'solution-mix']

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(
        self
    ):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(
        self,
        val: float,
        n: int=1
    ):
        """
        Update the values
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def parse_arguments(
    return_default: bool=False,
) -> object:
    """
    Parse the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu', type=int, default=0,
         help='GPU index to use'
    )
    parser.add_argument(
        '--seed', type=int, default=-1,
         help='Random seed'
    )
    parser.add_argument(
        '--dataset', type=str, default='',
        help='Dataset'
    )
    parser.add_argument(
        '--n_clusters', type=int, default=10,
        help='Number of clusters'
    )
    parser.add_argument(
        '--hidden_dim', type=int, default=1024,
        help='Size of hidden dimension in the model'
    )
    parser.add_argument(
        '--proj_dim', type=int, default=128,
        help='Size of projected dimension'
    )
    parser.add_argument(
        '--M', type=int, default=20,
        help='Maximum number of discovered features to use in learning'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='Batch Size'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=1e-4,
        help='Learning Rate'
    )
    parser.add_argument(
        '--iterations', type=int, default=1000,
        help='Number of iterations'
    )
    parser.add_argument(
        '--data_dir', type=str, default='data',
        help='The directory name that contains data'
    )
    parser.add_argument(
        '--proxy_dir', type=str, default='proxy',
        help='The directory name to save proxy results'
    )
    parser.add_argument(
        '--result_dir', type=str, default='results',
        help='The directory name to save results'
    )

    if return_default:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
        
    return args

def check_numerical(s):
    assert(type(s) == pd.Series)
    numerical = True
    try:
        for temp in set(s):
            float(temp)
    except:
        numerical = False
        
    return numerical

def discretize_cols(
    train_df,
    test_df,
    seed,
    n_clusters,
):
    disc_train_df = copy.deepcopy(train_df)
    disc_test_df = copy.deepcopy(test_df)
    
    for c in tqdm(train_df.columns):
        numerical = check_numerical(train_df[c])

        if numerical:
            model = KMeans(n_clusters = n_clusters, random_state=seed, n_init=10)
            model.fit(train_df[c].values.reshape(-1, 1))

            disc_train_df[c] = model.predict(train_df[c].values.reshape(-1, 1))
            disc_test_df[c] = model.predict(test_df[c].values.reshape(-1, 1))
        else:
            label_list = sorted(list(set(train_df[c]) | set(test_df[c])))
            disc_train_df[c] = [label_list.index(k) for k in train_df[c]]
            disc_test_df[c] = [label_list.index(k) for k in test_df[c]]
    
    return disc_train_df, disc_test_df

def get_indicator_df(
    dataset,
    seed,
    n_clusters,
    proxy_path,
    df_path,
):
    path_detail = 'disc_dataframe'
    disc_save_path = proxy_path / path_detail
    disc_save_path.mkdir(exist_ok=True, parents=True)
    
    print(f'Dicretizing {dataset} {seed}')

    new_train_df = pd.read_csv(df_path / f'{dataset}_{seed}_train.csv')
    new_test_df = pd.read_csv(df_path / f'{dataset}_{seed}_test.csv')

    new_train_df, new_test_df = opt.fill_missing(new_train_df, new_test_df)
    ind_train_df, ind_test_df = discretize_cols(
        new_train_df,
        new_test_df,
        seed=seed,
        n_clusters=n_clusters
    )

    ind_train_df.to_csv(disc_save_path / f'{dataset}_{seed}_train.csv', index=False)
    ind_test_df.to_csv(disc_save_path / f'{dataset}_{seed}_test.csv', index=False)

        
    return ind_train_df, ind_test_df

class TstLLMDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, indicators=None):
        assert(type(X) == np.ndarray)
        
        self.X = X
        self.y = y
        self.indicators = indicators
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if type(self.indicators) == pd.DataFrame:
            indicator = self.indicators.values
            return self.X[idx], self.y[idx], indicator[idx]
        elif type(self.indicators) == np.ndarray:
            indicator = self.indicators
            return self.X[idx], self.y[idx], indicator[idx]
        else:
            assert(self.indicators == None)
            return self.X[idx], self.y[idx]

def get_loader(
    X,
    y,
    indicators=None,
    batch_size=128,
    shuffle=False,
):
    ds = TstLLMDataset(X, y, indicators)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=8
    )
    
    return dl

class MLPEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim=1024):
        super(MLPEncoder, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, inputs):
        embeddings = self.encoder(inputs)
        return embeddings
    
class TstLLM(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_dim=1024,
        proj_dim=128,
        M=-1
    ):
        super(TstLLM, self).__init__()
        assert(M > 0)
        
        self.encoder = MLPEncoder(in_features, hidden_dim=hidden_dim)

        self.head = nn.ModuleList(
            [nn.Linear(hidden_dim, proj_dim) for i in range(M)]
        )

    def forward(self, x):
        feat = self.encoder(x)
        new_feat = [F.normalize(h(feat), dim=1) for h in self.head]

        return new_feat, None
    
class SupConLoss(nn.Module):
    """
    Reference: https://github.com/HobbitLong/SupContrast
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR.
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).cuda().float()
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        one_matrix = torch.ones_like(mask_pos_pairs).float()
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, one_matrix, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
def get_model(
    in_features,
    hidden_dim=1024,
    proj_dim=128,
    M=-1,
):
    model = TstLLM(
        in_features=in_features,
        hidden_dim=hidden_dim,
        proj_dim=proj_dim,
        M=M,
    )
    
    criterion = SupConLoss()

    return model, criterion

def train(
    model,
    train_loader,
    criterion,
    seed,
    learning_rate,
    iters,
    model_save_path=None,
):
    train_iterator = iter(train_loader)
    epochs = math.ceil(iters / len(train_loader))
    remainder = iters % len(train_loader)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    model.train()
    pbar = tqdm(range(iters))
    train_losses = AverageMeter()
    
    n = 0
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            if n in [0, 5, 10, 50, 100, 500]:
                torch.save(model.state_dict(), model_save_path / f'model_iter_{n}.pt')
            
            if remainder != 0 and epoch == epochs - 1 and i == remainder:
                break
                
            inputs, _, indicator = batch

            inputs = inputs.cuda().float()

            bsz = inputs.shape[0]
            features, ae_features = model(inputs)

            supcon_loss_list = []
            for i, feature in enumerate(features):
                feature = feature.unsqueeze(dim=1)
                label = indicator[:, i]
                label = label.cuda()
                loss = criterion(features=feature, labels=label)
                supcon_loss_list.append(loss)
            loss = torch.stack(supcon_loss_list).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.update(loss.item(), bsz)

            pbar.update(1)
            desc = f'Train Loss: {train_losses.avg:.3f}'
            pbar.set_description(desc)
            
            n += 1
        
        scheduler.step()
    pbar.close()

    return model

def extract_embedding(
    model,
    dataloader,
):
    if model != None:
        model.eval()
        
    with torch.no_grad():
        feature_list = []
        label_list = []
        input_list = []
        for batch in tqdm(dataloader):
            inputs, labels = batch
            input_list.append(inputs)
            label_list.append(labels)
            
            if model != None:
                inputs = inputs.cuda().float()
                features = model.encoder(inputs)
                feature_list.append(features.cpu())
            
        input_list = torch.cat(input_list, dim=0).numpy()
        label_list = torch.cat(label_list, dim=0).numpy()
        if model != None:
            feature_list = torch.cat(feature_list, dim=0).numpy()

    return input_list, feature_list, label_list