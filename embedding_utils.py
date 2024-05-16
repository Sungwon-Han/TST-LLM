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

REGRESSION_DATASETS = ['insurance', 'housing', 'solution-mix', 'forest-fires', 'bike', 'crab', 'wine']

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
        '--backbone', type=str, default='mlp',
        help='Backbone'
    )
    parser.add_argument(
        '--loss', type=str, default='clip',
        help='Loss Objective'
    )
    parser.add_argument(
        '--with_ae', action='store_true',
        help='With Auto Encoding of Original Columns'
    )
    parser.add_argument(
        '--wo_selection', action='store_true',
        help='Withour Selection of Columns'
    )
    parser.add_argument(
        '--wo_selection_v2', action='store_true',
        help='Withour Selection (V2) of Columns'
    )
    parser.add_argument(
        '--with_negative', action='store_true',
        help='With Negative Columns'
    )
    parser.add_argument(
        '--no_diversity', action='store_true',
        help='With No Diversity Prompt Results'
    )
    parser.add_argument(
        '--with_random', action='store_true',
        help='With Random Columns'
    )
    parser.add_argument(
        '--with_diverse', action='store_true',
        help='With Diverse Columns'
    )
    parser.add_argument(
        '--allow_duplicates', action='store_true',
        help='Allow Duplicates'
    )
    parser.add_argument(
        '--embedding_dim', type=int, default=1024,
        help='Embedding Dimension'
    )
    parser.add_argument(
        '--hidden_dim', type=int, default=1024,
        help='Hidden Dimension'
    )
    parser.add_argument(
        '--proj_dim', type=int, default=128,
        help='Projector Dimension'
    )
    parser.add_argument(
        '--max_num_head', type=int, default=-1,
        help='Maximum number of heads'
    )
    parser.add_argument(
        '--shots', type=str, default='full',
        help='Number of Shots'
    )
    parser.add_argument(
        '--with_origin', action='store_true',
        help='Concatenate with Original Columns'
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
        help='Number of Epochs'
    )
    parser.add_argument(
        '--rule_dir', type=str, default='boosting-wR_5rules_40trials',
        help='The directory name to save results'
    )
    parser.add_argument(
        '--n_clusters', type=int, default=10,
        help='Number of clusters'
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
            model = KMeans(n_clusters = n_clusters, random_state=seed)
            model.fit(train_df[c].values.reshape(-1, 1))

            disc_train_df[c] = model.predict(train_df[c].values.reshape(-1, 1))
            disc_test_df[c] = model.predict(test_df[c].values.reshape(-1, 1))
        else:
            label_list = sorted(list(set(train_df[c]) | set(test_df[c])))
            disc_train_df[c] = [label_list.index(k) for k in train_df[c]]
            disc_test_df[c] = [label_list.index(k) for k in test_df[c]]
    
    return disc_train_df, disc_test_df

def get_indicator_df(
    loss_type,
    dataset,
    seed,
    n_clusters,
    proxy_path,
    df_path,
    allow_duplicates=False,
):
    if loss_type in ['supcon', 'supcon_v3']:
        DISCRETIZE = False
        
        if 'negative' in df_path.name:
            path_detail = 'disc_dataframe_negative'
        if 'random' in df_path.name:
            path_detail = 'disc_dataframe_random'
        elif 'diverse' in df_path.name:
            path_detail = 'disc_dataframe_diverse'
        elif 'wo_selection_v2' in df_path.name:
            path_detail = 'disc_dataframe_wo_selection_v2'
        elif 'wo_selection' in df_path.name:
            path_detail = 'disc_dataframe_wo_selection'
        elif 'no_diversity' in df_path.name:
            path_detail = 'disc_dataframe_no_diversity'
        else:
            path_detail = 'disc_dataframe'
            
        if allow_duplicates:
            path_detail += '_allow_duplicates'
            
        disc_save_path = proxy_path / path_detail
            
        disc_save_path.mkdir(exist_ok=True, parents=True)
        if DISCRETIZE:
            print(f'Dicretizing {dataset} {seed}')
            if 'diverse' in df_path.name:
                new_train_df = pd.read_csv(df_path / f'{dataset}_{seed}_10000_train.csv')
                new_test_df = pd.read_csv(df_path / f'{dataset}_{seed}_10000_test.csv')
                
                if not allow_duplicates:
                    new_train_df = new_train_df.T.drop_duplicates(keep='first').T
                    
                new_test_df = new_test_df[new_train_df.columns]
            else:
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
        else:
            ind_train_df = pd.read_csv(disc_save_path / f'{dataset}_{seed}_train.csv')
            ind_test_df = pd.read_csv(disc_save_path / f'{dataset}_{seed}_test.csv')

    elif loss_type in ['clip', 'supcon_v2']:
        ind_train_df = pd.read_csv(df_path / f'{dataset}_{seed}_train.csv')
        ind_test_df = pd.read_csv(df_path / f'{dataset}_{seed}_test.csv')
        
        
    else:
        assert(0)
        
    return ind_train_df, ind_test_df

class LifeDataset(torch.utils.data.Dataset):
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
    ds = LifeDataset(X, y, indicators)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=8
    )
    
    return dl

class MLPProto(nn.Module):
    def __init__(self, in_features, embedding_dim, hidden_dim=1024):
        super(MLPProto, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, inputs):
        embeddings = self.encoder(inputs)
        return embeddings
    
class FTTransformer(nn.Module):
    def __init__(self, in_features, hidden_dim, n_cat, n_num, enc=None):
        super().__init__()
        self.n_cat = n_cat
        self.n_num = n_num
        self.hidden_dim = hidden_dim
        
        n_cat_options = in_features - n_num
        self.padding_idx = n_cat_options
        
        self.catnum = []
        for c in enc.categories_:
            self.catnum.append(len(c))

        self.cls = nn.Parameter(torch.ones(1, hidden_dim), requires_grad=True)
        
        self.cat_w = nn.Embedding(
            num_embeddings=n_cat_options + 1,
            embedding_dim=hidden_dim,
            padding_idx=self.padding_idx,
        )
        self.cat_b = nn.Parameter(torch.randn(self.n_cat, hidden_dim), requires_grad=True)
        
        self.num_w = nn.Parameter(torch.randn(self.n_num, hidden_dim), requires_grad=True)
        self.num_b = nn.Parameter(torch.randn(self.n_num, hidden_dim), requires_grad=True)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.te_layer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
    def forward(self, x):
        bsz, n_columns = x.shape
        
        x_num = x[:, :self.n_num]
        x_cat = x[:, self.n_num:].int()
                
        cls = self.cls.unsqueeze(0).tile(bsz, 1, 1)
        
        cur_start = 0
        cur_end = 0
        
        x_cat_index = []
        for i, cn in enumerate(self.catnum):
            cur_end += cn
            xci = []
            for b in x_cat[:, cur_start:cur_end]:
                tg = b.nonzero(as_tuple=True)[0]
                if len(tg) == 0:
                    tg = torch.tensor([self.padding_idx], device=b.device)
                xci.append(tg)

            xci = torch.stack(xci, dim=0)
            x_cat_index.append(xci)

            cur_start += cn

        x_cat_index = torch.cat(x_cat_index, dim=1)
        
        cat_emb = self.cat_w(x_cat_index) + self.cat_b.unsqueeze(dim=0).tile(bsz, 1, 1)
            
        num_emb = torch.mul(
            x_num.unsqueeze(dim=-1).tile(1, 1, self.hidden_dim),
            self.num_w.unsqueeze(dim=0).tile(bsz, 1, 1)
        ) +  self.num_b.unsqueeze(dim=0).tile(bsz, 1, 1)
        
        emb = torch.cat((cls, cat_emb, num_emb), dim=1)
        
        out = self.te_layer(emb)
        out = out[:, 0, :]
        
        return out
    
class LifeMLP(nn.Module):
    def __init__(
        self,
        model_type,
        loss_type,
        in_features,
        n_cat,
        n_num,
        embedding_dim=1024,
        hidden_dim=1024,
        proj_dim=128,
        ind_dim=-1,
        max_num_head=-1,
        enc=None,
        with_ae=False,
    ):
        super(LifeMLP, self).__init__()
        self.with_ae = with_ae
        self.loss_type = loss_type
        
        if model_type == 'mlp':
            self.encoder = MLPProto(in_features, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        elif model_type == 'tf':
            self.encoder = FTTransformer(in_features, hidden_dim, n_cat, n_num, enc=enc)
        else:
            assert(0)
            
        if loss_type == 'supcon':
            self.head = nn.Linear(embedding_dim, proj_dim)
        elif loss_type == 'supcon_v2':
            self.head = nn.Linear(embedding_dim, proj_dim)
        elif loss_type == 'supcon_v3':
            if max_num_head == -1:
                num_head = ind_dim
            else:
                num_head = min(max_num_head, ind_dim)
                
            self.head = nn.ModuleList(
                [nn.Linear(embedding_dim, proj_dim) for i in range(num_head)]
            )
        elif loss_type == 'clip':
            self.head = nn.Linear(embedding_dim, ind_dim)
        else:
            assert(0)
            
        if with_ae:
            self.ae_head = nn.Linear(embedding_dim, in_features)

    def forward(self, x):
        feat = self.encoder(x)
        if self.loss_type == 'supcon_v3':
            new_feat = [F.normalize(h(feat), dim=1) for h in self.head]
        else:
            new_feat = F.normalize(self.head(feat), dim=1)
            
        if self.with_ae:
            ae_feat = F.normalize(self.ae_head(feat), dim=1)
        else:
            ae_feat = None
        
        return new_feat, ae_feat
    
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
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
    
class ClipLoss(nn.Module):
    def __init__(self):
        super(ClipLoss, self).__init__()

    def forward(self, features, labels):
        sim_matrix = torch.mm(features, labels.T)
        one_diag = torch.eye(sim_matrix.shape[0], device=sim_matrix.device)
        
        sim_matrix = torch.pow(sim_matrix - one_diag, exponent=2)
        loss = torch.sum(sim_matrix)

        return loss
    
def get_model(
    model_type,
    loss_type,
    in_features,
    n_cat,
    n_num,
    ind_dim=-1,
    embedding_dim=1024,
    hidden_dim=1024,
    proj_dim=128,
    max_num_head=-1,
    enc=None,
    with_ae=False,
):
    model = LifeMLP(
        model_type=model_type,
        loss_type=loss_type,
        in_features=in_features,
        n_cat=n_cat,
        n_num=n_num,
        ind_dim=ind_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        proj_dim=proj_dim,
        max_num_head=max_num_head,
        enc=enc,
        with_ae=with_ae,
    )
    
    supcon_criterion = SupConLoss()
    clip_criterion = ClipLoss()
    
    return model, supcon_criterion, clip_criterion

def train(
    model,
    train_loader,
    loss_type,
    supcon_criterion,
    clip_criterion,
    seed,
    learning_rate,
    iters,
    with_ae=False,
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
    
    if loss_type == 'supcon_v2':
        indicator_matrix = copy.deepcopy(train_loader.dataset.indicators)
    
    n = 0
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            if n in [0, 5, 10, 50, 100, 500]:
                torch.save(model.state_dict(), model_save_path / f'model_iter_{n}.pt')
            
            if remainder != 0 and epoch == epochs - 1 and i == remainder:
                break
                
            if loss_type == 'supcon_v2':
                if n % 100 == 0:
                    best_score = -1
                    best_k = -1
                    best_kmeans = None
                    for k in [5, 10, 15, 20]:
                        d = indicator_matrix.shape[1]
                        kmeans = faiss.Kmeans(d, k, niter=20, nredo=1, verbose=False, seed=n, gpu=1)
                        kmeans.train(indicator_matrix)
                        _, cluster_labels = kmeans.index.search(indicator_matrix, 1)
                        cluster_labels = cluster_labels.squeeze(axis=-1)
                        score = silhouette_score(indicator_matrix, cluster_labels)

                        if score >= best_score:
                            best_score = score
                            best_k = k
                            best_kmeans = kmeans
                
            inputs, _, indicator = batch

            inputs = inputs.cuda().float()

            bsz = inputs.shape[0]
            features, ae_features = model(inputs)
            
            if loss_type == 'supcon':
                features = features.unsqueeze(dim=1)
                num_indicator = indicator.shape[1]
                indicator_idx = random.sample(range(num_indicator), 1)[0]
                labels = indicator[:, indicator_idx]
                labels = labels.cuda()
                loss = supcon_criterion(features=features, labels=labels)
                
            elif loss_type == 'supcon_v2':
                _, labels = kmeans.index.search(indicator, 1)
                labels = labels.squeeze(axis=-1)
                labels = torch.tensor(labels).cuda()
                
                features = features.unsqueeze(dim=1)
                loss = supcon_criterion(features=features, labels=labels)
                
            elif loss_type == 'supcon_v3':
                supcon_loss_list = []
                for i, feature in enumerate(features):
                    feature = feature.unsqueeze(dim=1)
                    label = indicator[:, i]
                    label = label.cuda()
                    loss = supcon_criterion(features=feature, labels=label)
                    supcon_loss_list.append(loss)
                loss = torch.stack(supcon_loss_list).mean()
                
            elif loss_type == 'clip':
                labels = F.normalize(indicator.float(), dim=1)
                labels = labels.cuda()
                loss = clip_criterion(features=features, labels=labels)
                
            ae_labels = F.normalize(inputs, dim=1)
            
            
            if with_ae:
                ae_loss = clip_criterion(features=ae_features, labels=ae_labels)
                loss = loss + ae_loss

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