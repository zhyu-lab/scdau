from torch.utils.data import DataLoader

import pandas as pd
import torch
import numpy as np
import random
import os
import scanpy as sc
import episcanpy.api as epi
import scipy.sparse as sp
from scipy.stats.mstats import gmean
import matplotlib
from sklearn.feature_extraction.text import TfidfTransformer
import anndata as ad
from scipy.sparse import csr_matrix

def leiden_clustering_split(adata, test_cluster=1):



    if 'X_pca' not in adata.obsm:
        sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)

    sc.tl.leiden(adata, resolution=1.0, key_added="leiden_clusters")

    clusters = adata.obs["leiden_clusters"].astype(int)
    test_idx = clusters == test_cluster
    train_idx = ~test_idx

    return adata[train_idx], adata[test_idx]


def TFIDF(count_mat):
    """
    TF-IDF transformation for matrix.
    """
    count_mat = count_mat.T
    divide_title = np.tile(np.sum(count_mat, axis=0), (count_mat.shape[0], 1))
    nfreqs = count_mat / divide_title
    multiply_title = np.tile(np.log(1 + count_mat.shape[1] / np.sum(count_mat, axis=1)).reshape(-1, 1), (1, count_mat.shape[1]))
    tfidf_mat = sp.csr_matrix(np.multiply(nfreqs, multiply_title)).T
    return tfidf_mat, divide_title, multiply_title

def get_input_dimensions_ADT(RNA_data, ADT_data):
    RNA_input_dim = len([i for i in RNA_data.var['highly_variable'] if i])
    ADT_input_dim = ADT_data.X.shape[1]

    return RNA_input_dim, ADT_input_dim
    
def get_input_dimensions(RNA_data, ATAC_data):
    
    
    
    
    RNA_input_dim = len([i for i in RNA_data.var['highly_variable'] if i])
    ATAC_input_dim = ATAC_data.X.shape[1]


    chrom_list = calculate_chrom_list_from_h5ad(ATAC_data)

    return RNA_input_dim, ATAC_input_dim, chrom_list



def read_ae_input_dimensions(file_path):

    with open(file_path) as f:
      
        line = f.readline().rstrip()
        RNA_input_dim = int(line)
        line = f.readline().rstrip()
        ATAC_input_dim = int(line)

        line = f.readline().rstrip()
        fields = line.split(',')
        chrom_list = []
        for j in fields:
            chrom_list.append(int(j))


    return RNA_input_dim, ATAC_input_dim, chrom_list



def RNA_data_preprocessing(RNA_data, normalize_total=True, log1p=True, use_hvg=True, n_top_genes=3000):
    """
    Preprocessing for RNA data, using scanpy.
    """

    RNA_data.var_names_make_unique()

    if normalize_total:
        sc.pp.normalize_total(RNA_data)

    if log1p:
        sc.pp.log1p(RNA_data)

    if use_hvg:

        sc.pp.highly_variable_genes(RNA_data, n_top_genes=n_top_genes)
        RNA_data = RNA_data[:, RNA_data.var['highly_variable']]


    return RNA_data


def ATAC_data_preprocessing(ATAC_data, binary_data=True, filter_features=True, fpeaks=0.005, tfidf=True, normalize=True):
    
    
    if binary_data:
        epi.pp.binarize(ATAC_data)


    if filter_features:
        initial_features = ATAC_data.var_names.tolist()
        epi.pp.filter_features(ATAC_data, min_cells=np.ceil(fpeaks * ATAC_data.shape[0]))
        filtered_features = set(initial_features) - set(ATAC_data.var_names)
        filtered_features_indices = [initial_features.index(f) + 1 for f in filtered_features]
    else:
        filtered_features_indices = []

    
    if tfidf:
        
        transformer = TfidfTransformer()
        ATAC_data.X = transformer.fit_transform(ATAC_data.X)

    if normalize:
        max_temp = ATAC_data.X.max()
        ATAC_data.X /= max_temp
    return ATAC_data, filtered_features_indices


def preprocess_atac_data(file_path):

    ATAC_data = sc.read_h5ad(file_path)
    print('preprocessing atac data')
    ATAC_data, _ = ATAC_data_preprocessing(ATAC_data)


    return ATAC_data


def calculate_chrom_list_from_h5ad(adata):

    chrom_list = []
    last_chrom = ''
    

    for chrom in adata.var['chrom']:

        if '-' in chrom:
            chrom_name = chrom.split('-')[0]
        else:
            chrom_name = chrom


        if chrom_name.startswith('chr'):
            if chrom_name != last_chrom:
                chrom_list.append(1)
                last_chrom = chrom_name
            else:
                chrom_list[-1] += 1
        else:

            if chrom_list:
                chrom_list[-1] += 1
            else:
                chrom_list.append(1)
                last_chrom = chrom_name

    print(f"Chrom list: {chrom_list}")
    return chrom_list


def CLR_transform(ADT_data):
    """
    Centered log-ratio transformation for ADT data.

    Parameters
    ----------
    ADT_data: Anndata
        ADT anndata for processing.

    Returns
    ----------
    ADT_data_processed: Anndata
        ADT data with CLR transformation preprocessed.

    gmean_list
        vector of geometric mean for ADT expression of each cell.
    """
    ADT_matrix = ADT_data.X.todense()
    gmean_list = []
    for i in range(ADT_matrix.shape[0]):
        temp = []
        for j in range(ADT_matrix.shape[1]):
            if not ADT_matrix[i, j] == 0:
                temp.append(ADT_matrix[i, j])
        gmean_temp = gmean(temp)
        gmean_list.append(gmean_temp)
        for j in range(ADT_matrix.shape[1]):
            if not ADT_matrix[i, j] == 0:
                ADT_matrix[i, j] = np.log(ADT_matrix[i, j] / gmean_temp)
    ADT_data_processed = ad.AnnData(csr_matrix(ADT_matrix), obs=ADT_data.obs, var=ADT_data.var)
    return ADT_data_processed, gmean_list
    
    

def preprocess_rna_data(file_path):
    """
    Preprocessing RNA data and split into train, val, and test sets.
    """
    # Step 1: Read RNA data
    RNA_data = sc.read_h5ad(file_path)
    print('preprocessing rna data')

    # Step 2: Preprocess the data
    RNA_data = RNA_data_preprocessing(RNA_data)

    return RNA_data


def preprocess_adt_data(file_path):

    ADT_data = sc.read_h5ad(file_path)
    print('preprocessing adt data')
    ADT_data  = CLR_transform(ADT_data)[0]
    return ADT_data
    
    
def split_dataset(data, seed=0):
    
    
    temp = [i for i in range(data.shape[0])]

    random.shuffle(temp, random.seed(seed))


    validation_count = int(0.1 * len(temp))

    validation_id = temp[:validation_count]

    train_id = temp[validation_count:]


    return train_id, validation_id



def create_data_tensors(data, device=torch.device('cuda')):


    train_id, val_id = split_dataset(data, seed=0)
    data_train, data_val = data[train_id], data[val_id]
    tensor_train, tensor_val = convert_to_tensors(data_train, data_val, device)

    return tensor_train, tensor_val


def create_tensor(data, device=torch.device('cuda')):

    tensor_train = torch.tensor(data, device=device, dtype=torch.float32)

    return tensor_train

def convert_to_tensors(data_train, data_val, device=torch.device('cuda')):
    """
    Convert the AnnData object to PyTorch tensors.
    """
    tensor_train = torch.tensor(data_train, dtype=torch.float32).to(device)
    tensor_val = torch.tensor(data_val, dtype=torch.float32).to(device)

    return tensor_train, tensor_val


def infinite_generator(data, batch_size, shuffle=True, drop_last=False):
    while True:
        train_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        for batch in train_loader:
            yield batch, {}
