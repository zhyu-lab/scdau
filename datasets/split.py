import scanpy as sc
import anndata as ad
import pandas as pd
import torch
import numpy as np
import random
import configs



def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def five_fold_split_dataset(
        RNA_data,
        ATAC_data,
        seed=19193
):
    if not seed is None:
        setup_seed(seed)

    temp = [i for i in range(len(RNA_data.obs_names))]
    random.shuffle(temp)

    id_list = []

    test_count = int(0.2 * len(temp))

    for i in range(5):
        test_id = temp[:test_count]
        train_id = temp[test_count:]
        temp.extend(test_id)
        temp = temp[test_count:]

        id_list.append([train_id, test_id])

    return id_list


def convert_to_tensors(data_train, data_test):
    """
    Convert the AnnData object to PyTorch tensors.
    """
    tensor_train = torch.tensor(data_train.X.toarray(), dtype=torch.float32).to(configs.DEVICE)
    tensor_test = torch.tensor(data_test.X.toarray(), dtype=torch.float32).to(configs.DEVICE)

    return tensor_train, tensor_test
    
    
def create_data_loaders_ADT(RNA_data, ADT_data, fold_idx=1):
    # Split dataset into training, validation, and test sets
    split_ids = five_fold_split_dataset(RNA_data, ADT_data, seed=19193)
    train_id, test_id = split_ids[fold_idx]
    train_id_r = train_id.copy()
    train_id_a = train_id.copy()
    test_id_r = test_id.copy()
    test_id_a = test_id.copy()

    RNA_data_train,  RNA_data_test = RNA_data[train_id_r],  RNA_data[test_id_r]
    ADT_data_train,  ADT_data_test = ADT_data[train_id_a],  ADT_data[test_id_a]
    

    return RNA_data_train, RNA_data_test, ADT_data_train, ADT_data_test
    
    
def create_data_loaders(RNA_data, ATAC_data, fold_idx=1):
    # Split dataset into training and test sets
    split_ids = five_fold_split_dataset(RNA_data, ATAC_data, seed=19193)
    train_id, test_id = split_ids[fold_idx]

    RNA_data_train, RNA_data_test = RNA_data[train_id], RNA_data[test_id]
    ATAC_data_train, ATAC_data_test = ATAC_data[train_id], ATAC_data[test_id]


    return RNA_data_train, RNA_data_test, ATAC_data_train, ATAC_data_test