# calculate_metrics.py




import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, roc_auc_score
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy import sparse
import anndata as ad
import pandas as pd


def plot_pearson_correlation(original_data, generated_data, pearson_data, save_path,ty = 0):
    
    fig, ax = plt.subplots(figsize=(7, 6))  # Create figure and axes

   
    norm = Normalize(vmin=min(original_data.min(), generated_data.min()),
                     vmax=max(original_data.max(), generated_data.max()))
    cmap = plt.get_cmap("YlGn_r")  
    sm = ScalarMappable(cmap=cmap, norm=norm)


    scatter = ax.scatter(original_data.flatten(), generated_data.flatten(),
                         alpha=0.7, s=1, c=original_data.flatten(), cmap=cmap, marker='s')


    min_val = min(original_data.min(), generated_data.min())
    max_val = max(original_data.max(), generated_data.max())
    ax.plot([min_val, max_val], [min_val, max_val], '--', color='gray', alpha=0.5, label='$y=x$')
    
    
    if ty == 0:
        title_text = f'ATAC > RNA (test set) (r = {pearson_data:.2f})'
    elif ty == 1:
        title_text = f'ADT > RNA (test set) (r = {pearson_data:.2f})'
    elif ty == 2:
        title_text = f'RNA > ADT (test set) (r = {pearson_data:.2f})'

    ax.set_title(title_text, fontsize=14, pad=12)


    ax.set_xlabel('Original norm counts (log)', fontsize=12)
    ax.set_ylabel('Inferred norm counts (log)', fontsize=12)


    ax.set_aspect(0.75, adjustable='box') 

    ax.legend(loc='upper left', frameon=False)


    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()


def plot_roc_curve(original_data, generated_data, save_path):
    
    
    fpr, tpr, thresholds = roc_curve(original_data.flatten(), generated_data.flatten())
    auc_value = roc_auc_score(original_data.flatten(), generated_data.flatten())

    plt.figure(figsize=(7, 6))

    plt.plot(fpr, tpr, color='#1f77b4', lw=2)
    plt.title(f'RNA > ATAC (AUROC={auc_value:.2f})', fontsize=14, pad=12)

    plt.xlabel('False positive rate', fontsize=12)
    plt.ylabel('True positive rate', fontsize=12)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks(np.arange(0.0, 1.1, 0.2))
    plt.yticks(np.arange(0.0, 1.1, 0.2))


    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()


def visualize_metrics(RNA_data, G_RNA, ATAC_data, G_ATAC, pearson_data, save_dir):
   
   
    RNA_data_np = RNA_data.cpu().numpy() if torch.is_tensor(RNA_data) else RNA_data
    G_RNA_np = G_RNA.cpu().numpy() if torch.is_tensor(G_RNA) else G_RNA
    ATAC_data_np = ATAC_data.cpu().numpy() if torch.is_tensor(ATAC_data) else ATAC_data
    G_ATAC_np = G_ATAC.cpu().numpy() if torch.is_tensor(G_ATAC) else G_ATAC

    os.makedirs(save_dir, exist_ok=True)

    pearson_save_path = os.path.join(save_dir, "pearson.png")
    auroc_save_path = os.path.join(save_dir, "AUROC.png")
    ATAC_data_np_binary = (ATAC_data_np > 0.1).astype(int)

    plot_pearson_correlation(
        RNA_data_np, 
        G_RNA_np, 
        pearson_data,
        pearson_save_path,
        ty=0
    )

    plot_roc_curve(
        ATAC_data_np_binary, 
        G_ATAC_np, 
        auroc_save_path
    )


def visualize_metrics_ADT(RNA_data, G_RNA, ADT_data, G_ADT, pearson_data, pearson_data2, save_dir):


    RNA_data_np = RNA_data.cpu().numpy() if torch.is_tensor(RNA_data) else RNA_data
    G_RNA_np = G_RNA.cpu().numpy() if torch.is_tensor(G_RNA) else G_RNA
    ADT_data_np = ADT_data.cpu().numpy() if torch.is_tensor(ADT_data) else ADT_data
    G_ADT_np = G_ADT.cpu().numpy() if torch.is_tensor(G_ADT) else G_ADT


    os.makedirs(save_dir, exist_ok=True)


    pearson_save_path = os.path.join(save_dir, "pearson.png")
    pearson_adt_save_path = os.path.join(save_dir, "pearson_adt.png")
    

    plot_pearson_correlation(
        RNA_data_np,  
        G_RNA_np,  
        pearson_data,
        pearson_save_path,
        ty=1
    )

    plot_pearson_correlation(
        ADT_data_np,  
        G_ADT_np,  
        pearson_data2,
        pearson_adt_save_path,
        ty=2
    )


def calculate_auc(ATAC_data, G_ATAC):
    ATAC_data_np = ATAC_data.cpu().numpy() if torch.is_tensor(ATAC_data) else ATAC_data
    G_ATAC_np = G_ATAC.cpu().numpy() if torch.is_tensor(G_ATAC) else G_ATAC

    threshold = 0.1

    ATAC_data_np = (ATAC_data_np > threshold).astype(int)

    auc = roc_auc_score(ATAC_data_np.flatten(), G_ATAC_np.flatten())

    return auc


def calculate_pearson_correlation(original_data, generated_data):

    correlation, _ = pearsonr(original_data.flatten(), generated_data.flatten())
    return correlation


def tensor2adata(x_tensor, adata_obs=None, val=1e-4, feature_names=None,
                 output_path=None, filename=None):
  
  
    x_cpu = x_tensor.detach().cpu().numpy()
    x_filtered = np.where(np.abs(x_cpu) < val, 0, x_cpu)
    x_sparse = sparse.csr_matrix(x_filtered)


    if adata_obs is not None:
        if not isinstance(adata_obs, pd.DataFrame):
            raise TypeError("adata_obs must be pandas DataFrame")
        if len(adata_obs) != x_sparse.shape[0]:
            raise ValueError(f"obs rows ({len(adata_obs)}) != tensor rows ({x_sparse.shape[0]})")


    adata = ad.AnnData(
        X=x_sparse,
        obs=adata_obs.copy() if adata_obs is not None else None,
        var=pd.DataFrame(index=feature_names if feature_names else
        [f"feature_{i}" for i in range(x_sparse.shape[1])])
    )


    if adata_obs is not None:
        adata.obs_names = adata_obs.index


    if output_path is not None and filename is not None:
        save_path = os.path.join(output_path, filename)
        adata.write_h5ad(save_path)
        print(f"Saved to: {save_path}")

    return adata

