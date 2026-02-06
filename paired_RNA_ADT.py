"""
Train a generative model based on DDIMs for generating paired single-cell data.
"""
import os
import argparse
import numpy as np
import scanpy as sc
import random
import torch
from torch.optim import Adam
import warnings
import torch.distributed as dist
import umap
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

from diffusion import dist_util, logger
from diffusion.resample import create_named_schedule_sampler
from utils.script_util import (
    diffusion_defaults,
    create_dae,
    create_diffusion,
    args_to_dict,
    add_dict_to_argparser
)
import einops
from utils.script_util import *
from utils.train_util import TrainLoop
from utils.training import (
    train_rna_ae_all,
    train_atac_ae_all,
    train_dae_all,
    train_adt_ae_all
)
from datasets.split import (
    five_fold_split_dataset,
    convert_to_tensors,
    create_data_loaders_ADT,
    create_data_loaders
)
from datasets.loader import (
    preprocess_rna_data,
    preprocess_atac_data,
    preprocess_adt_data,
    get_input_dimensions,
    get_input_dimensions_ADT,
    create_data_tensors,
    infinite_generator,
    leiden_clustering_split,
    create_tensor
)
from models.ed import AE_RNA, AE_ATAC, AE_ADT
from models.calculate_metrics import (
    calculate_pearson_correlation,
    calculate_auc,
    visualize_metrics,
    visualize_metrics_ADT,
    tensor2adata
)
from models.cross_cond_unet import (
    CrossConditionedUNet,
    train_cross_conditioned_unet
)
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    homogeneity_score
)

print(f"Using GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    warnings.filterwarnings("ignore")
    parser = create_argparser()
    args = parser.parse_args()
    setup_seed(args.seed)

    dist_util.setup_dist()
    log_dir = args.log_dir

    ae_dir = os.path.join(log_dir, 'ae')
    os.makedirs(ae_dir, exist_ok=True)
    logger.configure(dir=ae_dir)

    logger.log("loading dataset...")

    RNA_data = preprocess_rna_data(args.rna_path)
    ADT_data = preprocess_adt_data(args.adt_path)
    RNA_input_dim, ADT_input_dim = get_input_dimensions_ADT(RNA_data, ADT_data)

    file_o = open(ae_dir + '/paras.txt', 'w')
    np.savetxt(file_o, np.c_[RNA_input_dim], fmt='%d')
    np.savetxt(file_o, np.c_[ADT_input_dim], fmt='%d')

    file_o.close()

    rna_train, rna_test, adt_train, adt_test = create_data_loaders_ADT(RNA_data, ATAC_data,fold_idx=args.fold)
    
    rna_train_tensor = create_tensor(rna_train.X.toarray(), dist_util.dev())
    rna_test_tensor = create_tensor(rna_test.X.toarray(), dist_util.dev())
    adt_train_tensor = create_tensor(atac_train.X.toarray(), dist_util.dev())
    adt_test_tensor = create_tensor(atac_test.X.toarray(), dist_util.dev())
    

    # Step 1: load or train autoencoders
    rna_ae = AE_RNA(RNA_input_dim, args.latent_size)
    rna_ae.to(dist_util.dev())
    if len(args.rna_ae_path) > 0:
        logger.log("loading RNA autoencoder...")
        rna_ae.load_state_dict(torch.load(args.rna_ae_path))
    else:
        logger.log("training RNA autoencoder...")
        optimizer_rna_ae = Adam(rna_ae.parameters(), lr=args.rna_ae_lr)
        train_rna_ae_all(rna_ae,
                         rna_train_tensor,
                         optimizer_rna_ae,
                         ae_dir,
                         train_epoch=args.ae_epochs,
                         batch_size=args.ae_batch_size)

    adt_ae = AE_ADT(ADT_input_dim, args.latent_size)
    adt_ae.to(dist_util.dev())
    if len(args.adt_ae_path) > 0:
        logger.log("loading ADT autoencoder...")
        adt_ae.load_state_dict(torch.load(args.adt_ae_path))
    else:
        logger.log("training ADT autoencoder...")
        optimizer_adt_ae = Adam(adt_ae.parameters(), lr=args.adt_ae_lr)
        train_adt_ae_all(adt_ae,
                          adt_train_tensor,
                          optimizer_adt_ae,
                          ae_dir,
                          train_epoch=args.ae_epochs,
                          batch_size=args.ae_batch_size)

    # Step 2: training a diffusion autoencoder model
    rna_ae.eval()
    with torch.no_grad():
        z_x, recon_r = rna_ae(rna_train_tensor)
        z_x_t, recon_r_t = rna_ae(rna_test_tensor)
    
        

    adt_ae.eval()
    with torch.no_grad():
        z_y, recon_a = adt_ae(adt_train_tensor)
        z_y_t, recon_a_t = adt_ae(adt_test_tensor)
    
        

    labels = LabelEncoder().fit_transform(RNA_data.obs['cell_type'])

    ## create and train dae
    dae_dir = os.path.join(log_dir, 'dae')
    os.makedirs(dae_dir, exist_ok=True)
    logger.configure(dir=dae_dir)

    logger.log("training a diffusion autoencoder based on z_x and z_y...")
    dae = create_dae(args.latent_size, args.emb_size, args.hidden_dims)
    diffusion_x = create_diffusion(
        **args_to_dict(args, diffusion_defaults().keys())
    )

    dae.to(dist_util.dev())

    optimizer_dae = Adam(dae.parameters(), lr=args.dae_lr)
    schedule_sampler_x = create_named_schedule_sampler(args.schedule_sampler, diffusion_x)

    rna_tr = create_tensor(z_x, dist_util.dev())
    adt_tr = create_tensor(z_y, dist_util.dev())
    rna_te = create_tensor(z_x_t, dist_util.dev())
    adt_te = create_tensor(z_y_t, dist_util.dev())

    

    if len(args.dae_path) > 0:
        logger.log("loading dae ...")
        dae.load_state_dict(torch.load(args.dae_path))

    else:

        train_dae_all(dae,
                      diffusion_x,
                      rna_tr, adt_tr,
                      optimizer_dae,
                      dae_dir,
                      args.bl,
                      args.dae_epochs,
                      args.dae_batch_size,
                      schedule_sampler_x,
                      )

    dae.eval()
    with torch.no_grad():
        zx, zy, z_x0, z_y0, z_x_a, z_y_a, zxy = dae.encoder(rna_tr, adt_tr)
        zx_t, zy_t, z_x0_t, z_y0_t, z_x_a_t, z_y_a_t, zxy_t = dae.encoder(rna_te, adt_te)

    enc_dims = [256, 128, 64, 32]
    dec_dims = [64, 128, 256, 512]

    unt_r_dir = os.path.join(log_dir, 'unet_x')
    os.makedirs(unt_r_dir, exist_ok=True)
    logger.configure(dir=unt_r_dir)

    logger.log("create rna cross conditional unet")

    rna_cross_conditioned_unet = CrossConditionedUNet(
        source_features=args.image_size,
        target_features=args.d_size,
        cond_features=0,
        enc_dims=enc_dims,
        dec_dims=dec_dims
    )

    rna_cross_conditioned_unet.to(dist_util.dev())

    rccu, history = train_cross_conditioned_unet(
        model=rna_cross_conditioned_unet,
        source_data=z_y_a,
        target_dist=[],
        target_labels=z_x,
        model_dir=unt_r_dir,
        epochs=args.unet_epochs,
        batch_size=32,
    )

    unt_a_dir = os.path.join(log_dir, 'unet_y')
    os.makedirs(unt_a_dir, exist_ok=True)
    logger.configure(dir=unt_a_dir)

    logger.log("create adt cross conditional unet")

    adt_cross_conditioned_unet = CrossConditionedUNet(
        source_features=args.image_size,
        target_features=args.d_size,
        cond_features=0,
        enc_dims=enc_dims,
        dec_dims=dec_dims
    )

    adt_cross_conditioned_unet.to(dist_util.dev())

    accu, history = train_cross_conditioned_unet(
        model=adt_cross_conditioned_unet,
        source_data=z_x_a,
        target_dist=[],
        target_labels=z_y,
        model_dir=unt_a_dir,
        epochs=args.unet_epochs,
        batch_size=64,
    )

    device = dist_util.dev()

    rccu = rccu.to(device)
    accu = accu.to(device)

    z_x_a_t = z_x_a_t.to(device)
    z_y_a_t = z_y_a_t.to(device)

    with torch.no_grad():
        rccu.eval()
        G_RNA = rccu(z_y_a_t, None)

    with torch.no_grad():
        accu.eval()
        G_ADT = accu(z_x_a_t, None)

    print(G_RNA.shape)
    print(G_ADT.shape)

    rna_ae.eval()

    with torch.no_grad():

        G_R_D = rna_ae.decoder(G_RNA)
        G_R_D = torch.clamp(G_R_D, min=0)

    adt_ae.eval()

    with torch.no_grad():

        G_A_D = adt_ae.decoder(G_ADT)
    
    G_R_R = torch.expm1(G_R_D)
    G_R_R_cpu = G_R_R.cpu().detach().numpy()
    G_R_D_cpu = G_R_D.cpu().detach().numpy()
    G_A_D_cpu = G_A_D.cpu().detach().numpy()
    RNA_data_tensor_C = torch.clamp(rna_train_tensor, min=0)
    RNA_data_tensor_C_cpu = RNA_data_tensor_C.cpu().detach().numpy()
    ADT_data_tensor_C_cpu = adt_train_tensor.cpu().detach().numpy()


    output_path = f"{args.result_path}/pr_data/{args.d}"
    print(output_path)
    
    pd.DataFrame(G_R_R_cpu).to_csv(f'{output_path}/A2R.csv', index=False)
    pd.DataFrame(G_A_D_cpu).to_csv(f'{output_path}/R2A.csv', index=False)

   
    print("Calculate the Pearson correlation coefficient of RNA_z...")
    rp = calculate_pearson_correlation(RNA_data_tensor_C_cpu, G_R_D_cpu)
    print(f"Pearson correlation coefficient of RNA: {rp}")

    print("Calculate the Pearson correlation coefficient of ADT_z...")
    ap = calculate_pearson_correlation(ADT_data_tensor_C_cpu, G_A_D_cpu)
    print(f"Pearson correlation coefficient of ADT: {ap}")

    output_file = f"{args.result_path}/{args.d}/clustering_nc{args.d}.txt"
    picture_save_dir = f"{args.result_path}/{args.d}"
    with open(output_file, 'w') as f:

        f.write("Metrics Calculation Results\n")
        f.write("=" * 30 + "\n")

        f.write(f"Pearson correlation coefficient of RNA: {rp}\n")
        f.write(f"Pearson correlation coefficient of ADT: {ap}\n")

    print(f"Results saved to {output_file}")
    

    logger.log("complete.")


def create_argparser():
    latent_size = 256
    defaults = dict(
        # paras for ae
        rna_ae_path='',
        adt_ae_path='',
        latent_size=latent_size,
        rna_ae_lr=1e-3,
        adt_ae_lr=1e-3,
        ae_epochs=250,
        ae_batch_size=32,
        lambda1=1,
        lambda2=1,
        lambda3=1,

        # paras for dae
        dae_path='',
        input_dim=latent_size,
        emb_size=64,
        hidden_dims=[512, 512, 256, 128],
        schedule_sampler="uniform",
        dae_lr=1e-3,
        dae_epochs=120,
        dae_batch_size=32,
        unet_epochs=20,

        # common paras
        fold=1,
        bl=1.0,
        result_path='./results/re',
        rna_path='',
        adt_path='',
        log_dir='./results/CITE',
        seed=0
    )
    defaults.update(diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
