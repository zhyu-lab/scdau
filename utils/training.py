import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
sys.path.append("..")
from models.dae import DAE
from diffusion import dist_util, logger
from diffusion.resample import LossAwareSampler, UniformSampler

from .losses import mi_loss



def train_rna_ae_all(ae,
                 train_data,
                 optimizer,
                 model_dir,
                 train_epoch=100,
                 batch_size=32,
                 r_loss_fn=nn.MSELoss()):

    train_losses = []

    for epoch in range(train_epoch):

        ae.train()

        train_loss = 0

        dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

        for batch in dataloader:

            optimizer.zero_grad()

            z, y = ae(batch)

            loss = r_loss_fn(y, batch)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()
        avg_train_loss = train_loss / (train_data.size(0) // batch_size)
        train_losses.append(avg_train_loss)

        print(f'Epoch [{epoch + 1}/{train_epoch}], Train Loss: {train_losses[-1]:.4f}')

    final_model_path = os.path.join(model_dir, 'final_rna_ae.pt')
    torch.save(ae.state_dict(), final_model_path)
    print(f'Final model saved at {final_model_path}')

    return train_losses

def train_adt_ae_all(ae,
                 train_data,
                 optimizer,
                 model_dir,
                 train_epoch=100,
                 batch_size=32,
                 r_loss_fn=nn.MSELoss()):

    train_losses = []

    for epoch in range(train_epoch):

        ae.train()

        train_loss = 0

        dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

        for batch in dataloader:

            optimizer.zero_grad()

            z, y = ae(batch)

            loss = r_loss_fn(y, batch)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()
        avg_train_loss = train_loss / (train_data.size(0) // batch_size)
        train_losses.append(avg_train_loss)

        print(f'Epoch [{epoch + 1}/{train_epoch}], Train Loss: {train_losses[-1]:.4f}')

    final_model_path = os.path.join(model_dir, 'final_rna_ae.pt')
    torch.save(ae.state_dict(), final_model_path)
    print(f'Final model saved at {final_model_path}')

    return train_losses

def train_atac_ae_all(ae,
                  train_data,
                  optimizer,
                  model_dir,
                  train_epoch=100,
                  batch_size=32,
                  a_loss_fn=nn.BCELoss()):



    train_losses = []


    for epoch in range(train_epoch):

        ae.train()

        train_loss = 0

        dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)


        for batch in dataloader:

            optimizer.zero_grad()
            z, y = ae(batch)

            loss = a_loss_fn(y, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()


        avg_train_loss = train_loss / (train_data.size(0) // batch_size)
        train_losses.append(avg_train_loss)


        print(f'Epoch [{epoch + 1}/{train_epoch}], Train Loss: {train_losses[-1]:.4f}')




    final_model_path = os.path.join(model_dir, 'final_atac_ae.pt')
    torch.save(ae.state_dict(), final_model_path)
    print(f'Final model saved at {final_model_path}')

    return train_losses


def train_dae_all(dae,
              diffusion_x,
              rna_tensor_train, atac_tensor_train,
              optimizer,
              model_dir,
              bl=1,
              train_epoch=100,
              batch_size=32,
              schedule_sampler_x=None):

    rna_train_losses = []
    atac_train_losses = []
    zero_train_losses = []
    train_losses = []


    train_data = TensorDataset(rna_tensor_train, atac_tensor_train)

    zero_loss_fn = nn.L1Loss()

    schedule_sampler_x = schedule_sampler_x or UniformSampler(diffusion_x)

    for epoch in range(train_epoch):
        dae.train()

        rna_train_loss = 0
        atac_train_loss = 0
        zero_train_loss = 0
        train_loss = 0

        dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

        for rna_batch, atac_batch in dataloader:
            optimizer.zero_grad()

            z_x0, z_y0, cond_x, cond_y = dae(rna_batch, atac_batch)

            t_x, weights_x = schedule_sampler_x.sample(rna_batch.shape[0], dist_util.dev())
            model_kwargs = {'y': cond_x}
            loss_diffx = diffusion_x.training_losses(dae.unet_x, rna_batch, t_x, model_kwargs)
            loss_r = (loss_diffx["loss"] * weights_x).mean()

            t_y, weights_y = schedule_sampler_x.sample(atac_batch.shape[0], dist_util.dev())
            model_kwargs = {'y': cond_y}
            loss_diffx = diffusion_x.training_losses(dae.unet_x, atac_batch, t_y, model_kwargs)
            loss_a = (loss_diffx["loss"] * weights_y).mean()

            loss_zero = zero_loss_fn(z_x0, torch.zeros_like(z_x0)) + zero_loss_fn(z_y0, torch.zeros_like(z_y0))
            loss = bl*(loss_r + loss_a) + loss_zero

            loss.backward()
            optimizer.step()

            rna_train_loss += loss_r.item()
            atac_train_loss += loss_a.item()
            zero_train_loss += loss_zero.item()
            train_loss += loss.item()


        avg_train_loss = train_loss / (rna_tensor_train.size(0) // batch_size)
        train_losses.append(avg_train_loss)
        rna_train_losses.append(rna_train_loss / (rna_tensor_train.size(0) // batch_size))
        atac_train_losses.append(atac_train_loss / (rna_tensor_train.size(0) // batch_size))
        zero_train_losses.append(zero_train_loss / (rna_tensor_train.size(0) // batch_size))

        print(f'train step [{epoch + 1}/{train_epoch}]:')
        print(
            f'train loss: {train_losses[-1]:.4f}, '
            f'RNA loss: {rna_train_losses[-1]:.4f}, '
            f'ATAC loss: {atac_train_losses[-1]:.4f}, '
            f'Zero loss: {zero_train_losses[-1]:.4f}')

    torch.save(dae.state_dict(), model_dir + '/final_dae.pt')
    dae.load_state_dict(torch.load(model_dir + '/final_dae.pt'))
    print(f'training complete.')

    return train_losses, rna_train_losses, atac_train_losses, zero_train_losses
