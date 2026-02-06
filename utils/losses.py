import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_selection import mutual_info_regression

def ce_loss(masks_prob, true_masks, eps=sys.float_info.epsilon):
    loss = true_masks * torch.log(masks_prob + eps) + (1 - true_masks) * torch.log(1- masks_prob + eps)
    loss = -torch.mean(loss)
    return loss

# joint probability
def compute_joint(x, y):
    """Compute the joint probability matrix P"""

    bn, k = x.size()
    assert (y.size(0) == bn and y.size(1) == k)

    p_i_j = x.unsqueeze(2) * y.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def mi_loss(x, y, eps=sys.float_info.epsilon):
    """mutual information"""
    bn, k = x.size()
    p_i_j = compute_joint(x, y)

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < eps, torch.tensor([eps], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < eps, torch.tensor([eps], device=p_j.device), p_j)
    p_i = torch.where(p_i < eps, torch.tensor([eps], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_j) + torch.log(p_i) - torch.log(p_i_j))

    loss = loss.sum()

    return loss


def cosine_similarity_loss(x, y):
    cos = nn.CosineSimilarity(dim=1)
    outputs = cos(x, y) + 1
    loss = outputs.sum()
    return loss

