import sys
import os
import functools
import torch
import numpy as np
from ipdb import set_trace
from scipy import integrate
from utils.genpose_utils import get_pose_dim

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


#----- VE SDE -----
#------------------
def ve_marginal_prob(x, t, sigma_min=0.01, sigma_max=90):
    std = sigma_min * (sigma_max / sigma_min) ** t
    mean = x
    return mean, std

def ve_sde(t, sigma_min=0.01, sigma_max=90):
    sigma = sigma_min * (sigma_max / sigma_min) ** t
    drift_coeff = torch.tensor(0)
    diffusion_coeff = sigma * torch.sqrt(torch.tensor(2 * (np.log(sigma_max) - np.log(sigma_min)), device=t.device))
    return drift_coeff, diffusion_coeff

def ve_prior(shape, sigma_min=0.01, sigma_max=90, T=1.0):
    _, sigma_max_prior = ve_marginal_prob(None, T, sigma_min=sigma_min, sigma_max=sigma_max)
    return torch.randn(*shape) * sigma_max_prior

#----- VP SDE -----
#------------------
def vp_marginal_prob(x, t, beta_0=0.1, beta_1=20):
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    mean = torch.exp(log_mean_coeff) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

def vp_sde(t, beta_0=0.1, beta_1=20):
    beta_t = beta_0 + t * (beta_1 - beta_0)
    drift_coeff = -0.5 * beta_t
    diffusion_coeff = torch.sqrt(beta_t)
    return drift_coeff, diffusion_coeff

def vp_prior(shape, beta_0=0.1, beta_1=20):
    return torch.randn(*shape)

#----- sub-VP SDE -----
#----------------------
def subvp_marginal_prob(x, t, beta_0, beta_1):
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    mean = torch.exp(log_mean_coeff) * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

def subvp_sde(t, beta_0, beta_1):
    beta_t = beta_0 + t * (beta_1 - beta_0)
    drift_coeff = -0.5 * beta_t
    discount = 1. - torch.exp(-2 * beta_0 * t - (beta_1 - beta_0) * t ** 2)
    diffusion_coeff = torch.sqrt(beta_t * discount)
    return drift_coeff, diffusion_coeff

def subvp_prior(shape, beta_0=0.1, beta_1=20):
    return torch.randn(*shape)

#----- EDM SDE -----
#------------------
def edm_marginal_prob(x, t, sigma_min=0.002, sigma_max=80):
    std = t
    mean = x
    return mean, std

def edm_sde(t, sigma_min=0.002, sigma_max=80):
    drift_coeff = torch.tensor(0)
    diffusion_coeff = torch.sqrt(2 * t)
    return drift_coeff, diffusion_coeff

def edm_prior(shape, sigma_min=0.002, sigma_max=80):
    return torch.randn(*shape) * sigma_max

def init_sde(sde_mode):
    # the SDE-related hyperparameters are copied from https://github.com/yang-song/score_sde_pytorch
    if sde_mode == 'edm':
        sigma_min = 0.002
        sigma_max = 80
        eps = 0.002
        prior_fn = functools.partial(edm_prior, sigma_min=sigma_min, sigma_max=sigma_max)
        marginal_prob_fn = functools.partial(edm_marginal_prob, sigma_min=sigma_min, sigma_max=sigma_max)
        sde_fn = functools.partial(edm_sde, sigma_min=sigma_min, sigma_max=sigma_max)
        T = sigma_max
    elif sde_mode == 've':
        sigma_min = 0.01
        sigma_max = 50
        eps = 1e-5
        marginal_prob_fn = functools.partial(ve_marginal_prob, sigma_min=sigma_min, sigma_max=sigma_max)
        sde_fn = functools.partial(ve_sde, sigma_min=sigma_min, sigma_max=sigma_max)
        T = 1.0
        prior_fn = functools.partial(ve_prior, sigma_min=sigma_min, sigma_max=sigma_max)
    elif sde_mode == 'vp':
        beta_0 = 0.1
        beta_1 = 20
        eps = 1e-3
        prior_fn = functools.partial(vp_prior, beta_0=beta_0, beta_1=beta_1)
        marginal_prob_fn = functools.partial(vp_marginal_prob, beta_0=beta_0, beta_1=beta_1)
        sde_fn = functools.partial(vp_sde, beta_0=beta_0, beta_1=beta_1)
        T = 1.0
    elif sde_mode == 'subvp':
        beta_0 = 0.1
        beta_1 = 20
        eps = 1e-3
        prior_fn = functools.partial(subvp_prior, beta_0=beta_0, beta_1=beta_1)
        marginal_prob_fn = functools.partial(subvp_marginal_prob, beta_0=beta_0, beta_1=beta_1)
        sde_fn = functools.partial(subvp_sde, beta_0=beta_0, beta_1=beta_1)
        T = 1.0
    else:
        raise NotImplementedError
    return prior_fn, marginal_prob_fn, sde_fn, eps, T

