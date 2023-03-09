from pathlib import Path
from dataclasses import dataclass
from utils import *
import torch 

DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else 'cpu'
DEBUG_PRINT = True

@dataclass
class PathConfig:
    DATA_DIR = Path('/home/ubuntu/newSDE/dataset')
    CELEBA_DIR = DATA_DIR / "celeba/"
    FIGURES = Path('/home/ubuntu/newSDE/figures')
    SAVE_LOSS = FIGURES / "loss"
    SAVE_IMGS = FIGURES / "images_test"

    #   Checkpoints
    CHECKPOINTS = Path('/home/ubuntu/newSDE/ckpts')

@dataclass
class ViTConfig:

    image_size  = 64
    patch_size  = 8
    num_classes = 1
    dim         = 1024
    depth       = 12
    heads       = 5
    mlp_dim     = 768
    pool        = 'cls'
    channels    = 3
    dim_heads   = 64
    dropout     = 0.0
    emb_dropout = 0.0

@dataclass
class TrainingConfig:

    num_worker = 0
    batch_size = 512
    lr = 1e-4
    epochs = 1
    save_per_epochs = 10
    save_imgs = True
    
@dataclass
class DiffusionConfig:
    T                               = 500
    betas                           = cosine_beta_schedule(T).to(DEVICE)
    alphas                          = 1. - betas.to(DEVICE)
    alphas_cumprod                  = torch.cumprod(alphas, dim=0).to(DEVICE)
    alphas_cumprod_prev             = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value = 1.).to(DEVICE)

    sqrt_alphas_cumprod             = torch.sqrt(alphas_cumprod).to(DEVICE)
    sqrt_one_minus_alphas_cumprod   = torch.sqrt(1. - alphas_cumprod).to(DEVICE)
    log_one_minus_alphas_cumprod    = torch.log(1. - alphas_cumprod).to(DEVICE)
    sqrt_recip_alphas_cumprod       = torch.sqrt(1. / alphas_cumprod).to(DEVICE)
    sqrt_recipm1_alphas_cumprod     = torch.sqrt(1. / alphas_cumprod - 1).to(DEVICE)
