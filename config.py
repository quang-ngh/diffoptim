from pathlib import Path
from dataclasses import dataclass
from utils import *
import torch 

DEVICE = torch.device('cuda:2') if torch.cuda.is_available() else 'cpu'

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

    image_size  = 64        #   batch_size
    patch_size  = 8         #   patch size
    num_classes = 1         #   output size at the last mlp layer
    dim         = 768      #   embedding dim
    depth       = 6        #   number of encoders
    heads       = 5         #   number of heads
    mlp_dim     = 768       #   last mlp layer dim
    pool        = 'cls'     #   
    channels    = 3
    dim_heads   = 64
    dropout     = 0.0
    emb_dropout = 0.0

@dataclass
class TrainingConfig:

    num_worker = 0
    batch_size = 128
    lr = 1e-3
    epochs = 1000
    save_per_epochs = 1
    loss_per_iter   = 20
    save_imgs = True
    
@dataclass
class DiffusionConfig:

    #   Sampling
    sampling_steps = 500
    sampling_batch_size = 16

    T                               = 500
    betas                           = cosine_beta_schedule(T).to(DEVICE).double()
    alphas                          = 1. - betas.to(DEVICE).double()
    alphas_cumprod                  = torch.cumprod(alphas, dim=0).to(DEVICE).double()
    alphas_cumprod_prev             = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value = 1.).to(DEVICE).double()

    sqrt_alphas_cumprod             = torch.sqrt(alphas_cumprod).to(DEVICE).double()
    sqrt_one_minus_alphas_cumprod   = torch.sqrt(1. - alphas_cumprod).to(DEVICE).double()
    log_one_minus_alphas_cumprod    = torch.log(1. - alphas_cumprod).to(DEVICE).double()
    sqrt_recip_alphas_cumprod       = torch.sqrt(1. / alphas_cumprod).to(DEVICE).double()
    sqrt_recipm1_alphas_cumprod     = torch.sqrt(1. / alphas_cumprod - 1).to(DEVICE).double()