from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch
import math

def get_celeba(batch_size, dataset_directory, dataloader_workers):
    # 1. Download this file into dataset_directory and unzip it:
    #  https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
    # 2. Put the `img_align_celeba` directory into the `celeba` directory!
    # 3. Dataset directory structure should look like this (required by ImageFolder from torchvision):
    #  +- `dataset_directory`
    #     +- celeba
    #        +- img_align_celeba
    #           +- 000001.jpg
    #           +- 000002.jpg
    #           +- 000003.jpg
    #           +- ...
    train_transformation = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = torchvision.datasets.ImageFolder(dataset_directory, train_transformation)
    train_dataset = torch.utils.data.Subset(train_dataset, range(0, 400, 1)) 

    # Use sampler for randomization
    training_sampler = torch.utils.data.SubsetRandomSampler(range(len(train_dataset)))

    # Prepare Data Loaders for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=training_sampler,
                                               pin_memory=False, num_workers=dataloader_workers)

    return train_dataset, train_loader
def exists(x):
    return x is not None

def extract(a, t, x_shape):
    # breakpoint()     
    b, *_ = t.shape
    out = a.gather(-1, t)
    
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = 1e-4
    beta_end = 1e-2
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def q_sample(x_start, t, sqrt_alphas_cumprod, \
            sqrt_one_minus_alphas_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start, device = 'cuda:0'))

        return (
            extract(sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

