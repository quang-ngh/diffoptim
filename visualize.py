from config import *
from model import *
import torch
import matplotlib.pyplot as plt 
from config import DEVICE, get_celeba
from main import save_images

path_config = PathConfig()
diffusion_config = DiffusionConfig()
vit_config = ViTConfig()
betas = diffusion_config.betas
alphas_cumprod = diffusion_config.alphas_cumprod
one_minus_alpha_cumprod = 1. - alphas_cumprod

def convert_plot(x):
    return x.detach().cpu().numpy()

def plot_noise_config():

    plt.subplot(1,3,1)
    plt.plot(convert_plot(betas))

    plt.subplot(1,3,2)
    plt.plot(convert_plot(alphas_cumprod))

    plt.subplot(1,3,3)
    plt.plot(convert_plot(one_minus_alpha_cumprod))
    plt.savefig(path_config.FIGURES / 'linear.png')

def sample(batch, dict_path):

    #   Load model 
    model = ViT(
        image_size          = vit_config.image_size,
        patch_size          = vit_config.patch_size,
        num_classes         = vit_config.num_classes,
        dim                 = vit_config.dim,
        depth               = vit_config.depth,
        heads               = vit_config.heads,
        mlp_dim             = vit_config.mlp_dim,
        pool                = vit_config.pool,
        channels            = vit_config.channels,
        dim_head            = vit_config.dim_heads,
        dropout             = vit_config.dropout,
        emb_dropout         = vit_config.emb_dropout
    ).double().to(DEVICE)
    model.load_state_dict(torch.load(dict_path))
    model.eval()

    lr = 2e-3
    B = batch
    C,H,W = 3, 64, 64
    z = torch.randn(B, C, H, W, device = DEVICE).double()
    z = torch.nn.parameter.Parameter(z, requires_grad = True)
    logits = torch.zeros(B, device=DEVICE, requires_grad = False).double()
    optimizer = torch.optim.AdamW([z], lr = lr)

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction = 'mean')

    sampling_steps = 10000
    for i in range(sampling_steps):
        out = model(z)
        out = torch.nn.Sigmoid()(out).squeeze()

        loss = loss_fn(out, logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Loss = {loss.item()} -- Noise level = {out[0]}')

    save_images(z, path_config.SAVE_IMGS / 'sample.png')

def explicit_sample(batch, dict_path): 
    pass


def test_model(dict_path, dataloader):
    
    model = ViT(
        image_size          = vit_config.image_size,
        patch_size          = vit_config.patch_size,
        num_classes         = vit_config.num_classes,
        dim                 = vit_config.dim,
        depth               = vit_config.depth,
        heads               = vit_config.heads,
        mlp_dim             = vit_config.mlp_dim,
        pool                = vit_config.pool,
        channels            = vit_config.channels,
        dim_head            = vit_config.dim_heads,
        dropout             = vit_config.dropout,
        emb_dropout         = vit_config.emb_dropout
    ).double().to(DEVICE)
    model.load_state_dict(torch.load(dict_path))
    model.eval()

    sampling_steps = 1000
    lr             = 2e-3

    batch = next(iter(dataloader))
    images, _ = batch
    images = images[:4].double().to(DEVICE)
    B, C, H, W = images.shape

    betas                               = diffusion_config.betas
    alphas_cumprod                      = diffusion_config.alphas_cumprod
    one_minus_alpha_cumprod             = 1. - diffusion_config.alphas_cumprod
    sqrt_alphas_cumprod                 = diffusion_config.sqrt_alphas_cumprod
    sqrt_one_minus_alphas_cumprod       = diffusion_config.sqrt_one_minus_alphas_cumprod
    
    #   Add noise respected to noise level
    t = torch.randint(0, diffusion_config.T, (B,), device=DEVICE).long()
    z = q_sample(images, t, sqrt_alphas_cumprod, \
                sqrt_one_minus_alphas_cumprod).double()
    output = model(z).squeeze()
    for i in range(4):
        path = f'test_mse_{i+1}.png'

        save_images(z[i, :, :, :], path_config.SAVE_IMGS / path)
        print(path + f'-- Estimation = {output[i]} -- Real = {one_minus_alpha_cumprod[t[i]]} -- Time = {t[i]}')
    exit()

    save_images(z, path_config.SAVE_IMGS / "noisy_images.png") 

    z = torch.nn.parameter.Parameter(z, requires_grad = True)
    sampling_optimizer = torch.optim.AdamW([z], lr = lr)
    loss_fn             = torch.nn.MSELoss(reduction = 'mean')
    logits = torch.zeros(B, device = DEVICE, requires_grad = False).double()

    for step in range(sampling_steps):

        output = model(z)
        output = output.squeeze() 
        loss = loss_fn(output, logits)
        sampling_optimizer.zero_grad()
        loss.backward()
        sampling_optimizer.step() 
        print(f'Loss = {loss.item()}')


    save_images(z, path_config.SAVE_IMGS / 'denoise_image.png')
    # for i in range(t.shape[0]):
    #     print(f'Time step = {t[i]} -- Real noise level = {one_minus_alpha_cumprod[t[i]]} -- Predict = {output[i]}')
if __name__ == '__main__':
    dataset, dataloader = get_celeba(16, path_config.CELEBA_DIR, 0)
    ckpt_path = path_config.CHECKPOINTS / 'mse/checkpoints_epochs_15000'
    test_model(ckpt_path, dataloader)

    # dataset, dataloader = get_celeba(5, path_config.CELEBA_DIR, 0)
    # test_model(path_config.CHECKPOINTS / "cosine/config1/checkpoints_epochs_40001", dataloader)