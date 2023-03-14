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
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    sampling_steps = 1000
    lr             = 1e-2

    batch = next(iter(dataloader))
    images, _ = batch
    images = images[:16].double().to(DEVICE)
    B, C, H, W = images.shape

    betas                               = diffusion_config.betas
    betas.requires_grad = False
    alphas_cumprod                      = diffusion_config.alphas_cumprod
    one_minus_alpha_cumprod             = 1. - diffusion_config.alphas_cumprod
    sqrt_alphas_cumprod                 = diffusion_config.sqrt_alphas_cumprod
    sqrt_one_minus_alphas_cumprod       = diffusion_config.sqrt_one_minus_alphas_cumprod
    T                                   = diffusion_config.T 
    noise_level = torch.cat(
        (sqrt_alphas_cumprod.view(T, 1),                #   1 - 0
        sqrt_one_minus_alphas_cumprod.view(T,1)),       #   0 - 1
        dim = 1
    )
    
    #   Add noise respected to noise level
    # t                   = torch.randint(0, diffusion_config.T, (B,), device=DEVICE).long()
    # noise, z                   = q_sample(images, t, sqrt_alphas_cumprod, \
    #                         sqrt_one_minus_alphas_cumprod)
    # z                   = z.double()

    z                   =   torch.randn_like(images, device = DEVICE, requires_grad = True).double()
    save_images(z, path_config.FIGURES / 'init_noise.png')

    z                   = torch.nn.parameter.Parameter(z, requires_grad = True)
    sampling_optimizer  = torch.optim.AdamW([z], lr = lr)
    loss_fn             = torch.nn.MSELoss(reduction = 'mean')

    # logits              = torch.cat(
    #     (
    #         torch.ones(16,1 , requires_grad = False, device = DEVICE).double(),
    #         torch.zeros(16,1 , requires_grad = False, device = DEVICE).double()
    #     ),
    #     dim = 1
    # )

    alpha_sample = (1. - betas).view(T, 1, 1, 1)

    print(f'Sampling in {sampling_steps}')
    for step in range(sampling_steps):

        output = model(z)

        B, D    = output.shape      #   batch x 2

        #   Diff between sqrt_one_minus predicted with sqrt_one_minus predefined

        logits  = torch.ones(B, 1, device = DEVICE, requires_grad = False) * sqrt_one_minus_alphas_cumprod[sampling_steps - step - 1]
        logits  = logits.double()
        loss    = loss_fn(output[:, 1], logits)
        # breakpoint()

        #   Implicit sampling
        sampling_optimizer.zero_grad()
        loss.backward()
        sampling_optimizer.step() 

        z.requires_grad = False
        z += betas[sampling_steps - step - 1] * torch.randn_like(z, requires_grad = False, device = DEVICE).double()
        z.requires_grad = True
        print(f'Loss = {loss.item()}')
        print(f'Sqrt one minus predicted at {sampling_steps - step} = {output[:, 1]}')

        # if step == sampling_steps - 1:
        #     z_T = output[:, 0].view(16,1,1,1) * images + output[:, 1].view(16,1,1,1) * noise
        #     save_images(z_T, path_config.FIGURES / 'real_level_noise.png')

    save_images(z, path_config.FIGURES / "sample_implicit_46500.png")

if __name__ == '__main__':
    dataset, dataloader = get_celeba(16, path_config.CELEBA_DIR, 0)
    ckpt_path = path_config.CHECKPOINTS / 'mse/config3/checkpoints_epochs_46500'
    test_model(ckpt_path, dataloader)

    # dataset, dataloader = get_celeba(5, path_config.CELEBA_DIR, 0)
    # test_model(path_config.CHECKPOINTS / "cosine/config1/checkpoints_epochs_40001", dataloader)