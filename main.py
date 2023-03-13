from torchsummary import summary
import pyrallis
import torch
from torchvision.utils import  make_grid, save_image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *
from config import *
from model import ViT
from torch.utils.tensorboard import SummaryWriter

"""
Train MSE config3
    * Predict sqrt_alphas_cumprod and sqrt_one_minus
"""

logger          = SummaryWriter() 
logits          = torch.cat(
    (
        torch.ones(16,1 , requires_grad = False, device = DEVICE).double(),
        torch.zeros(16,1 , requires_grad = False, device = DEVICE).double()
    ),
    dim = 1
)
# logits          = torch.zeros(16, requires_grad = False, device = DEVICE).double()       #   expected
loss_fn         = torch.nn.MSELoss(reduction = 'mean') 
invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                    std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                            transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                    std = [ 1., 1., 1. ]),
                            ])

def train_noise_level_estimator(model: ViT, dataloader, path_config: PathConfig, \
                                train_config: TrainingConfig, diffusion_config: DiffusionConfig):
    #   Training config
    epochs              = train_config.epochs
    lr                  = train_config.lr
    save_per_epochs     = train_config.save_per_epochs
    save_imgs           = train_config.save_imgs
    save_model_per_epochs   = train_config.save_model_per_epochs
    #   Path config
    save_imgs_dir       = path_config.SAVE_IMGS
    ckpt_dir            = path_config.CHECKPOINTS

    #   Setup train
    loss_train_estimator = torch.nn.MSELoss(reduction = 'mean')
    loss_reconstruction  = torch.nn.MSELoss(reduction = 'mean')
    optimizer            = torch.optim.AdamW(model.parameters(), lr = lr)
    

    betas                           = diffusion_config.betas
    alphas_cumprod                  = diffusion_config.alphas_cumprod
    one_minus_alphas_cumprod        = 1. - alphas_cumprod
    sqrt_alphas_cumprod             = diffusion_config.sqrt_alphas_cumprod
    sqrt_one_minus_alphas_cumprod   = diffusion_config.sqrt_one_minus_alphas_cumprod
    T                               = diffusion_config.T
    
    # Noise level: sqrt_alphas_cumprod and sqrt_one_minus
    noise_level = torch.cat(
        (sqrt_alphas_cumprod.view(T, 1),                #   1 - 0
        sqrt_one_minus_alphas_cumprod.view(T,1)),       #   0 - 1
        dim = 1
    )
    pbar_dataloader = tqdm(range(0, epochs))
    
    for epoch in pbar_dataloader:

        #   Get input images
        batch = next(iter(dataloader))
        images, _ = batch
        images = images.double().to(DEVICE)
        B, C, H, W = images.shape

        #   Add noise respected to noise level
        t = torch.randint(0, T, (B,), device=DEVICE).long()
        noise, z = q_sample(images, t, sqrt_alphas_cumprod, \
                    sqrt_one_minus_alphas_cumprod)
        z       = z.double()
        noise   = noise.double()

        # save_images(invTrans(z[:16]), path_config.SAVE_IMGS / "inter.png")

        output = model(z)               #   Predict 1 - alphas_cumprod  
        sqrt_alphas_cumprod_pred             = torch.reshape(output[:, 0], (B, 1, 1, 1))
        sqrt_one_minus_alphas_cumprod_pred   = torch.reshape(output[:, 1], (B, 1,1,1))

        # breakpoint()
        #   Train with loss noise level prediction to learn noise level first
        #   Learn how to reconstruct
        
        z_reconstruct                   = sqrt_alphas_cumprod_pred * images + sqrt_one_minus_alphas_cumprod_pred * noise
        
        loss                            = loss_train_estimator(output.squeeze(), noise_level[t]) 
        loss_reconstruct                = loss_reconstruction(z, z_reconstruct) 
        ensemble_loss                   = loss + loss_reconstruct 
        optimizer.zero_grad()
        ensemble_loss.backward()
        optimizer.step()        

        # for i in range(4):
        #     path = f'save_img_mse_{i+1}.png'

        #     save_images(z[i, :, :, :], path_config.SAVE_IMGS / path)
        #     print(path + f'-- Estimation = {output[i]} -- Real = {one_minus_alphas_cumprod[t[i]]} -- Time = {t[i]}')
        # exit()

        #   Optmization
        
        pbar_dataloader.set_postfix({"Loss": loss.item()})
        pbar_dataloader.set_postfix({'Reconstruction loss': loss_reconstruct.item()})
        logger.add_scalar('train/mse/config3/noise_level', loss.item(), epoch)
        logger.add_scalar('train/mse/config3/reconstruct', loss_reconstruct.item(), epoch)
        logger.add_scalar('train/mse/config3/ensemble', ensemble_loss.item(), epoch)

    #   Validate to save model
        if epoch % save_model_per_epochs == 0:
            save_model(model, ckpt_dir / "mse" / "config3" /"checkpoints_epochs_{}".format(epoch))
 
        if epoch % save_per_epochs == 0:
            freeze_model(model)
            model.eval()
            z = sampling(model, epoch, path_config, diffusion_config)
            unfreeze_model(model)
            model.train()

        logger.flush()

def sampling(estimator: ViT, epoch: int, path_config: PathConfig, diffusion_config: DiffusionConfig): 
    
    steps           = diffusion_config.sampling_steps         #   Sampling steps
    B               = diffusion_config.sampling_batch_size
    
    C, H, W         = 3, 64, 64
    z               = torch.randn(B, C, H, W, device = DEVICE).double()
    z               = torch.nn.parameter.Parameter(z, requires_grad = True)

    lr = 1e-3

    sampling_optimizer = torch.optim.AdamW([z], lr = lr)
    
                            #   loss function
    #   Optimize to generate
    for step in range(steps): 
        
        output = estimator(z).squeeze()                                             #   noise level prediction
        loss = loss_fn(output, logits)       
        sampling_optimizer.zero_grad()                       
        loss.backward()
        sampling_optimizer.step()
        logger.add_scalar('sampling/mse/config3/loss', loss.item(), step)
    
    
    gen = invTrans(z)
    grid = make_grid(gen[:16], nrow = 4)
    logger.add_image(f'sample_at_epoch{epoch}', grid, epoch)
    return gen

def freeze_model(model):

    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

def main(param: ViTConfig, path: PathConfig, train_param: TrainingConfig, diffusion_config: DiffusionConfig):

    print("Prepare model and dataloader")
    dataset, dataloader = get_celeba(train_param.batch_size, path.CELEBA_DIR, train_param.num_worker)
    model = ViT(
        image_size          = param.image_size,
        patch_size          = param.patch_size,
        num_classes         = param.num_classes,
        dim                 = param.dim,
        depth               = param.depth,
        heads               = param.heads,
        mlp_dim             = param.mlp_dim,
        pool                = param.pool,
        channels            = param.channels,
        dim_head            = param.dim_heads,
        dropout             = param.dropout,
        emb_dropout         = param.emb_dropout
    ).double().to(DEVICE)
    print("Loaded model and dataloader!")

    #   Train noise level estimator
    train_noise_level_estimator(
        model               = model,
        dataloader          = dataloader,
        path_config         = PathConfig(),
        train_config        = TrainingConfig(),
        diffusion_config    = DiffusionConfig()
    )


def save_model(model, path):
    
    print("Saving...")
    torch.save(model.state_dict(), path)
    print("Save!")

def save_loss(loss: list, path: str):

    plt.plot(loss)
    plt.savefig(path)    
    return loss

def save_images(images, path, row = 4):
    """
        images: 4D (B, C, H, W)
    """
    print("Save images...")
    grid = make_grid(images, nrow =  row)
    save_image(grid, path)
    print("Save!")


if __name__ == '__main__':

    # torch.set_default_tensor_type(torch.cuda.FloatTensor)

    print("MSE Loss without softplus config3")
    print("Load configurations...")
    param = pyrallis.parse(config_class = ViTConfig)
    path =  pyrallis.parse(config_class = PathConfig)
    train_param =  pyrallis.parse(config_class = TrainingConfig)
    diffusion_config = pyrallis.parse(config_class = DiffusionConfig)
    print("Configurations are set")

    main(param, path, train_param, diffusion_config)

    
