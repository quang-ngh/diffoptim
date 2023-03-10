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

logger = SummaryWriter() 
logits          = torch.zeros(16, requires_grad = False, device = DEVICE).double()       #   expected
loss_fn         = torch.nn.BCELoss(reduction = 'mean') 
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

    #   Path config
    save_imgs_dir       = path_config.SAVE_IMGS
    ckpt_dir            = path_config.CHECKPOINTS

    #   Setup train
    losses      = []
    optimizer   = torch.optim.AdamW(model.parameters(), lr = lr)
    betas = diffusion_config.betas

    pbar_dataloader = tqdm(dataloader)
    count = 0
    for epoch in range(epochs):

        unfreeze_model(model)
        model.train()
        total_loss = 0
        
        for idx, batch in enumerate(pbar_dataloader):
            #   Get input images
            images, _ = batch
            images = images.double().to(DEVICE)
            B, C, H, W = images.shape

            
            #   Add noise respected to noise level
            t = torch.randint(0, diffusion_config.T, (B,), device=DEVICE).long()
            z = q_sample(images, t, diffusion_config.sqrt_alphas_cumprod, \
                        diffusion_config.sqrt_one_minus_alphas_cumprod).double()
            

            output = model(z)
            output = output.squeeze() 
            #   Optmization
            loss = torch.nn.BCELoss(reduction = "mean")(output, betas[t]) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1

            pbar_dataloader.set_postfix({"Loss": loss.item()})
            if idx % train_config.loss_per_iter == 0:
                logger.add_scalar('train/loss', loss.item(), count)

    #   Validate to save model
        curr_loss = total_loss / count

        if len(losses) == 0:
            save_model(model, ckpt_dir / "checkpoint_epoch1") 
        else:
            if curr_loss < min(losses):             # Minimum loss
                print("Current loss = {} < {}".format(curr_loss, min(losses)))
                save_model(model, ckpt_dir / "checkpoints_epochs_{}".format(epoch+1))

        losses.append(curr_loss)
        
        if epoch % save_per_epochs == 0:
            freeze_model(model)
            model.eval()
            z = sampling(model, epoch, path_config, diffusion_config)

        logger.flush()
    save_loss(losses, save_imgs_dir / "loss.png")

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
        logger.add_scalar('sampling/loss', loss.item(), step)
    
    
    gen = invTrans(z)
    path = "sample_epochs_"+str(epoch) + ".png"
    save_images(gen, path_config.SAVE_IMGS / path) 
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

    print("Load configurations...")
    param = pyrallis.parse(config_class = ViTConfig)
    path =  pyrallis.parse(config_class = PathConfig)
    train_param =  pyrallis.parse(config_class = TrainingConfig)
    diffusion_config = pyrallis.parse(config_class = DiffusionConfig)
    print("Configurations are set")

    main(param, path, train_param, diffusion_config)

    