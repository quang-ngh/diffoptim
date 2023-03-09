from torchsummary import summary
import pyrallis
import torch
from torchvision.utils import  make_grid, save_image
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *
from config import *
from model import ViT

def train_noise_level_estimator(model: ViT, dataloader, epochs: int, \
                                lr: float, save_per_epochs: int, save_imgs : bool, ckpt_path, \
                                diffusion_config):

    losses      = []
    optimizer   = torch.optim.AdamW(model.parameters(), lr = lr)
    betas = diffusion_config.betas

    for epoch in range(epochs):
        total_loss = 0
        count = 0
        for idx, batch in enumerate(tqdm(dataloader)):
            #   Get input images
            images, _ = batch
            images = images.double().to(DEVICE)    
            B, C, H, W = images.shape

            # breakpoint()
            
            #   Add noise respected to noise level
            t = torch.randint(0, diffusion_config.T, (B,), device=DEVICE).long()
            z = q_sample(images, t, diffusion_config.sqrt_alphas_cumprod, \
                        diffusion_config.sqrt_one_minus_alphas_cumprod).double()

            # output = model(z) 
            output = model(z)
            output = output.squeeze() 
            #   Optmization
            # breakpoint()
            loss = torch.nn.BCELoss(reduction = "mean")(output, betas[t]) 
            
            loss.backward()
            
            optimizer.step()
            total_loss += loss.item()
            count += 1
        print("Epoch: {} -- Loss = {}".format(epoch+1, loss.item()))

    #   Validate to save model
        curr_loss = total_loss / count
        if len(losses) == 0:
            save_model(model, ckpt_path / "checkpoint_epoch1") 
        else:
            if curr_loss < min(losses):             # Minimum loss
                print("Current loss = {} < {}".format(curr_loss, min(losses)))
                save_model(model, ckpt_path / "checkpoints_epochs_{}".format(epoch+1))

        losses.append(curr_loss)
    save_loss(losses, SAVE_LOSS / "loss.png")


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
        model = model,
        dataloader = dataloader,
        epochs = train_param.epochs,
        lr = train_param.lr, 
        save_per_epochs = train_param.save_per_epochs,
        save_imgs = train_param.save_imgs,
        ckpt_path = path.CHECKPOINTS,
        diffusion_config = diffusion_config

    )
    if DEBUG_PRINT:
        print(images.shape)
        print(output.shape)

def sampling(estimator: ViT): 
    pass

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

    