import torch
from tqdm import tqdm


@torch.no_grad()
def sampling(model, low_res_imgs, scheduler, config):
    model.eval()

    num_images = low_res_imgs.shape[0]

    # noise
    x = torch.randn((num_images, config.IMG_CHANNELS, config.IMG_SIZE,config.IMG_SIZE,),device=config.DEVICE)

    #Loop - noise to clean
    for i in tqdm(range(config.TIMESTEPS-1, -1, -1),desc='Sampling'):
        t = torch.full(size = (num_images,), fill_value = i,  device = config.DEVICE)

        pred_noise = model(x, t, low_res_imgs)

        # Denoise one step using the DDPM formula
        alpha_t_reshaped = scheduler.alphas[t][:, None, None, None]
        alpha_cumprod_t_reshaped = scheduler.alphas_cumprod[t][:, None, None, None]
        beta_t_reshaped = scheduler.betas[t][:, None, None, None]
        sigma = torch.sqrt(beta_t_reshaped)

        z = torch.randn_like(x) if i > 0 else torch.zeros_like(x)

        x = 1 / torch.sqrt(alpha_t_reshaped) * (x - ((1 - alpha_t_reshaped) / (torch.sqrt(1 - alpha_cumprod_t_reshaped))) * pred_noise) + sigma * z

    return x
