import torch
import torch.nn.functional as F

class Scheduler:
    def get_linear_beta_scheduler(self, timesteps, device, config):
        betas = torch.linspace(config.BETA_START,config.BETA_STOP,timesteps, device = device, dtype = torch.float32)
        return betas

    def __init__(self, timesteps, device):
        self.betas = self.get_linear_beta_scheduler(timesteps, device)
        self.alphas = 1 - self.betas

        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1-self.alphas_cumprod)

    def add_noise(self, x_start, t, noise):
        # x_start = (B, C, H, W) - batch is the same at time as each image is a different time?

        x_t = self.sqrt_alphas_cumprod[t].view(-1,1,1,1) * x_start + self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1) * noise
        return x_t

