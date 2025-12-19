import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F_transforms

# Sinusoidal position embeddings

class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim): #dim is the dimension of the positional embeddings
        super().__init__()
        self.dim = dim
    def forward(self, t):

        ind = torch.arange(start=0, end=self.dim // 2, device=t.device)
        denom = torch.pow(10000, 2 * ind / self.dim)

        argument = t.float()[:, None] / denom[None, :]

        embeddings = torch.cat((argument.sin(), argument.cos()), dim = -1)

        return embeddings


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = in_channels,
                               out_channels = out_channels,
                               kernel_size = 3,
                               padding = 1,
                               padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels = out_channels,
                               out_channels = out_channels,
                               kernel_size = 3,
                               padding = 1,
                               padding_mode='reflect')

        self.bnorm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.bnorm2= nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.time_mlp = nn.Linear(in_features=time_embed_dim, out_features = out_channels)

        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, t):

        x_residual = self.shortcut(x)
        t_embed = F.relu(self.time_mlp(t))

        x = self.conv1(x)
        x = self.bnorm1(x)
        x = x + t_embed.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bnorm2(x)
        x = x + x_residual

        x = F.relu(x)

        return x

class ConditionalUNet(nn.Module):
    def __init__(self, img_channels, time_embed_dim, config):
        super().__init__()

        down_channels = config.DOWN_CHANNELS
        up_channels = config.UP_CHANNELS

        # Initial Projection
        # Concat noisy image + low res condition therefore img_channels * 2
        self.conv0 = nn.Conv2d(in_channels=img_channels * 2,
                               out_channels=down_channels[0],
                               kernel_size=3, padding=1,
                               padding_mode='reflect')

        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, 4 * time_embed_dim),
            nn.GELU(),
            nn.Linear(4 * time_embed_dim, time_embed_dim)
        )

        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(len(down_channels) - 1):
            self.downs.append(Block(in_channels=down_channels[i],
                                    out_channels=down_channels[i],
                                    time_embed_dim=time_embed_dim))

            # downsamples done seperately
            self.downsamples.append(nn.Conv2d(in_channels=down_channels[i],
                                              out_channels=down_channels[i+1],
                                              kernel_size=4, stride=2, padding=1,padding_mode='reflect'))

        self.mid_block = Block(in_channels=down_channels[-1],
                               out_channels=down_channels[-1],
                               time_embed_dim=time_embed_dim)

        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in range(len(up_channels) - 1):
            self.upsamples.append(nn.ConvTranspose2d(in_channels=up_channels[i],
                                                     out_channels=up_channels[i+1],
                                                     kernel_size=4, stride=2, padding=1))

            self.ups.append(Block(in_channels=up_channels[i+1] * 2,
                                  out_channels=up_channels[i+1],
                                  time_embed_dim=time_embed_dim))

        # 6. Final Output Projection
        self.output = nn.Conv2d(up_channels[-1], img_channels, kernel_size=1)

    def forward(self, x, t, low_res_img):
        # Embed time
        t_emb = self.time_embed(t)

        # Initial convolution (Concat noise + condition)
        x = self.conv0(torch.cat([x, low_res_img], dim=1))

        # Save skip connections
        skips = []

        # --- Down Path ---
        for block, downsample in zip(self.downs, self.downsamples):
            x = block(x, t_emb)
            skips.append(x)      # Save High-Res Feature
            x = downsample(x)    # Downscale

        # --- Bottleneck ---
        x = self.mid_block(x, t_emb)

        # --- Up Path ---
        for upsample, block in zip(self.upsamples, self.ups):
            x = upsample(x)             # 1. Upscale
            skip = skips.pop()          # 2. Retrieve Skip

            # Ensure sizes match (handling potential rounding errors in sizing)
            if x.shape != skip.shape:
                x = F_transforms.resize(x, size=skip.shape[2:])

            x = torch.cat((x, skip), dim=1)
            x = block(x, t_emb)

        return self.output(x)