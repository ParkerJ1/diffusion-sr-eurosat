import torch
import torch.nn as nn
import numpy as np
from torchvision import models

class metrics:
    class PSNR:
        """
        Peak Signal-to-Noise Ratio metric.

        Usage:
            psnr_metric = PSNR(data_range=1.0)
            psnr_value = psnr_metric(generated, ground_truth)
            print(f"PSNR: {psnr_value:.2f} dB")
        """
        def __init__(self, data_range=1.0):
            """
            Args:
                data_range: Maximum possible pixel value (1.0 for normalized images)
            """
            self.data_range = data_range

        def __call__(self, generated, ground_truth):
            """
            Compute PSNR between generated and ground truth images.

            Args:
                generated: Tensor [B, C, H, W] in range [0, data_range]
                ground_truth: Tensor [B, C, H, W] in range [0, data_range]

            Returns:
                PSNR in dB (higher is better). Typical range: 20-40 dB for SR
            """
            mse = torch.mean((generated - ground_truth) ** 2)
            if mse == 0:
                return float('inf')
            psnr = 20 * torch.log10(torch.tensor(self.data_range) / torch.sqrt(mse))
            return psnr.item()

    class SSIM:
        """
        Structural Similarity Index metric.

        Usage:
            ssim_metric = SSIM(window_size=11, data_range=1.0)
            ssim_value = ssim_metric(generated, ground_truth)
            print(f"SSIM: {ssim_value:.4f}")
        """
        def __init__(self, window_size=11, data_range=1.0):
            """
            Args:
                window_size: Size of Gaussian window (default: 11)
                data_range: Maximum possible pixel value
            """
            self.window_size = window_size
            self.data_range = data_range
            self.window = None
            self.channel = None

        def _gaussian_kernel(self, window_size, sigma=1.5):
            """Create 1D Gaussian kernel."""
            gauss = torch.tensor([
                np.exp(-(x - window_size//2)**2 / (2*sigma**2))
                for x in range(window_size)
            ], dtype=torch.float32)
            return gauss / gauss.sum()

        def _create_window(self, channel, device):
            """Create 2D Gaussian window for SSIM."""
            _1D_window = self._gaussian_kernel(self.window_size).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(channel, 1, self.window_size, self.window_size).contiguous()
            return window.to(device)

        def __call__(self, generated, ground_truth):
            """
            Compute SSIM between generated and ground truth images.

            Args:
                generated: Tensor [B, C, H, W] in range [0, data_range]
                ground_truth: Tensor [B, C, H, W] in range [0, data_range]

            Returns:
                SSIM value in [0, 1] (higher is better). >0.9 is good for SR
            """
            channel = generated.size(1)

            # Create window if needed
            if self.window is None or self.channel != channel:
                self.window = self._create_window(channel, generated.device)
                self.channel = channel

            # Compute local means
            mu1 = torch.nn.functional.conv2d(generated, self.window, padding=self.window_size//2, groups=channel)
            mu2 = torch.nn.functional.conv2d(ground_truth, self.window, padding=self.window_size//2, groups=channel)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            # Compute local variances and covariance
            sigma1_sq = torch.nn.functional.conv2d(
                generated * generated, self.window, padding=self.window_size//2, groups=channel) - mu1_sq
            sigma2_sq = torch.nn.functional.conv2d(
                ground_truth * ground_truth, self.window, padding=self.window_size//2, groups=channel) - mu2_sq
            sigma12 = torch.nn.functional.conv2d(
                generated * ground_truth, self.window, padding=self.window_size//2, groups=channel) - mu1_mu2

            # SSIM formula
            C1 = (0.01 * self.data_range) ** 2
            C2 = (0.03 * self.data_range) ** 2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                    ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

            return ssim_map.mean().item()

    class LPIPS:
        """
        Learned Perceptual Image Patch Similarity using VGG16 features.

        Usage:
            lpips_metric = LPIPS(device='cuda')
            lpips_value = lpips_metric(generated, ground_truth)
            print(f"LPIPS: {lpips_value:.4f}")

        Note: For multispectral images, only first 3 channels are used.
        """
        def __init__(self, device='cuda'):
            """
            Args:
                device: Device to run computation on ('cuda' or 'cpu')
            """
            self.device = device
            self._build_model()

        def _build_model(self):
            """Build VGG16 feature extractor."""


            vgg = models.vgg16(pretrained=True).features.eval().to(self.device)

            # Extract features from specific layers
            self.layers = nn.ModuleList([
                vgg[:4],   # relu1_2
                vgg[4:9],  # relu2_2
                vgg[9:16], # relu3_3
                vgg[16:23],# relu4_3
                vgg[23:30] # relu5_3
            ])

            # Freeze all parameters
            for param in self.layers.parameters():
                param.requires_grad = False

            # Layer weights (equal weighting)
            self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]

            # ImageNet normalization
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        def _preprocess(self, x):
            """Preprocess images for VGG."""
            # Handle multispectral: take first 3 channels as RGB
            if x.size(1) > 3:
                x = x[:, :3, :, :]
            elif x.size(1) == 1:
                # Grayscale to RGB
                x = x.repeat(1, 3, 1, 1)

            # Normalize to ImageNet stats
            x = (x - self.mean) / self.std
            return x

        def _extract_features(self, x):
            """Extract features from all VGG layers."""
            features = []
            for layer in self.layers:
                x = layer(x)
                features.append(x)
            return features

        def __call__(self, generated, ground_truth):
            """
            Compute LPIPS distance between generated and ground truth images.

            Args:
                generated: Tensor [B, C, H, W] in range [0, 1]
                ground_truth: Tensor [B, C, H, W] in range [0, 1]

            Returns:
                LPIPS distance (lower is better). <0.2 is good for SR
            """
            with torch.no_grad():
                # Preprocess
                gen_norm = self._preprocess(generated)
                gt_norm = self._preprocess(ground_truth)

                # Extract features
                gen_features = self._extract_features(gen_norm)
                gt_features = self._extract_features(gt_norm)

                # Compute perceptual distance
                lpips_value = 0.0
                for w, f_gen, f_gt in zip(self.weights, gen_features, gt_features):
                    # Normalize features (channel-wise L2 normalization)
                    f_gen_norm = f_gen / (torch.norm(f_gen, dim=1, keepdim=True) + 1e-10)
                    f_gt_norm = f_gt / (torch.norm(f_gt, dim=1, keepdim=True) + 1e-10)

                    # Compute squared difference
                    diff = (f_gen_norm - f_gt_norm) ** 2
                    lpips_value += w * diff.mean()

                return lpips_value.item()