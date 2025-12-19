import torch
from torchvision.utils import make_grid, save_image


def visualize_and_save_samples(generated_images, val_high_res_img, val_low_res_img,
                                epoch, config, dataset_stats=None):
    """
    Denormalize and visualize samples based on dataset type.

    Args:
        generated_images: Tensor in [0, 1] range from sampling
        val_high_res_img: Tensor in [-1, 1] range from dataset
        val_low_res_img: Tensor in [-1, 1] range from dataset
        epoch: Current epoch number
        dataset_stats: Dict with 'channel_2' and 'channel_98' for EuroSAT, None for others
    """

    if config.DATASET == "EuroSAT":
        # EuroSAT: Convert from normalized space to raw values, then to [0,1] for visualization
        channel_2 = dataset_stats['channel_2']
        channel_98 = dataset_stats['channel_98']

        # All three are now in [-1, 1] space
        gen_unnorm = ((generated_images + 1) / 2) * (channel_98 - channel_2) + channel_2
        high_unnorm = ((val_high_res_img + 1) / 2) * (channel_98 - channel_2) + channel_2
        low_unnorm = ((val_low_res_img + 1) / 2) * (channel_98 - channel_2) + channel_2

        # Map to [0, 1] for visualization using percentile clipping
        gen_vis = (gen_unnorm - channel_2) / (channel_98 - channel_2 + 1e-6)
        high_vis = (high_unnorm - channel_2) / (channel_98 - channel_2 + 1e-6)
        low_vis = (low_unnorm - channel_2) / (channel_98 - channel_2 + 1e-6)

        gen_vis = torch.clamp(gen_vis, 0.0, 1.0)
        high_vis = torch.clamp(high_vis, 0.0, 1.0)
        low_vis = torch.clamp(low_vis, 0.0, 1.0)

        # Select RGB bands (B4, B3, B2 = indices 3, 2, 1)
        # rgb_indices = [0, 1, 2]
        rgb_indices = [3, 2, 1]
        low_res_rgb = low_vis[:, rgb_indices, :, :]
        generated_rgb = gen_vis[:, rgb_indices, :, :]
        high_res_rgb = high_vis[:, rgb_indices, :, :]

    elif config.DATASET in ["MNIST", "CIFAR10", "Flowers102"]:
        # All images now in [-1, 1] space (after removing sampling conversion)
        # Convert all from [-1, 1] to [0, 1]
        generated_rgb = (generated_images + 1) / 2
        high_res_rgb = (val_high_res_img + 1) / 2
        low_res_rgb = (val_low_res_img + 1) / 2

        # Clamp to [0, 1]
        generated_rgb = torch.clamp(generated_rgb, 0.0, 1.0)
        high_res_rgb = torch.clamp(high_res_rgb, 0.0, 1.0)
        low_res_rgb = torch.clamp(low_res_rgb, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown dataset: {config.DATASET}")

    # Create grid: [low_res, generated, high_res] arranged in rows
    samples = torch.cat([low_res_rgb, generated_rgb, high_res_rgb])
    grid = make_grid(samples, nrow=config.GENERATE_SAMPLES, normalize=False)
    save_image(grid, f"{config.OUTPUT_DIR}/sample_epoch_{epoch+1}.png")
    print(f"Saved samples to: {config.OUTPUT_DIR}/sample_epoch_{epoch+1}.png")
