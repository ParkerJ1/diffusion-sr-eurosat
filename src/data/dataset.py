"""Dataset classes for diffusion model training"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F_transforms
from torchgeo.datasets import EuroSAT
from tqdm import tqdm


class EuroSATSuperResData(Dataset):
    """EuroSAT dataset for super-resolution with 13-band multispectral imagery"""
    
    def __init__(self, config):
        self.config = config
        self.bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 
                      'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
        
        # Load dataset
        self.dataset_local = EuroSAT(
            root=config.LOCAL_DATASET_PATH,
            download=True,
            bands=self.bands
        )
        
        # Load or compute normalization statistics
        n_channels = len(self.bands)
        stats_file = config.STATS_FILE
        loaded_stats = False
        
        while not loaded_stats:
            try:
                stats = torch.load(stats_file)
                self.channel_means = stats['means'].view(n_channels, 1, 1)
                self.channel_stds = stats['stds'].view(n_channels, 1, 1)
                self.channel_2 = stats['q_2'].view(n_channels, 1, 1)
                self.channel_98 = stats['q_98'].view(n_channels, 1, 1)
                print("Loaded channel statistics for normalization.")
                loaded_stats = True
            except FileNotFoundError:
                print(f"ERROR: stats file not found at {stats_file}. Recalculating stats...")
                self.analyse_stats()
    
    def __len__(self):
        return len(self.dataset_local)
    
    def __getitem__(self, index):
        sample = self.dataset_local[index]
        image = sample["image"]
        
        # Clip to 2-98 percentile and normalize to [-1, 1]
        image = torch.clamp(image, self.channel_2, self.channel_98)
        norm_image = 2 * (image - self.channel_2) / (self.channel_98 - self.channel_2 + 1e-6) - 1
        
        # Create high-res and low-res pairs
        high_res_img = norm_image
        low_res_img = F_transforms.resize(
            high_res_img,
            size=[self.config.IMG_SIZE_LOW, self.config.IMG_SIZE_LOW],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True
        )
        low_res_img = F_transforms.resize(
            low_res_img,
            size=[self.config.IMG_SIZE, self.config.IMG_SIZE],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True
        )
        
        return (high_res_img, low_res_img)
    
    def analyse_stats(self):
        """Compute and save dataset statistics"""
        loader = DataLoader(
            dataset=self.dataset_local,
            batch_size=self.config.BATCH_SIZE,
            num_workers=0,
            pin_memory=True
        )
        
        # Load all data
        all_data = []
        print("Iterate through dataset and load into tensor")
        for batch in tqdm(loader):
            all_data.append(batch["image"])
        all_data = torch.cat(all_data, dim=0)
        print(f"all_data shape: {all_data.shape}")
        
        # Compute statistics
        channel_means = all_data.mean(dim=[0, 2, 3])
        channel_stds = all_data.std(dim=[0, 2, 3])
        channel_mins = all_data.amin(dim=[0, 2, 3])
        channel_maxs = all_data.amax(dim=[0, 2, 3])
        
        # Compute quantiles on subset
        num_images = all_data.shape[0]
        subset_size = min(1000, num_images)
        indices = torch.randperm(num_images)[:subset_size]
        stats_data = all_data[indices]
        flat_data = stats_data.permute(1, 0, 2, 3).reshape(self.config.IMG_CHANNELS, -1)
        
        channel_2 = flat_data.quantile(q=0.02, dim=1)
        channel_98 = flat_data.quantile(q=0.98, dim=1)
        
        # Print statistics
        print(f"{'Band':<5} | {'Mean':<10} | {'Std Dev':<10} | {'Min':<10} | {'q_2':<10} | {'q_98':<10} | {'Max':<10}")
        for i, band_name in enumerate(self.bands):
            print(f"{band_name:<5} | {channel_means[i]:<10.2f} | {channel_stds[i]:<10.2f} | "
                  f"{channel_mins[i]:<10.2f} | {channel_2[i]:<10.2f} | {channel_98[i]:<10.2f} | {channel_maxs[i]:<10.2f}")
        
        # Save statistics
        stats_file = self.config.STATS_FILE
        stats_dict = {
            'means': channel_means,
            'stds': channel_stds,
            'q_2': channel_2,
            'q_98': channel_98
        }
        torch.save(stats_dict, stats_file)
        print(f"Statistics saved to {stats_file}")


class MNISTSuperResData(Dataset):
    """MNIST dataset for super-resolution testing"""
    
    def __init__(self, config):
        self.config = config
        self.dataset = datasets.MNIST(
            root=config.DATASET_PATH,
            train=True,
            download=True,
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, _ = self.dataset[index]
        
        # Convert to tensor and normalize to [-1, 1]
        image_tensor = transforms.ToTensor()(image)
        norm_image = 2 * image_tensor - 1
        
        # Resize to target size
        high_res_img = F_transforms.resize(
            norm_image,
            size=[self.config.IMG_SIZE, self.config.IMG_SIZE],
            interpolation=InterpolationMode.BILINEAR
        )
        
        # Create low-res version
        low_res_img = F_transforms.resize(
            high_res_img,
            size=[self.config.IMG_SIZE_LOW, self.config.IMG_SIZE_LOW],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True
        )
        low_res_img = F_transforms.resize(
            low_res_img,
            size=[self.config.IMG_SIZE, self.config.IMG_SIZE],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True
        )
        
        return (high_res_img, low_res_img)


class Flowers102SuperResData(Dataset):
    """Flowers102 dataset for super-resolution testing"""
    
    def __init__(self, config):
        self.config = config
        self.dataset = datasets.Flowers102(
            root=config.DATASET_PATH,
            split='train',
            download=True,
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, _ = self.dataset[index]
        
        # Convert to tensor and normalize to [-1, 1]
        transform = transforms.Compose([
            transforms.Resize(self.config.IMG_SIZE),
            transforms.CenterCrop(self.config.IMG_SIZE),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image)
        norm_image = 2 * image_tensor - 1
        
        high_res_img = norm_image
        
        # Create low-res version
        low_res_img = F_transforms.resize(
            high_res_img,
            size=[self.config.IMG_SIZE_LOW, self.config.IMG_SIZE_LOW],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True
        )
        low_res_img = F_transforms.resize(
            low_res_img,
            size=[self.config.IMG_SIZE, self.config.IMG_SIZE],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True
        )
        
        return (high_res_img, low_res_img)


def get_dataloader(config):
    """Get dataloader based on config"""
    if config.DATASET == "EuroSAT":
        dataset = EuroSATSuperResData(config)
    elif config.DATASET == "MNIST":
        dataset = MNISTSuperResData(config)
    elif config.DATASET == "Flowers102":
        dataset = Flowers102SuperResData(config)
    else:
        raise ValueError(f"Unknown dataset: {config.DATASET}")
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader
