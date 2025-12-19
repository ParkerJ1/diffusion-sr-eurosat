"""Configuration for diffusion model experiments"""
import os
import torch
from datetime import datetime


class BaseConfig:
    """Base configuration for diffusion model"""
    
    def __init__(self):
        # Dataset selection
        self.DATASET = "EuroSAT"  # Options: "EuroSAT", "MNIST", "Flowers102"
        
        # Training vs Inference mode
        self.TRAIN = True  # Set to False to load existing model
        self.MODEL_PATH = None  # Will auto-detect most recent if None
        
        # Diffusion hyperparameters
        self.BETA_START = 0.0001
        self.BETA_STOP = 0.02
        self.TIMESTEPS = 1000
        
        # Model hyperparameters
        self.IMG_SIZE = 64
        self.IMG_SIZE_LOW = 16
        
        # Set channels based on dataset
        if self.DATASET == "MNIST":
            self.IMG_CHANNELS = 1
        elif self.DATASET == "EuroSAT":
            self.IMG_CHANNELS = 13
        else:
            self.IMG_CHANNELS = 3
            
        self.TIME_EMBED_DIM = 128
        self.WEIGHT_INIT_GAIN = 0.5
        self.WEIGHT_DECAY = 1e-4
        
        # U-Net architecture
        self.DOWN_CHANNELS = (128, 256, 512)
        self.UP_CHANNELS = (512, 256, 128)
        
        # Training hyperparameters
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = 128
        self.LR = 5e-5
        self.EPOCHS = 100
        self.SAMPLE_EVERY = 5
        self.NUM_EVAL_SAMPLES = 16
        self.GRADIENT_CLIP = 1.0
        
        # Paths (Colab default)
        self.PROJECT_ROOT = f"/content/drive/MyDrive/SharedColab/DiffusionSR_{self.DATASET}/"
        self.DATASET_PATH = os.path.join(self.PROJECT_ROOT, "data/")
        self.LOCAL_DATASET_PATH = "/data/"
        self.STATS_FILE = os.path.join(self.DATASET_PATH, "eurosat_stats.pt")
        
        # Output directory with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.OUTPUT_DIR = os.path.join(self.PROJECT_ROOT, f"output_{self.timestamp}/")


class ExperimentConfig(BaseConfig):
    """Configuration for specific experiments"""
    
    def __init__(self):
        super().__init__()
        # Override defaults for experiments
        # Example: self.EPOCHS = 200


# Default config instance
CONFIG = BaseConfig()
