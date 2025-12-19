# Quick Code Extraction Guide

## 🎯 Goal
Extract code from `DiffusionSR_original.ipynb` into Python modules

## 📝 Extraction Map

### Cell 2 → `configs/config.py` ✅ DONE
The Config class has been extracted

### Cell 3 → `src/utils/helpers.py`
```python
def find_latest_model(project_root):
    # Copy from Cell 3
```

### Cell 4 → `src/models/diffusion.py` 
```python
class Scheduler:
    def get_linear_beta_scheduler(self, timesteps, device):
        # Copy from Cell 4
```

### Cells 5, 6, 7, 8 → `src/data/dataset.py` ✅ DONE
Dataset classes have been extracted

### Cell 9 → `src/evaluation/metrics.py`
```python
class metrics:
    class PSNR:
        # Copy from Cell 9
    class SSIM:
        # Copy from Cell 9
    class LPIPS:
        # Copy from Cell 9
```

### Cell 10 → `src/models/unet.py`
```python
class SinusoidalPositionalEmbeddings(nn.Module):
    # Copy from Cell 10
    
class TimeEmbedding(nn.Module):
    # Copy from Cell 10
    
class DownBlock(nn.Module):
    # Copy from Cell 10
    
class UpBlock(nn.Module):
    # Copy from Cell 10
    
class ConditionalUNet(nn.Module):
    # Copy from Cell 10
```

### Cell 11 → `src/training/train.py`
```python
def train(model, dataloader, scheduler, optimizer, device, epoch):
    # Copy from Cell 11
```

### Cell 12 → `src/training/sample.py`
```python
@torch.no_grad()
def sampling(model, low_res_imgs, scheduler):
    # Copy from Cell 12
```

### Cell 13 → `src/utils/visualization.py`
```python
def visualize_and_save_samples(generated_images, val_high_res_img, ...):
    # Copy from Cell 13
```

### Cell 14 → Keep in notebook
This is the main() function - orchestration logic that should stay in notebook

## 🔧 Modifications Needed

When copying code, make these changes:

### 1. Replace `CONFIG` with `config` parameter
**Before:**
```python
class ConditionalUNet(nn.Module):
    def __init__(self):
        self.channels = CONFIG.IMG_CHANNELS
```

**After:**
```python
class ConditionalUNet(nn.Module):
    def __init__(self, config):
        self.channels = config.IMG_CHANNELS
```

### 2. Add imports at top of each file
```python
import torch
import torch.nn as nn
# ... other imports as needed
```

### 3. Remove cell-specific code
- Remove `!pip install` commands
- Remove `drive.mount()` calls
- Remove print statements that are just for debugging

## ✅ Verification

After extraction, test that you can import:
```python
from configs.config import CONFIG
from src.data.dataset import get_dataloader
from src.models.unet import ConditionalUNet
from src.models.diffusion import Scheduler
from src.training.train import train
from src.training.sample import sampling
```

If imports work, you're done! 🎉
