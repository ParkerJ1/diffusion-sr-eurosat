# Setup Guide: Converting Jupyter Notebook to Project Structure

## 📁 Project Structure Created

```
diffusion-sr-eurosat/
├── notebooks/              # Your Jupyter notebooks go here
│   └── DiffusionSR.ipynb  # Your original notebook (to be updated)
├── src/
│   ├── data/              
│   │   └── dataset.py      # ✅ CREATED - EuroSAT, MNIST, Flowers102 datasets
│   ├── models/            
│   │   ├── unet.py         # ⏳ TO CREATE - U-Net architecture
│   │   └── diffusion.py    # ⏳ TO CREATE - Noise scheduler
│   ├── training/          
│   │   ├── train.py        # ⏳ TO CREATE - Training loop
│   │   └── sample.py       # ⏳ TO CREATE - Sampling function
│   ├── evaluation/        
│   │   └── metrics.py      # ⏳ TO CREATE - PSNR, SSIM, LPIPS
│   └── utils/             
│       ├── visualization.py # ⏳ TO CREATE - Plotting functions
│       └── helpers.py      # ⏳ TO CREATE - Utility functions
├── configs/               
│   └── config.py           # ✅ CREATED - Configuration class
├── outputs/               # Model outputs (not in Git)
├── data/                  # Dataset storage (not in Git)
├── docs/                  # Experiment logs
└── tests/                 # Unit tests
```

---

## 🚀 Quick Start Instructions

### Step 1: Download This Entire Folder

You now have a `diffusion-sr-eurosat/` folder with:
- ✅ `configs/config.py` - Already created
- ✅ `src/data/dataset.py` - Already created
- ⏳ Other modules - **You need to create these**

### Step 2: What You Need to Do

I've prepared the structure for you, but you need to **extract code from your notebook** into the remaining Python files.

Here's what code goes where:

#### **A. Models (src/models/)**

**File: `src/models/unet.py`**
Extract from your notebook:
- `SinusoidalPositionalEmbeddings` class
- `TimeEmbedding` class  
- `DownBlock` class
- `UpBlock` class
- `ConditionalUNet` class

**File: `src/models/diffusion.py`**
Extract from your notebook:
- `Scheduler` class (the one with `get_linear_beta_scheduler`)

#### **B. Training (src/training/)**

**File: `src/training/train.py`**
Extract from your notebook:
- `train()` function (the one that takes model, dataloader, scheduler, optimizer, device, epoch)

**File: `src/training/sample.py`**
Extract from your notebook:
- `sampling()` function (the @torch.no_grad() decorated one)

#### **C. Evaluation (src/evaluation/)**

**File: `src/evaluation/metrics.py`**
Extract from your notebook:
- `metrics` class containing:
  - `PSNR` class
  - `SSIM` class
  - `LPIPS` class

#### **D. Utils (src/utils/)**

**File: `src/utils/visualization.py`**
Extract from your notebook:
- `visualize_and_save_samples()` function
- Any plotting/visualization helper functions

**File: `src/utils/helpers.py`**
Extract from your notebook:
- `find_latest_model()` function
- Any other utility functions

---

## 📝 Step-by-Step Extraction Process

### Example: Extracting the U-Net Model

**1. Open your notebook, find the U-Net code (Cell 11)**
**2. Copy the entire code**
**3. Create `src/models/unet.py`:**

```python
"""U-Net architecture for conditional diffusion model"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# PASTE YOUR CODE HERE
# Example:
class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        # ... your code ...
        pass

class ConditionalUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... your code ...
    
    def forward(self, x, t, low_res_img):
        # ... your code ...
        return x
```

**4. Update imports at the top to use config:**

```python
# Add this import
from configs.config import CONFIG

# Or pass config as parameter:
class ConditionalUNet(nn.Module):
    def __init__(self, config):
        self.config = config
        # Use config.IMG_CHANNELS instead of CONFIG.IMG_CHANNELS
```

---

## 🔧 Updating Your Notebook

Once you've extracted code to modules, update your notebook to import from them:

**Original notebook (messy):**
```python
# Cell 1: Imports
import torch
...

# Cell 2: Config class definition
class Config:
    ...

# Cell 3: Dataset class
class EuroSATSuperResData:
    ...

# Cell 4: Model class  
class ConditionalUNet:
    ...

# Cell 10: Training
for epoch in range(100):
    train(...)
```

**New notebook (clean):**
```python
# Cell 1: Setup
import sys
sys.path.append('..')  # Go up one level to project root

%load_ext autoreload
%autoreload 2

# Cell 2: Imports
from configs.config import CONFIG
from src.data.dataset import get_dataloader
from src.models.unet import ConditionalUNet
from src.models.diffusion import Scheduler
from src.training.train import train
from src.training.sample import sampling
from src.evaluation.metrics import metrics
from src.utils.visualization import visualize_and_save_samples

# Cell 3: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 4: Initialize
dataloader = get_dataloader(CONFIG)
model = ConditionalUNet(CONFIG).to(CONFIG.DEVICE)
scheduler = Scheduler()
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.LR)

# Cell 5: Training Loop
for epoch in range(CONFIG.EPOCHS):
    loss = train(model, dataloader, scheduler, optimizer, CONFIG.DEVICE, epoch)
    
    if (epoch + 1) % CONFIG.SAMPLE_EVERY == 0:
        samples = sampling(model, val_images, scheduler)
        visualize_and_save_samples(samples, epoch)
```

---

## 🌐 Setting Up GitHub

### Step 1: Create Repository on GitHub
1. Go to github.com → New Repository
2. Name: `diffusion-sr-eurosat`
3. Make it **Private** (for now)
4. Don't initialize with README (you have one)

### Step 2: Initialize Git Locally

```bash
# On your local machine (not Colab)
cd path/to/diffusion-sr-eurosat/

# Initialize Git
git init
git add .
git commit -m "Initial commit: Project structure setup"

# Connect to GitHub (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/diffusion-sr-eurosat.git
git branch -M main
git push -u origin main
```

### Step 3: Load in Colab

**In Colab notebook:**
```python
# Clone from GitHub
!git clone https://github.com/YOUR_USERNAME/diffusion-sr-eurosat.git
%cd diffusion-sr-eurosat

# Install dependencies
!pip install -r requirements.txt

# Mount Drive for data
from google.colab import drive
drive.mount('/content/drive')

# Now run your code!
```

**When you make changes locally:**
```bash
git add .
git commit -m "Add attention layer"
git push
```

**In Colab to get updates:**
```python
%cd diffusion-sr-eurosat
!git pull origin main
```

---

## ✅ Checklist

- [ ] Download `diffusion-sr-eurosat/` folder to your local machine
- [ ] Extract remaining code from notebook to Python files:
  - [ ] `src/models/unet.py`
  - [ ] `src/models/diffusion.py`
  - [ ] `src/training/train.py`
  - [ ] `src/training/sample.py`
  - [ ] `src/evaluation/metrics.py`
  - [ ] `src/utils/visualization.py`
  - [ ] `src/utils/helpers.py`
- [ ] Create `requirements.txt` with all dependencies
- [ ] Update notebook to import from modules
- [ ] Test locally that imports work
- [ ] Create GitHub repository
- [ ] Push to GitHub
- [ ] Test loading from GitHub in Colab
- [ ] Update your workflow to: edit locally → commit → push → pull in Colab

---

## 🆘 Need Help?

If you get stuck:
1. Start with just extracting **config** (already done ✅)
2. Then extract **one module** at a time
3. Test after each extraction
4. Commit after each working module

Don't try to do everything at once!

---

## 📚 Next Steps After Setup

Once your structure is working:
1. Add experiment logging in `docs/`
2. Create different config files for experiments
3. Add unit tests gradually
4. Keep improving the structure as needed

**Remember:** This is an iterative process. You don't need perfect structure on day 1!
