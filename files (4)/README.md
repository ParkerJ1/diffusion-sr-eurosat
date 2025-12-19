# Diffusion-based Super-Resolution for EuroSAT

Research project implementing diffusion models for 13-channel satellite imagery super-resolution.

## 🎯 Project Goals

- Implement diffusion-based super-resolution for multispectral satellite imagery
- Target: 22-24 dB PSNR for journal publication
- Demonstrate advantages over traditional SR methods for remote sensing applications

## 📊 Current Results

| Configuration | PSNR | SSIM | Epoch | Notes |
|--------------|------|------|-------|-------|
| Baseline (3-layer U-Net) | 19.31 dB | 0.7379 | 55 | Best result so far |

See `docs/experiment_log.md` for detailed experimental results.

## 🚀 Setup

### Local Development
```bash
git clone https://github.com/YOUR_USERNAME/diffusion-sr-eurosat.git
cd diffusion-sr-eurosat
pip install -r requirements.txt
```

### Google Colab
```python
!git clone https://github.com/YOUR_USERNAME/diffusion-sr-eurosat.git
%cd diffusion-sr-eurosat
!pip install -r requirements.txt

# Mount Drive for data and checkpoints
from google.colab import drive
drive.mount('/content/drive')
```

## 📁 Project Structure

```
diffusion-sr-eurosat/
├── notebooks/          # Jupyter notebooks for experiments
├── configs/            # Configuration files
├── src/
│   ├── data/          # Dataset classes
│   ├── models/        # U-Net, diffusion scheduler
│   ├── training/      # Training and sampling functions
│   ├── evaluation/    # Metrics (PSNR, SSIM, LPIPS)
│   └── utils/         # Visualization and helpers
├── outputs/           # Model checkpoints, samples, logs
├── data/              # Dataset storage
└── docs/              # Experiment logs and notes
```

## 📖 Usage

### Training
```python
from configs.config import CONFIG
from src.data.dataset import get_dataloader
from src.models.unet import ConditionalUNet
from src.training.train import train

# Initialize
dataloader = get_dataloader(CONFIG)
model = ConditionalUNet(CONFIG).to(CONFIG.DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.LR)

# Train
for epoch in range(CONFIG.EPOCHS):
    loss = train(model, dataloader, scheduler, optimizer, CONFIG.DEVICE, epoch)
```

See `notebooks/` for complete examples.

## 🔬 Planned Improvements

### Phase 1: Quick Wins
- [ ] Add self-attention layers to U-Net
- [ ] Implement learning rate scheduling (cosine annealing)
- [ ] Implement DDIM sampling for improved quality
- [ ] Add perceptual loss using TorchGeo ResNet50

### Phase 2: Comprehensive Evaluation
- [ ] Baseline comparisons (bicubic, SRCNN, deterministic U-Net)
- [ ] Downstream task validation (segmentation IoU)
- [ ] Data augmentation
- [ ] Ablation studies

### Phase 3: Publication
- [ ] Comprehensive evaluation on 100+ test samples
- [ ] Statistical significance testing
- [ ] Code cleanup and documentation
- [ ] Paper writing

## 📚 Dataset

**EuroSAT**: 13-band multispectral Sentinel-2 satellite imagery
- Download: [EuroSAT Repository](https://github.com/phelber/EuroSAT)
- 27,000 labeled images across 10 land-use classes
- Resolution: 64×64 pixels per band

See `data/README.md` for setup instructions.

## 🤝 Contributing

This is a research project. For collaboration inquiries, please open an issue.

## 📄 License

[To be determined]

## 📧 Contact

Jonathan - PhD Candidate, UKZN
Research Focus: ML for Weather Downscaling & Remote Sensing

## 🙏 Acknowledgments

- TorchGeo library for Sentinel-2 pre-trained models
- EuroSAT dataset creators
- Anthropic Claude for code organization assistance
