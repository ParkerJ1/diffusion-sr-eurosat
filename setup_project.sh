#!/bin/bash
# Project Setup Script
# This script creates the full project structure

echo "Creating diffusion-sr-eurosat project structure..."

# Create main directory
mkdir -p diffusion-sr-eurosat
cd diffusion-sr-eurostat

# Create all subdirectories
mkdir -p notebooks
mkdir -p src/{data,models,training,evaluation,utils}
mkdir -p configs
mkdir -p outputs/{checkpoints,samples,logs}
mkdir -p data
mkdir -p docs
mkdir -p tests

# Create __init__.py files for Python packages
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/training/__init__.py
touch src/evaluation/__init__.py
touch src/utils/__init__.py

# Create .gitkeep files for empty directories
touch outputs/checkpoints/.gitkeep
touch outputs/samples/.gitkeep
touch outputs/logs/.gitkeep
touch data/.gitkeep
touch docs/.gitkeep
touch tests/.gitkeep

echo "✅ Directory structure created!"
echo ""
echo "Next steps:"
echo "1. Download the files I provided and place them in the correct locations"
echo "2. Copy your original notebook to notebooks/"
echo "3. Read START_HERE.md for the complete setup guide"
echo ""
echo "Project created in: $(pwd)"
