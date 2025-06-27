## Overview

This project implements fine-tuning of Stable Diffusion models (v1.5 and XL) using LoRA (Low-Rank Adaptation) to create personalized image generation models. 
It includes training, inference, preprocessing utilities, and a web interface for working with custom image datasets.

### Demo: [![Generic badge][logo-hf_spaces]][hf_spaces]

[hf_spaces]: https://huggingface.co/spaces/ixarchakos/my-lora

[logo-hf_spaces]: https://img.shields.io/badge/🤗-Demo-blue.svg?style=plastic


## Project Structure

```
Dyla/
├── main.py                 # Main entry point with CLI interface
├── app.py                  # Gradio web application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── ioannis_images/       # Training images and prompts.csv
├── src/
│   ├── preprocessing.py   # Image preprocessing utilities
│   ├── blip.py           # BLIP caption generation
│   ├── train_sd_lora.py  # SD v1.5 LoRA training
│   ├── train_sdxl_lora.py # SDXL LoRA training
│   └── inference.py      # Image generation utilities
└── generated_images/     # Output directory for generated images
```

## Configuration

The `config.py` file contains all configuration settings:

- Device settings (CUDA/MPS)
- Model paths and versions
- Project directories
- Training parameters

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Dyla
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Set the following variable in config.py **:
   ```
   PROJECT_PATH = ""
   ```

4.  **Set up WandB** (optional but recommended):
   ```bash
   wandb login
   ```

## Dataset Preparation

### Image Dataset Structure

The project expects images to be organized in the `ioannis_images/` directory with a corresponding `prompts.csv` file:

```
ioannis_images/
├── IMG_6139.jpeg
├── IMG_6140.jpeg
├── ...
└── prompts.csv
```

### CSV Format

The `prompts.csv` file should contain two columns:
- `image`: Image filename (e.g., "img1.jpg")
- `text`: Corresponding caption/prompt

Example:
```csv
image,text
IMG_6139.jpeg,Ioannis is a man with a beard and a black shirt stading in front of the window
IMG_6140.jpeg,Ioannis is a man with a beard and a black shirt standing in front of the kitchen
```

## Usage

### Preprocessing

Preprocess images and generate captions:

```bash
python main.py --preprocess
```

This will:
- Resize all images to 1024x1024 pixels
- Crop images to square format
- Generate captions using BLIP model
- Save captions to `prompts.csv`

### Training

#### Stable Diffusion v1.5 Training

Run the training script with default parameters (on multiple GPU):

```bash
python main.py --train
```

Or use accelerate for distributed training:

```bash
accelerate launch main.py --train
```

#### Stable Diffusion XL Training

Train SDXL model with LoRA:

```bash
python main.py --train_sdxl
```

Or use accelerate for distributed training:

```bash
accelerate launch main.py --train_sdxl
```

#### Training Parameters

Key parameters you can customize:

- `--data_path`: Path to training data directory
- `--output_dir`: Output directory for checkpoints
- `--mixed_precision`: Training weight precision
- `--train_batch_size`: Batch size for training
- `--learning_rate`: Learning rate
- `--resolution`: Image resolution (default: 512 for SD, 1024 for SDXL)
- `--checkpointing_steps`: Save checkpoint every N steps

Example with custom parameters (on multiple gpus):
```bash
accelerate launch main.py --train_sdxl --mixed_precision=bf16 --train_batch_size 4 --resolution 768
```

### Inference

Generate images using trained checkpoints:

```bash
python main.py --inference --model_type sdxl
```

The inference script will:
- Load all checkpoints from the LoRA weights directory
- Create separate folders for each checkpoint
- Generate images for predefined prompts
- Save results in `generated_images/` directory

### Web Interface

Launch the Gradio web interface:

```bash
python app.py
```

Features:
- Interactive image generation
- Support for both SD v1.5 and SDXL models
- Advanced generation parameters
- Example prompts
- Real-time image preview
