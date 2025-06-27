import torch
import numpy as np
import os

# Device configuration
device = "cuda" if torch.cuda.is_available() else "mps"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Constants
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

# Project paths
PROJECT_PATH = ""

# Model configurations
PRETRAINED_MODEL = 'runwayml/stable-diffusion-v1-5'
PRETRAINED_MODEL_XL = 'stabilityai/stable-diffusion-xl-base-1.0'
LORA_WEIGHTS_PATH = os.path.join(PROJECT_PATH, 'sd-checkpoints-finetuned-lora/')
IMAGE_FOLDER = "ioannis_images"
DATA_DIR = os.path.join(PROJECT_PATH, IMAGE_FOLDER)
CSV_PATH = os.path.join(DATA_DIR, 'prompts.csv')
OUTPUT_DIR = 'generated_images'
