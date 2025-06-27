from src.preprocessing import resize_images
from src.train_sd_lora import train
from src.train_sdxl_lora import train_sdxl
from src.inference import infer
from src.blip import get_captions
from config import DATA_DIR
import argparse


def main() -> None:
    """
    Main function that handles command line arguments and orchestrates the pipeline.
    
    Supports three main operations:
    - preprocess: Resize images and generate captions
    - training:
        - train: Train the LoRA checkpoints for Stable Diffusion v1.5
        - train_sdxl: Train the LoRA checkpoints for Stable Diffusion XL
    - inference: Generate images using the trained checkpoints
    - model_type: 'sdxl' if you run the stable diffusion xl
    
    Parameters
    ----------
    None. Uses command line arguments for configuration
        
    Returns
    -------
    None
        Executes the specified operation based on command line arguments
        
    Raises
    ------
    SystemExit
        If no valid operation is specified
    """
    parser = argparse.ArgumentParser(description="Stable Diffusion LoRA Training and Inference")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess images")
    parser.add_argument("--train", action="store_true", help="Train the checkpoints")
    parser.add_argument("--train_sdxl", action="store_true", help="Train the checkpoints")
    parser.add_argument("--inference", action="store_true", help="Run inference")
    parser.add_argument("--model_type", type=str, default="sdxl", help="Model type")
    
    args = parser.parse_known_args()[0]
    
    if args.preprocess:
        resize_images(DATA_DIR, (1024, 1024))
        get_captions()
    elif args.train:
        train()
    elif args.train_sdxl:
        train_sdxl()
    elif args.inference:
        infer(args.model_type)
    else:
        print("Please specify a correct functionality")
        exit()


if __name__ == "__main__":
    main()
