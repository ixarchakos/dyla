import os
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from config import LORA_WEIGHTS_PATH, OUTPUT_DIR, PRETRAINED_MODEL, PRETRAINED_MODEL_XL, device, torch_dtype


def generate_images(prompts: list, model_type: str = 'sdxl') -> None:
    """
    Generate images for given prompts using trained LoRA checkpoints.
    
    This function loads all available LoRA checkpoints and generates images for each
    prompt using the specified model type. Images are saved in separate directories
    for each checkpoint.
    
    Parameters
    ----------
    prompts : list
        List of text prompts to generate images for
    model_type : str, optional
        Type of model to use ('sd' for Stable Diffusion v1.5, 'sdxl' for SDXL), 
        default is 'sdxl'
        
    Returns
    -------
    None
        Generated images are saved to OUTPUT_DIR in checkpoint-specific subdirectories
        
    Notes
    -----
    - Creates separate output directory for each checkpoint
    - Uses 100 inference steps and guidance scale of 7.5
    - Images are saved as PNG files with sequential numbering
    - Model is loaded once and reused for all checkpoints
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(model_type)
    pipeline = AutoPipelineForText2Image.from_pretrained(
        PRETRAINED_MODEL if model_type == 'sd' else PRETRAINED_MODEL_XL, torch_dtype=torch_dtype
    ).to(device)
    
    for checkpoint in os.listdir(LORA_WEIGHTS_PATH):
        checkpoint_output_dir = os.path.join(OUTPUT_DIR, checkpoint)
        os.makedirs(checkpoint_output_dir, exist_ok=True)
    
        pipeline.load_lora_weights(f"{LORA_WEIGHTS_PATH}/{checkpoint}", weight_name="pytorch_lora_weights.safetensors")

        for i, prompt in enumerate(prompts):
            image = pipeline(prompt, num_inference_steps=100, guidance_scale=7.5).images[0]
            filename = f"generated_{i+1:03d}.png"
            filepath = os.path.join(checkpoint_output_dir, filename)
            image.save(filepath)
            print(f"Saved: {filepath}")


def infer(model_type: str = 'sdxl') -> None:
    """
    Run inference with predefined prompts to generate sample images.
    
    This function generates images for a set of predefined prompts using the
    trained LoRA checkpoints. It serves as a demonstration of the model's
    capabilities.
    
    Parameters
    ----------
    model_type : str, optional
        Type of model to use ('sd' for Stable Diffusion v1.5, 'sdxl' for SDXL), 
        default is 'sdxl'
        
    Returns
    -------
    None
        Generated images are saved to OUTPUT_DIR
        
    Notes
    -----
    - Uses predefined prompts related to Ioannis in different scenarios
    - Calls generate_images function with the predefined prompt set
    """
    prompts = [
        "Ioannis hiking in the mountains",
        "Ioannis at an office desk",
        "Ioannis skiing in the Alps"
    ]
    
    generate_images(prompts, model_type)

