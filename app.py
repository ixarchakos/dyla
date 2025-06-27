import os

import gradio as gr
import random
import spaces
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
import torch
from config import device, torch_dtype, MAX_SEED, PRETRAINED_MODEL_XL, PRETRAINED_MODEL

pipe1 = AutoPipelineForText2Image.from_pretrained(
    PRETRAINED_MODEL_XL, torch_dtype=torch_dtype
).to(device)

pipe2 = AutoPipelineForText2Image.from_pretrained(
    PRETRAINED_MODEL, torch_dtype=torch_dtype
).to(device)


@spaces.GPU
def infer(
        prompt: str,
        negative_prompt: str,
        seed: int,
        randomize_seed: bool,
        width: int,
        height: int,
        guidance_scale: float,
        num_inference_steps: int,
        model_name: str,
        checkpoint: str,
        progress=gr.Progress(track_tqdm=True),
):
    """
    Generate an image using the specified parameters and LoRA checkpoint.
    
    This function loads a LoRA checkpoint and generates an image based on the provided
    prompt and generation parameters. It supports both Stable Diffusion v1.5 and SDXL models.
    
    Parameters
    ----------
    prompt : str
        Text prompt describing the image to generate
    negative_prompt : str
        Text prompt describing what to avoid in the generated image
    seed : int
        Random seed for reproducible generation
    randomize_seed : bool
        Whether to use a random seed instead of the provided one
    width : int
        Width of the generated image
    height : int
        Height of the generated image
    guidance_scale : float
        Strength of prompt guidance (higher values = stronger adherence to prompt)
    num_inference_steps : int
        Number of denoising steps for generation
    model_name : str
        Name of the base model to use
    checkpoint : str
        Name of the LoRA checkpoint to load
    progress : gr.Progress, optional
        Gradio progress tracker
        
    Returns
    -------
    tuple
        Tuple containing (generated_image, seed_used)
        
    Notes
    -----
    - Automatically selects the appropriate pipeline based on model_name
    - Loads LoRA weights from the checkpoints directory
    - Uses torch.Generator for reproducible results
    - Returns both the generated image and the seed used
    """
    pipe = pipe2 if model_name == 'runwayml/stable-diffusion-v1-5' else pipe1
    pipe.load_lora_weights(os.path.join('checkpoints', checkpoint),
                           weight_name="pytorch_lora_weights.safetensors")
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator().manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
    ).images[0]

    return image, seed


def main():
    """
    Create and configure the Gradio web interface.
    
    This function sets up the complete Gradio interface with all necessary components
    for image generation, including input controls, output display, and example prompts.

    """
    examples = [
        "Ioannis hiking in the mountains looking at the camera",
        "Ioannis at an office desk",
        "Ioannis skiing in the Alps looking at the camera without mask",
    ]

    css = """
    #col-container {
        margin: 0 auto;
        max-width: 640px;
    }
    """

    with gr.Blocks(css=css) as demo:
        with gr.Column(elem_id="col-container"):
            gr.Markdown(" # Text-to-Image Generation")

            with gr.Row():
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )

                run_button = gr.Button("Run", scale=0, variant="primary")

            result = gr.Image(label="Result", show_label=False)

            with gr.Accordion("Advanced Settings", open=True):
                negative_prompt = gr.Text(
                    label="Negative prompt",
                    max_lines=1,
                    value="lowres, text, error, cropped, worst quality, "
                          "low quality, jpeg artifacts, ugly, "
                          "duplicate, morbid, mutilated, out of frame, "
                          "extra fingers, mutated hands, "
                          "poorly drawn hands, poorly drawn face, "
                          "mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, "
                          "extra limbs, cloned face, disfigured, gross proportions, "
                          "malformed limbs, missing arms, missing legs, extra arms, extra legs, "
                          "fused fingers, too many fingers, long neck, "
                          "username, watermark, signature",
                    visible=True,
                )

                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=-1,
                )

                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

                with gr.Row():
                    width = gr.Slider(
                        label="Width",
                        minimum=512,
                        maximum=1024,
                        step=32,
                        value=1024,  # Replace with defaults that work for your checkpoints
                    )

                    height = gr.Slider(
                        label="Height",
                        minimum=512,
                        maximum=1024,
                        step=32,
                        value=1024,  # Replace with defaults that work for your checkpoints
                    )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=7.5,  # Replace with defaults that work for your checkpoints
                    )

                    num_inference_steps = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=400,
                        step=1,
                        value=200,  # Replace with defaults that work for your checkpoints
                    )

                with gr.Row():
                    model_name = gr.Dropdown(
                        label='Model',
                        choices=[
                            'stabilityai/stable-diffusion-xl-base-1.0',
                            'runwayml/stable-diffusion-v1-5'
                        ],
                    )
                    checkpoint = gr.Dropdown(
                        label='Checkpoint',
                        choices=[
                            'checkpoint-400', 'checkpoint-200',
                            'checkpoint-300', 'checkpoint-1500'
                        ]
                    )

            gr.Examples(examples=examples, inputs=[prompt])
        gr.on(
            triggers=[run_button.click, prompt.submit],
            fn=infer,
            inputs=[
                prompt,
                negative_prompt,
                seed,
                randomize_seed,
                width,
                height,
                guidance_scale,
                num_inference_steps,
                model_name,
                checkpoint
            ],
            outputs=[result, seed],
        )
        return demo


if __name__ == "__main__":
    demo = main()
    demo.launch(share=True)
