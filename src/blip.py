import os
import csv
from PIL import Image, UnidentifiedImageError
from transformers import BlipProcessor, BlipForConditionalGeneration
from config import device, DATA_DIR, CSV_PATH


def get_captions() -> None:
    """
    Generate captions for all images in the data directory using BLIP model.
    
    This function processes all images in the DATA_DIR and generates captions using
    the BLIP image captioning model. The captions are prefixed with "Ioannis is "
    and saved to a CSV file for training.
    
    Parameters
    ----------
    None. Uses global configuration from config module
        
    Returns
    -------
    None
        Saves captions to CSV file specified by CSV_PATH
        
    Notes
    -----
    - Uses BLIP base model for image captioning
    - Skips files that are not valid images
    - Captions are automatically prefixed with "Ioannis is "
    - Results are saved in CSV format with columns: image, text
    - Images are processed in sorted order
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base").to(device)
    results = []

    for img_name in sorted(os.listdir(DATA_DIR)):
        try:
            raw_image = Image.open(f"{DATA_DIR}/{img_name}").convert('RGB')
        except UnidentifiedImageError:
            print(f"Skipping {img_name} because it is not a valid image")
            continue

        inputs = processor(raw_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        out = model.generate(pixel_values=pixel_values)
        unconditional_caption = processor.decode(out[0], skip_special_tokens=True)
        adjusted_caption = "Ioannis is " + unconditional_caption

        results.append([img_name, adjusted_caption])

    with open(CSV_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "text"])
        writer.writerows(results)
