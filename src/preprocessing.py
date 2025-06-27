from PIL import Image, ImageOps
import os


def resize_images(input_dir: str, size: tuple = (1024, 1024)) -> None:
    """
    Resize all images in a directory to a specified size.
    
    This function processes all JPG, JPEG, and PNG images in the input directory.
    Images are cropped to square format and resized to the specified dimensions.
    
    Parameters
    ----------
    input_dir : str
        Path to the directory containing images to be processed
    size : tuple, optional
        Target size for the images as (width, height), default is (1024, 1024)
        
    Returns
    -------
    None
        Images are modified in-place and saved back to the original directory
        
    Notes
    -----
    - Images are converted to RGB format
    - EXIF orientation is corrected
    - Images are cropped to square format from the center
    - Original files are overwritten with the processed versions
    """
    for file in os.listdir(input_dir):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            img = Image.open(os.path.join(input_dir, file)).convert('RGB')
            img = ImageOps.exif_transpose(img)
            img = img.crop((0, img.height - img.width, img.width, img.height))
            img = img.resize(size).convert('RGB')
            img.save(os.path.join(input_dir, file))
