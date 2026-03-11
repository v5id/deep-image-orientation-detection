import torch
import logging
import sys
import torchvision.transforms as transforms
from config import IMAGE_SIZE
from PIL import Image, ImageOps


def setup_logging():
    """Configures the logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_device() -> torch.device:
    """
    Selects the best available device (CUDA, MPS, or CPU) and returns it.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("CUDA is available. Using GPU.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("MPS is available. Using Apple Silicon GPU.")
    else:
        device = torch.device("cpu")
        logging.info("CUDA and MPS not available. Using CPU.")
    return device


def get_data_transforms() -> dict:
    """
    Returns a dictionary of data transformations for training and validation.
    """
    return {
        "train": transforms.Compose(
            [
                # Use a crop that preserves more of the image center
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
                # ColorJitter is a good augmentation that doesn't affect orientation
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                # RandomErasing is also a good regularizer
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
            ]
        ),
        "val": transforms.Compose(
            [
                # Validation transform is fine as is
                transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    }


def load_image_safely(path: str) -> Image.Image:
    """
    Loads an image, respects EXIF orientation, and safely converts it to a
    3-channel RGB format. It handles palletized images and images with
    transparency by compositing them onto a white background. This is the
    most robust way to prevent processing errors.
    """
    # 1. Open the image
    img = Image.open(path)

    # 2. Respect the EXIF orientation tag before any other processing.
    img = ImageOps.exif_transpose(img)

    # 3. Handle different image modes appropriately
    if img.mode == "RGB":
        # Already RGB, just return
        return img
    
    elif img.mode == "L":
        # Grayscale - convert directly to RGB
        return img.convert("RGB")
    
    elif img.mode == "RGBA":
        # RGBA - composite onto white background using alpha channel as mask
        background = Image.new("RGB", img.size, (255, 255, 255))
        # Split to get alpha channel, use it as mask
        alpha = img.split()[3]
        background.paste(img.convert("RGB"), mask=alpha)
        return background
    
    elif img.mode == "LA":
        # Grayscale with alpha - similar to RGBA but with 2 channels
        background = Image.new("RGB", img.size, (255, 255, 255))
        # Convert to RGBA first to standardize
        rgba_img = img.convert("RGBA")
        alpha = rgba_img.split()[3]
        background.paste(rgba_img.convert("RGB"), mask=alpha)
        return background
    
    elif img.mode == "P":
        # Palletized - could have transparency
        # Check if the palette has transparency info
        if "transparency" in img.info:
            # Convert to RGBA to preserve transparency
            rgba_img = img.convert("RGBA")
            background = Image.new("RGB", rgba_img.size, (255, 255, 255))
            alpha = rgba_img.split()[3]
            background.paste(rgba_img.convert("RGB"), mask=alpha)
            return background
        else:
            # No transparency, just convert to RGB
            return img.convert("RGB")
    
    else:
        # For any other mode (CMYK, YCbCr, etc.), just convert to RGB
        # This might lose transparency, but these modes rarely have it
        return img.convert("RGB")