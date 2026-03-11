import torch
import logging
import sys
import random
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



class RandomHueShift:
    """
    Hue jitter implemented with Pillow only.

    This avoids NumPy's `uint8` casting behavior changes (NumPy 2.x raises on
    negative -> uint8 casts) that can break torchvision's hue jitter on some
    stacks.
    """

    def __init__(self, max_delta: float):
        # Matches torchvision semantics: hue is in [0, 0.5].
        if not (0.0 <= max_delta <= 0.5):
            raise ValueError("max_delta must be within [0.0, 0.5].")
        self.max_delta = float(max_delta)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.max_delta == 0.0:
            return img
        if not isinstance(img, Image.Image):
            return img
        if img.mode not in ("RGB", "RGBA"):
            return img

        # Shift hue channel in HSV space with modular arithmetic.
        hsv = img.convert("HSV")
        h, s, v = hsv.split()
        delta = int(random.uniform(-self.max_delta, self.max_delta) * 255)
        if delta != 0:
            h = h.point(lambda x: (x + delta) % 256)
        rgb = Image.merge("HSV", (h, s, v)).convert("RGB")

        if img.mode == "RGBA":
            # Preserve alpha if it exists (not expected with our loader, but safe).
            alpha = img.split()[3]
            rgb.putalpha(alpha)
            return rgb

        return rgb

def get_data_transforms() -> dict:
    """
    Returns a dictionary of data transformations for training and validation.
    """
    return {
        "train": transforms.Compose(
            [
                # Use a crop that preserves more of the image center
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
                # ColorJitter is a good augmentation that doesn't affect orientation.
                # We implement hue jitter separately to avoid NumPy 2.x uint8 casting issues.
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0
                ),
                RandomHueShift(max_delta=0.1),
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
    Ultra-robust image loader that handles all image modes and potential errors.
    Falls back to creating a blank image if all else fails.
    """
    try:
        # 1. Open the image
        img = Image.open(path)
        
        # 2. Respect EXIF orientation
        img = ImageOps.exif_transpose(img)
        
        # 3. Handle different modes safely
        if img.mode == 'RGB':
            return img
        
        elif img.mode == 'RGBA':
            # Create white background and composite
            background = Image.new('RGB', img.size, (255, 255, 255))
            # Split to get alpha channel safely
            if len(img.split()) == 4:
                alpha = img.split()[3]
                # Convert to RGB before pasting
                rgb_img = img.convert('RGB')
                background.paste(rgb_img, mask=alpha)
                return background
            else:
                return img.convert('RGB')
        
        elif img.mode == 'LA' or img.mode == 'PA':
            # Convert to RGBA first, then handle
            rgba_img = img.convert('RGBA')
            background = Image.new('RGB', rgba_img.size, (255, 255, 255))
            if len(rgba_img.split()) == 4:
                alpha = rgba_img.split()[3]
                rgb_img = rgba_img.convert('RGB')
                background.paste(rgb_img, mask=alpha)
                return background
            else:
                return rgba_img.convert('RGB')
        
        elif img.mode == 'P':
            # Palletized image - check for transparency
            if img.info.get('transparency', None) is not None:
                # Has transparency
                rgba_img = img.convert('RGBA')
                background = Image.new('RGB', rgba_img.size, (255, 255, 255))
                if len(rgba_img.split()) == 4:
                    alpha = rgba_img.split()[3]
                    rgb_img = rgba_img.convert('RGB')
                    background.paste(rgb_img, mask=alpha)
                    return background
                else:
                    return rgba_img.convert('RGB')
            else:
                # No transparency, just convert
                return img.convert('RGB')
        
        elif img.mode == 'L':
            # Grayscale - direct conversion
            return img.convert('RGB')
        
        elif img.mode == 'CMYK':
            # CMYK - convert directly (loses color profile but works)
            return img.convert('RGB')
        
        elif img.mode == 'YCbCr':
            # YCbCr - convert directly
            return img.convert('RGB')
        
        else:
            # Unknown mode - try direct conversion
            try:
                return img.convert('RGB')
            except:
                # If all else fails, create a blank image
                logging.warning(f"Could not convert {path} to RGB. Creating blank image.")
                return Image.new('RGB', (224, 224), (128, 128, 128))
    
    except Exception as e:
        logging.error(f"Error loading image {path}: {e}")
        # Return a blank image as fallback
        return Image.new('RGB', (224, 224), (128, 128, 128))