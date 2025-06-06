from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, gaussian_blur
from diffusers.utils import load_image

def crop_to_mask(original: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Crops the original image to the tightest bounding box around the non-zero area of the mask.
    
    Args:
        original (PIL.Image.Image): The original image.
        mask (PIL.Image.Image): The segmentation mask (assumed to be grayscale or binary).
        
    Returns:
        PIL.Image.Image: Cropped image based on the mask.
    """
    mask = mask.convert('L')  # Ensure grayscale
    np_mask = np.array(mask)

    # Get non-zero indices
    nonzero = np.argwhere(np_mask > 0)

    if nonzero.size == 0:
        # No segmentation present — return empty crop (or handle as needed)
        return original.crop((0, 0, 1, 1))

    # Get bounding box (min_row, min_col, max_row, max_col)
    top_left = nonzero.min(axis=0)
    bottom_right = nonzero.max(axis=0)

    # Format as (left, upper, right, lower) for Pillow crop
    bbox = (top_left[1], top_left[0], bottom_right[1]+1, bottom_right[0]+1)
    return original.crop(bbox)

def cosine_similarity_between_features(feat1: torch.Tensor, feat2: torch.Tensor) -> float:
    """
    Computes the cosine similarity between two feature tensors.

    Args:
        feat1 (torch.Tensor): First feature tensor of shape (1, feature_dim).
        feat2 (torch.Tensor): Second feature tensor of shape (1, feature_dim).

    Returns:
        float: Cosine similarity between the two feature vectors.
    """
    # Ensure the tensors are of the same shape
    if feat1.shape != feat2.shape:
        raise ValueError("Feature tensors must have the same shape.")

    # Compute cosine similarity
    similarity = F.cosine_similarity(feat1, feat2, dim=1)

    # Return the scalar value
    return similarity.item()

def preprocess_image(pil_image, device):
    image = to_tensor(load_image(pil_image))
    image = image.unsqueeze_(0).float() * 2 - 1 # [0,1] --> [-1,1]
    image = image.expand(-1, 3, -1, -1)
    image = F.interpolate(image, (1024, 1024))
    image = image.to(torch.float16).to(device)
    return image

def preprocess_mask(mask_path, device):
    mask = to_tensor((load_image(mask_path, convert_method=lambda img: img.convert('L'))))
    mask = mask.unsqueeze_(0).float()  # 0 or 1
    mask = F.interpolate(mask, (1024, 1024))
    mask = gaussian_blur(mask, kernel_size=(77, 77))
    mask[mask < 0.1] = 0
    mask[mask >= 0.1] = 1
    mask = mask.to(torch.float16).to(device)
    return mask