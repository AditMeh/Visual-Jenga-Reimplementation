import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

# Load the processor and model
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")
model.eval()  # Set the model to evaluation mode


def get_dino_features(pil_img: Image.Image) -> torch.Tensor:
    """
    Extracts DINOv2 features from a PIL image of any size.

    Args:
        pil_img (PIL.Image.Image): The input image.

    Returns:
        torch.Tensor: The DINO feature tensor of shape (1, feature_dim).
    """
    # Preprocess the image
    inputs = processor(images=pil_img, return_tensors="pt")

    # Extract features
    with torch.no_grad():
        outputs = model(**inputs)

    # Use the [CLS] token representation as the image feature
    features = outputs.last_hidden_state[:, 0, :]  # Shape: (1, feature_dim)
    return features
