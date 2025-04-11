from diffusers import AutoPipelineForInpainting
import torch
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from utils import crop_to_mask, cosine_similarity_between_features
from dino_metric import get_dino_features
import copy

# Load the inpainting pipeline
pipe = AutoPipelineForInpainting.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

# Load your image and mask
init_image = Image.open("meow.png").convert("RGB").resize((512, 512))

rgba_array = (np.load("2.npy") > 0).astype(np.float32)


if rgba_array.dtype != np.uint8:
    rgba_array = (rgba_array * 255).astype(np.uint8)

# Create a PIL Image from the NumPy array
rgba_image = Image.fromarray(rgba_array, mode="RGBA")

# Convert to grayscale
grayscale_image = rgba_image.convert("L")

# Resize the image
mask_image = grayscale_image.resize((512, 512), resample=Image.NEAREST)
mask_image = mask_image.filter(ImageFilter.MaxFilter(size=11))

mask_image.save("output.png", format="PNG")

inverted_mask = ImageOps.invert(mask_image)
alpha = inverted_mask.point(lambda x: x)  # no change, just ensure correct format
vis_image = copy.deepcopy(init_image)
vis_image.putalpha(alpha)
background = Image.new("RGBA", vis_image.size, (255, 255, 255, 255))
result = Image.alpha_composite(background, vis_image)
result.save("res.png")

# Define your prompt
prompt = "Full HD, 4K, high quality, high resolution, photorealistic"
negative_prompt = "bad anatomy, bad proportions, blurry, cropped, deformed, disfigured, duplicate, error, extra limbs, gross proportions, jpeg artifacts, long neck, low quality, lowres, malformed, morbid, mutated, mutilated, out of frame, ugly, worst quality"

# Perform inpainting
result = pipe(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    num_inference_steps=400,
    negative_prompt=negative_prompt,
).images[0]

# Save the result

crop1 = crop_to_mask(result, mask_image)
crop2 = crop_to_mask(init_image, mask_image)

crop1.save("crop1.png")
crop2.save("crop2.png")
result.save("inpainted_image.png")

crop1_dino = get_dino_features(crop1)
crop2_dino = get_dino_features(crop2)

print(cosine_similarity_between_features(crop1_dino, crop2_dino))