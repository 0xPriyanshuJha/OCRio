from transformers import AutoModel, AutoProcessor
from PIL import Image
import torch

model = AutoModel.from_pretrained("OpenGVLab/InternVL-Chat-V1-5", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL-Chat-V1-5")

def extract_text_with_internvl(image: Image.Image) -> str:
    """
    Extracts text from an image using the InternVL model.

    Args:
    - image (PIL.Image): The image to process.

    Returns:
    - str: The extracted text.
    """
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    extracted_text = outputs.logits.argmax(dim=-1).cpu().numpy() 
    return extracted_text

def process_image(image: Image.Image) -> str:
    """
    Processes the image and extracts text using InternVL.

    Args:
    - image (PIL.Image): The image to process.

    Returns:
    - str: The extracted text.
    """
    return extract_text_with_internvl(image)
