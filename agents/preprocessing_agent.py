import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class PreprocessingAgent:
    """
    Agent 1 — CLEANER
    Cleans, resizes, normalizes brain MRI images
    before passing to the next agent.
    """

    def __init__(self, img_size=256, augment=False):
        self.img_size = img_size
        self.augment = augment
        self.transform = self._build_transform()

    def _build_transform(self):
        steps = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]
        return transforms.Compose(steps)

    def process(self, image_path: str) -> dict:
        """
        Takes an image file path.
        Returns cleaned image tensor + metadata.
        """
        img = Image.open(image_path).convert("RGB")
        original_shape = (img.height, img.width, 3)

        tensor = self.transform(img)

        print(f"  ✅ PreprocessingAgent: {image_path} → shape {tensor.shape}")

        return {
            "tensor": tensor,
            "original_shape": original_shape,
            "image_path": image_path,
        }
