import torch
import numpy as np
import cv2
from PIL import Image

class SegmentationAgent:
    """
    Agent 2 — FINDER
    Uses image processing (Otsu thresholding + morphology)
    to find WHERE the bright tumor region is.
    No training required — works immediately!
    """

    def __init__(self, model_path: str = None, device: str = None):
        print("  ✅ SegmentationAgent: using smart thresholding (no training needed)")

    def process(self, preprocessed: dict) -> dict:
        image_path = preprocessed["image_path"]

        # Read original image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img_pil = Image.open(image_path).convert("L")
            img = np.array(img_pil)

        # Resize to standard size
        img_resized = cv2.resize(img, (256, 256))

        # ── Step 1: Enhance contrast with CLAHE ───────────────────────────────
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_resized)

        # ── Step 2: Otsu thresholding to find bright regions ──────────────────
        _, binary = cv2.threshold(enhanced, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ── Step 3: Remove skull/edge artifacts ───────────────────────────────
        # Create brain mask (remove outer ring)
        brain_mask = np.zeros_like(binary)
        center = (128, 128)
        cv2.ellipse(brain_mask, center, (100, 115), 0, 0, 360, 255, -1)
        binary = cv2.bitwise_and(binary, brain_mask)

        # ── Step 4: Morphological cleanup ─────────────────────────────────────
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        # ── Step 5: Keep only the largest bright blob ─────────────────────────
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )

        mask = np.zeros((256, 256), dtype=np.float32)

        if num_labels > 1:
            # Find largest component (excluding background label 0)
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask[labels == largest] = 1.0

        tumor_pixels = int(mask.sum())
        has_tumor    = tumor_pixels > 200

        print(f"  ✅ SegmentationAgent: tumor_pixels={tumor_pixels}, has_tumor={has_tumor}")

        return {
            **preprocessed,
            "mask": mask,
            "tumor_pixels": tumor_pixels,
            "has_tumor": has_tumor,
        }
