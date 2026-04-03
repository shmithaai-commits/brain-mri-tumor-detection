import numpy as np
from scipy import ndimage

class FeatureExtractionAgent:
    """
    Agent 4 — MEASURER
    Calculates tumor size, shape, location and intensity stats
    from the segmentation mask.
    """

    def process(self, classified: dict) -> dict:
        mask = classified.get("mask")

        # Default empty features
        features = {
            "tumor_area_px": 0,
            "tumor_area_pct": 0.0,
            "centroid": None,
            "bounding_box": None,
            "mean_intensity": 0.0,
            "std_intensity": 0.0,
            "circularity": 0.0,
        }

        # If no tumor detected or no mask, return defaults
        if mask is None or classified.get("predicted_class") == 0:
            print(f"  ✅ FeatureExtractionAgent: No tumor — skipping measurements")
            return {**classified, "features": features}

        tumor_px = int(mask.sum())

        # If mask exists but has no positive pixels
        if tumor_px == 0:
            print(f"  ✅ FeatureExtractionAgent: Mask empty — no tumor region found")
            return {**classified, "features": features}

        total_px = mask.shape[0] * mask.shape[1]
        area_pct = round((tumor_px / total_px) * 100, 2)

        # Centroid
        cy, cx = ndimage.center_of_mass(mask)

        # Bounding box — safe version
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]

        if len(row_indices) > 0 and len(col_indices) > 0:
            rmin, rmax = row_indices[[0, -1]]
            cmin, cmax = col_indices[[0, -1]]
            bounding_box = (int(cmin), int(rmin), int(cmax), int(rmax))
        else:
            bounding_box = None

        # Circularity
        eroded = ndimage.binary_erosion(mask)
        perimeter = float(np.sum(mask) - np.sum(eroded))
        circularity = (4 * np.pi * tumor_px / (perimeter ** 2)) if perimeter > 0 else 0.0

        features = {
            "tumor_area_px": tumor_px,
            "tumor_area_pct": area_pct,
            "centroid": (round(float(cx), 1), round(float(cy), 1)),
            "bounding_box": bounding_box,
            "mean_intensity": 0.0,
            "std_intensity": 0.0,
            "circularity": round(circularity, 4),
        }

        # Intensity stats inside tumor region
        tensor = classified.get("tensor")
        if tensor is not None:
            img_np = tensor.permute(1, 2, 0).cpu().numpy()
            gray = img_np.mean(axis=2)
            if gray.shape == mask.shape:
                tumor_vals = gray[mask == 1]
                if len(tumor_vals) > 0:
                    features["mean_intensity"] = round(float(tumor_vals.mean()), 4)
                    features["std_intensity"] = round(float(tumor_vals.std()), 4)

        print(f"  ✅ FeatureExtractionAgent: area={features['tumor_area_pct']}%, "
              f"circularity={features['circularity']}")

        return {**classified, "features": features}
