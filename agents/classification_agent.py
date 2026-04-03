import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.classifier import TumorClassifier

class ClassificationAgent:
    """
    Agent 3 — CLASSIFIER
    Takes the image and decides:
      0 = No Tumor
      1 = Tumor Present
    """

    LABELS = {0: "No Tumor", 1: "Tumor Detected"}

    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TumorClassifier(num_classes=2).to(self.device)

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"  ✅ ClassificationAgent: loaded weights from {model_path}")
        else:
            print("  ⚠️  ClassificationAgent: no weights loaded (run train.py first)")

        self.model.eval()

    def process(self, segmented: dict) -> dict:
        """
        Input:  output from SegmentationAgent
        Output: class label + confidence score
        """
        tensor = segmented["tensor"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0][pred_class].item()) * 100

        label = self.LABELS[pred_class]
        print(f"  ✅ ClassificationAgent: {label} ({confidence:.1f}% confidence)")

        return {
            **segmented,
            "predicted_class": pred_class,
            "label": label,
            "confidence": round(confidence, 2),
        }
