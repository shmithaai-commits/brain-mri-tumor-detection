import os
from agents.preprocessing_agent   import PreprocessingAgent
from agents.segmentation_agent    import SegmentationAgent
from agents.classification_agent  import ClassificationAgent
from agents.feature_extraction_agent import FeatureExtractionAgent
from agents.report_agent          import ReportAgent

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

class MedicalImagePipeline:
    """
    Master pipeline that connects all 5 agents in sequence.

    Flow:
      Image → Preprocess → Segment → Classify → Extract → Report
    """

    def __init__(self):
        print("🚀 Initializing Medical Image Pipeline...")
        self.agent1 = PreprocessingAgent(img_size=256, augment=False)
        self.agent2 = SegmentationAgent(
            model_path=os.path.join(MODELS_DIR, "unet.pth")
        )
        self.agent3 = ClassificationAgent(
            model_path=os.path.join(MODELS_DIR, "classifier.pth")
        )
        self.agent4 = FeatureExtractionAgent()
        self.agent5 = ReportAgent()
        print("✅ All agents ready.\n")

    def run(self, image_path: str) -> dict:
        print(f"🧠 Analyzing: {image_path}")
        print("-" * 50)

        result = self.agent1.process(image_path)   # Clean
        result = self.agent2.process(result)        # Find tumor
        result = self.agent3.process(result)        # Classify
        result = self.agent4.process(result)        # Measure
        result = self.agent5.process(result)        # Write report

        print("-" * 50)
        print("✅ Pipeline complete!\n")
        return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <path_to_image>")
        sys.exit(1)

    pipeline = MedicalImagePipeline()
    output = pipeline.run(sys.argv[1])

    print("\n📄 MEDICAL REPORT:")
    print("=" * 60)
    print(output["report"])
    print("=" * 60)
