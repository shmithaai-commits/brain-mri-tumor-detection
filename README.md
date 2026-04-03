# Medical Image Processing — Brain MRI Tumor Detection

A multi-agent AI pipeline for automated brain tumor detection and report generation.

## Project Structure
```
medical-image-processing/
├── agents/
│   ├── preprocessing_agent.py      ← Agent 1: Clean image
│   ├── segmentation_agent.py       ← Agent 2: Find tumor (U-Net)
│   ├── classification_agent.py     ← Agent 3: Classify (ResNet18)
│   ├── feature_extraction_agent.py ← Agent 4: Measure tumor
│   └── report_agent.py             ← Agent 5: Write report (Claude AI)
├── models/
│   ├── unet.py                     ← U-Net architecture
│   └── classifier.py               ← ResNet18 classifier
├── data/                           ← Dataset goes here
├── pipeline.py                     ← Connects all 5 agents
├── train.py                        ← Train the models
├── app.py                          ← Streamlit UI
├── download_dataset.py             ← Download from Kaggle
└── requirements.txt
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set environment variable
```bash
# Windows
set ANTHROPIC_API_KEY=your_key_here

# Mac/Linux
export ANTHROPIC_API_KEY=your_key_here
```

### 3. Download dataset
```bash
python download_dataset.py
```

### 4. Train models
```bash
python train.py
```

### 5. Run the app
```bash
streamlit run app.py
```

### 6. Test pipeline directly
```bash
python pipeline.py data/yes/image1.jpg
```

## Validation Targets
| Metric | Target |
|---|---|
| Classifier Accuracy | > 90% |
| Classifier Recall | > 95% |
| Segmentation IoU | > 75% |
| Report generation | < 10 sec |

## Dataset
- **Source:** Kaggle — navoneel/brain-mri-images-for-brain-tumor-detection
- **Classes:** yes (tumor) / no (no tumor)
- **Size:** ~253 images
