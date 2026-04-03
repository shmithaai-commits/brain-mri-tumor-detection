import os
import zipfile
import urllib.request

# ── Kaggle credentials ─────────────────────────────────────────────────────────
os.environ["KAGGLE_USERNAME"] = "shmitha"
os.environ["KAGGLE_KEY"]      = "KGAT_8fa3448db07daf1168d087efe8b63257"

DOWNLOAD_PATH = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DOWNLOAD_PATH, exist_ok=True)

# ── Download using kaggle API ──────────────────────────────────────────────────
try:
    import kaggle
    print("📥 Downloading Brain MRI dataset from Kaggle...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "navoneel/brain-mri-images-for-brain-tumor-detection",
        path=DOWNLOAD_PATH,
        unzip=True
    )
    print(f"✅ Dataset downloaded to: {DOWNLOAD_PATH}")

except Exception as e:
    print(f"❌ Kaggle API failed: {e}")
    print("💡 Trying alternative method...")

    # Alternative: direct HTTP download
    import requests, zipfile, io

    url = "https://storage.googleapis.com/kagglesdk/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/brain-mri-images-for-brain-tumor-detection.zip"
    print(f"📥 Downloading from: {url}")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(DOWNLOAD_PATH)
        print(f"✅ Downloaded and extracted to: {DOWNLOAD_PATH}")
    else:
        print("❌ Download failed. Please download manually from:")
        print("   https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection")
        print(f"   Then extract the zip into: {DOWNLOAD_PATH}")

# ── Show summary ───────────────────────────────────────────────────────────────
print("\n📊 Dataset Summary:")
total = 0
for root, dirs, files in os.walk(DOWNLOAD_PATH):
    imgs = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if imgs:
        print(f"  📁 {os.path.basename(root)}: {len(imgs)} images")
        total += len(imgs)
if total > 0:
    print(f"\n  🧠 Total images found: {total}")
    print("  ✅ Dataset is ready!")
else:
    print("  ⚠️  No images found yet.")
