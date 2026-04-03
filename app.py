import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from PIL import Image
from pipeline import MedicalImagePipeline

st.set_page_config(
    page_title="Brain MRI Analyzer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain MRI Tumor Detection System")
st.markdown("*AI-powered multi-agent medical image analysis*")
st.divider()

@st.cache_resource
def load_pipeline():
    return MedicalImagePipeline()

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This system uses **5 AI Agents**:
    1. 🧹 **Cleaner** — Preprocesses image
    2. 🔍 **Finder** — Segments tumor region
    3. 🏷️ **Classifier** — Detects tumor presence
    4. 📏 **Measurer** — Extracts features
    5. 📝 **Writer** — Generates report
    """)
    st.divider()
    st.warning("⚠️ For educational purposes only. Not a substitute for professional medical diagnosis.")

# ── Main UI ──────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Input Image")
        img = Image.open(uploaded)
        st.image(img, use_column_width=True)

    if st.button("🔬 Analyze Image", type="primary"):
        with st.spinner("Running 5-agent pipeline..."):
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                img.save(tmp.name)
                tmp_path = tmp.name

            try:
                pipeline = load_pipeline()
                result = pipeline.run(tmp_path)
            finally:
                os.unlink(tmp_path)

        with col2:
            st.subheader("🔍 Segmentation Mask")
            mask = result.get("mask")
            if mask is not None:
                mask_img = (mask * 255).astype(np.uint8)
                st.image(mask_img, use_column_width=True, clamp=True)
            else:
                st.info("No mask generated")

        st.divider()

        # Results row
        c1, c2, c3, c4 = st.columns(4)
        features = result.get("features", {})

        c1.metric("🏷️ Diagnosis",    result.get("label", "N/A"))
        c2.metric("🎯 Confidence",   f"{result.get('confidence', 0)}%")
        c3.metric("📐 Tumor Area",   f"{features.get('tumor_area_pct', 0)}%")
        c4.metric("⭕ Circularity",  features.get('circularity', 0))

        st.divider()
        st.subheader("📄 AI-Generated Medical Report")
        st.markdown(result.get("report", "No report generated."))

        # Download button
        st.download_button(
            label="⬇️ Download Report",
            data=result.get("report", ""),
            file_name="brain_mri_report.txt",
            mime="text/plain"
        )
