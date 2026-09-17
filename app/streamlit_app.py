"""Streamlit demo app. Run: streamlit run app/streamlit_app.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from src.config import DISCLAIMER, MODELS_DIR
from src.predict import predict_image

st.set_page_config(page_title="Brain MRI Classifier (Educational)", page_icon="🧠")

st.warning(f"**{DISCLAIMER}**", icon="⚠️")

st.title("Brain Tumor MRI Classification")
st.sidebar.header("Settings")
st.sidebar.markdown(f"> :warning: {DISCLAIMER}")

available = []
if os.path.isdir(MODELS_DIR):
    available = sorted(f[:-6] for f in os.listdir(MODELS_DIR) if f.endswith(".keras"))

if not available:
    st.info("No trained models found in `models/`. Train one first: "
            "`python src/train.py --model efficientnet`")
    st.stop()

model_name = st.sidebar.selectbox("Model", available)

uploaded = st.file_uploader("Upload a brain MRI image",
                            type=["jpg", "jpeg", "png", "bmp"])
if uploaded is not None:
    st.image(uploaded, caption=uploaded.name, width=300)
    if st.button("Predict"):
        tmp_path = os.path.join("/tmp", uploaded.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        try:
            with st.spinner("Predicting..."):
                result = predict_image(tmp_path, model_name)
            st.success(f"**{result['class']}** — confidence "
                       f"{result['confidence']:.1%}")
            st.bar_chart(result["probabilities"])
            st.caption(DISCLAIMER)
        except Exception as e:  # never crash on bad input
            st.error(f"Prediction failed: {e}")
