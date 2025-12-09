from __future__ import annotations

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

MODEL_NAME = os.getenv("MODEL_NAME", "wineclf_rf")
MODEL_STAGE = os.getenv("MODEL_STAGE")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@st.cache_resource
def load_model():
    if MODEL_STAGE:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    else:
        model_uri = f"models:/{MODEL_NAME}/latest"
    return mlflow.sklearn.load_model(model_uri)


model = load_model()

st.title("🍷 Wine Quality Classifier")


st.sidebar.header("Input features")

default_features = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}

st.sidebar.write("Add meg a feature-öket (csúszkákkal / number inputtal).")

features: dict[str, float] = {}
for name, default_value in default_features.items():
    features[name] = st.sidebar.number_input(name, value=float(default_value))

if st.button("Predict quality (good / not good)"):
    df = pd.DataFrame([features])
    pred = int(model.predict(df)[0])

    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(df)[0, 1])

    label = "jó bor" if pred == 1 else "nem jó bor"
    st.subheader("Predikció")
    st.write("Input feature-k:", features)
    st.success(f"Előrejelzés: **{label}** (osztály: {pred})")
    if proba is not None:
        st.write(f"Valószínűség a 'jó bor' osztályra: `{proba:.3f}`")

st.markdown("---")
st.subheader("Evidently drift report")

report_path = Path("artifacts/evidently/evidently_drift_report.html")
if report_path.exists():
    html = report_path.read_text(encoding="utf-8")
    components.html(html, height=800, scrolling=True)
else:
    st.info("Még nincs Evidently report. Futtasd a `src/wineclf/drift_report.py` szkriptet.")
