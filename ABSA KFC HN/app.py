

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="KFC Aspect Sentiment (ABSA)", layout="wide")
st.title("🍗 KFC Aspect Sentiment — Demo")

# 0=None, 1=Positive, 2=Negative, 3=Neutral
DEFAULT_LABEL_MAP = {0: "None", 1: "Positive", 2: "Negative", 3: "Neutral"}

# ========================================================
# 🔥 NEW: Model Switching Options
# ========================================================
MODEL_OPTIONS = {
    "Best Model": "kfc_absa_aspects.pkl",
    "SVM Model": "kfc_absa_aspects_svm.pkl",
    "Logistic Regression": "kfc_absa_aspects_lr.pkl",
    "Naive Bayes": "kfc_absa_aspects_nb.pkl",
}

# ========================================================
# Load model based on selection
# ========================================================
@st.cache_resource
def load_models(model_path: str):
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p.resolve()}")

    obj = joblib.load(p)
    pipelines = obj["pipelines"]
    aspects = obj.get("aspects", list(pipelines.keys()))
    label_map = obj.get("label_map", DEFAULT_LABEL_MAP)

    # Normalize label_map keys to int
    clean_map = {}
    for k, v in label_map.items():
        try:
            clean_map[int(k)] = str(v)
        except:
            pass
    for k, v in DEFAULT_LABEL_MAP.items():
        clean_map.setdefault(k, v)

    return pipelines, aspects, clean_map

# ========================================================
# 🔥 NEW: Choose model in the sidebar
# ========================================================
st.sidebar.header("⚙️ Choose Model")
selected_model_name = st.sidebar.selectbox("Model:", list(MODEL_OPTIONS.keys()))
selected_model_file = MODEL_OPTIONS[selected_model_name]

st.sidebar.write("📦 Using model file:")
st.sidebar.code(selected_model_file)

pipelines, aspects, label_map = load_models(selected_model_file)

# ======================
# Sidebar label explanation
# ======================
st.sidebar.header("Label explanation")
st.sidebar.write("0 = None (not mentioned), 1 = Positive, 2 = Negative, 3 = Neutral")

# ======================
# UI tabs
# ======================
tab1, tab2 = st.tabs(["🔍 Analyze Text", "ℹ️ Guide"])

with tab1:
    st.subheader("Comment")
    txt = st.text_area(
        "Enter a review (English).",
        height=150
    )

    if st.button("Analyze", use_container_width=True):
        if not txt.strip():
            st.warning("Please enter some content.")
        else:
            rows = []
            pos = neg = neu = 0

            for asp in aspects:
                pipe = pipelines.get(asp)
                if pipe is None:
                    continue

                pred_id = int(pipe.predict([txt])[0])
                pred_name = label_map.get(pred_id, "None")

                # 👉 Same as the previous version: auto-extract aspects
                # Only SHOW aspects whose sentiment is not "None"
                if pred_name != "None":
                    pretty_asp = asp.replace("_", " ").title()
                    rows.append((pretty_asp, pred_name))

                if pred_id == 1:
                    pos += 1
                elif pred_id == 2:
                    neg += 1
                elif pred_id == 3:
                    neu += 1

            if not rows:
                st.info("The model did not find any aspect with sentiment (all are 'None').")
            else:
                st.write("**Aspect results (auto-extracted):**")
                st.table(pd.DataFrame(rows, columns=["Aspect", "Sentiment"]))

            # Majority-vote overall (ignore None)
            total_votes = pos + neg + neu
            if total_votes == 0:
                overall = "None"
            else:
                if pos > neg:
                    overall = "Positive"
                elif neg > pos:
                    overall = "Negative"
                else:
                    overall = "Neutral"

            st.success(f"**Total / Overall:** {overall}")

with tab2:
    st.subheader("How to use")
    st.markdown(
        """
- Step 1: Run `kfc_preprocess_and_train.py` to generate:
    - `kfc_clean.csv`
    - `kfc_absa_aspects.pkl`
    - `kfc_absa_aspects_svm.pkl`
    - `kfc_absa_aspects_lr.pkl`
    - `kfc_absa_aspects_nb.pkl`
- Step 2: Run:
    ```bash
    streamlit run app.py
    ```
- Tab **Analyze Text**:
    - Paste one review into the text box.
    - The app will **run through each aspect** and only display aspects whose sentiment is not "None".
- You can **choose models** in the sidebar to compare results.
        """
    )

                
