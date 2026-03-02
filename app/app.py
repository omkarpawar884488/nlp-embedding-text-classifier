# app.py
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import streamlit as st

import sys, types
from sklearn.base import BaseEstimator, TransformerMixin

# add near the top with other imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# --- helpers to patch ColumnSelector(s) inside a loaded pipeline ---
def _patch_selector(obj, colname, found):
    # If it's our ColumnSelector
    if isinstance(obj, ColumnSelector):
        if getattr(obj, "column", None) is None and getattr(obj, "column_name", None) is None:
            obj.column = colname
            obj.column_name = colname
        # record what it ended up using
        found.append(getattr(obj, "column", None) or getattr(obj, "column_name", None))
        return

    # Recurse into Pipelines
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            _patch_selector(step, colname, found)
        return

    # Recurse into ColumnTransformers
    if isinstance(obj, ColumnTransformer):
        for _, trans, _ in obj.transformers:
            _patch_selector(trans, colname, found)
        return

    # Recurse into nested estimators with common attrs
    for attr in ("base_estimator", "estimator", "classifier", "regressor"):
        if hasattr(obj, attr):
            _patch_selector(getattr(obj, attr), colname, found)

def patch_column_selectors(model, colname="text"):
    """Set missing column names on any ColumnSelector within the model."""
    found = []
    _patch_selector(model, colname, found)
    return found


# ✅ Single, correct ColumnSelector definition
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column=None, column_name=None):
        # support both names, whichever was used in training
        self.column = column or column_name
        self.column_name = column_name or column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        col = getattr(self, "column", None) or getattr(self, "column_name", None)
        if col is None:
            raise ValueError("No column name found in ColumnSelector.")
        # Support both pandas DataFrame and dict-like inputs
        if hasattr(X, "loc"):
            return X.loc[:, col]
        return X[col]

# 👇 Make pickle think this class lives in "main"
ColumnSelector.__module__ = "main"
sys.modules.setdefault("main", types.ModuleType("main"))
setattr(sys.modules["main"], "ColumnSelector", ColumnSelector)


# ---------- CONFIG ----------
# Adjust ONLY if your pipeline expects a different feature column name
TEXT_COL_NAME = "text"




# Default model path (your path)
DEFAULT_MODEL_PATH = Path("E:/CV/Internship/Coding_Challenge_Omkar_Pawar/models/prod_model.joblib")

# Human-readable names for your numeric classes
CLASS_NAME = {
    0: "Irrelevant",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
}

# ---------- UI ----------
st.set_page_config(page_title="Sentiment Classifier", page_icon="🧠", layout="centered")
st.title("🧠 Sentiment Classifier")
st.caption("Type a comment and classify it into: 0=Irrelevant, 1=Negative, 2=Neutral, 3=Positive")

with st.sidebar:
    st.header("Settings")
    model_path_str = st.text_input(
        "Model path",
        value=str(DEFAULT_MODEL_PATH),
        help="Path to the saved scikit-learn pipeline (.joblib)"
    )
    show_probs_as = st.selectbox("Show probabilities as", ["Table", "Bar chart"], index=0)

@st.cache_resource(show_spinner=True)
def load_model(model_path: str, text_col_name: str):
    model = joblib.load(model_path)

    # 🔧 patch any ColumnSelector missing its column name
    used = patch_column_selectors(model, text_col_name)

    # classes_: numpy array of labels (e.g., [0,1,2,3])
    classes = getattr(model, "classes_", None)
    if classes is None:
        _ = model.predict(pd.DataFrame({text_col_name: ["warmup"]}))
        classes = getattr(model, "classes_", None)
    return model, np.array(classes), used


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    x -= x.max()
    e = np.exp(x)
    return e / e.sum()

# Load model (cached)
try:
    model, classes_ = None, None
    model, classes_, patched = load_model(model_path_str, TEXT_COL_NAME)
    classes_list = list(classes_)
except Exception as e:
    st.error(f"Failed to load model from:\n{model_path_str}\n\n{e}")
    st.stop()
#with st.expander("Debug"):
    #st.write("Patched ColumnSelector(s) to use column:", TEXT_COL_NAME)
    #st.write("Selectors reported using:", patched)


# Input
user_text = st.text_area("Enter comment", height=160, placeholder="Type or paste a comment…")
go = st.button("Classify")

# Inference
if go:
    if not user_text or not user_text.strip():
        st.warning("Please enter a non-empty comment.")
        st.stop()

    # Build input DataFrame that matches the pipeline schema
    X_new = pd.DataFrame({TEXT_COL_NAME: [user_text]})

    # Predict class
    try:
        y_pred = model.predict(X_new)[0]
    except Exception as e:
        st.error(
            f"Prediction failed. Make sure your pipeline expects a column named '{TEXT_COL_NAME}'.\n\n{e}"
        )
        st.stop()

    # Predict probabilities (or fallback to softmax over decision_function)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)[0]  # shape: (n_classes,)
    else:
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_new)
            if scores.ndim == 1:  # binary case can be shape (1,) or (n_classes,) depending on model
                scores = np.vstack([-scores, scores])
            proba = softmax(scores[0])
        else:
            # No probabilities available
            proba = np.ones(len(classes_list)) / len(classes_list)

    # Build a tidy probability table aligned with model.classes_ order
    prob_df = pd.DataFrame({
        "class_id": classes_list,
        "class_name": [CLASS_NAME.get(int(c), str(c)) for c in classes_list],
        "probability": proba
    }).sort_values("probability", ascending=False, ignore_index=True)

    # Top prediction (name + id)
    pred_name = CLASS_NAME.get(int(y_pred), str(y_pred))
    st.subheader("Prediction")
    st.markdown(f"**{pred_name}** (class id: `{int(y_pred)}`)")

    st.subheader("Class probabilities")
    if show_probs_as == "Table":
        st.dataframe(
            prob_df.style.format({"probability": "{:.4f}"}),
            use_container_width=True
        )
    else:
        chart_df = prob_df.set_index("class_name")["probability"]
        st.bar_chart(chart_df)

    with st.expander("Details"):
        st.write("Raw classes (in model order):", classes_list)
        st.write("Note: probabilities align with the estimator’s internal class order.")

st.markdown("---")
st.caption("Model loaded from: `{}`".format(model_path_str))
