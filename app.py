import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Model Evaluator", layout="wide")

st.title("📊 Multi-Model Classification Dashboard")
st.markdown("Upload test data to evaluate pre-trained ML models on performance metrics.")

# Sidebar Controls
st.sidebar.header("User Settings")

uploaded_file = st.sidebar.file_uploader("Upload Test CSV (contains target column)", type=["csv"])

model_choice = st.sidebar.selectbox(
    "Choose Classification Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Support Vector Machine"
    ]
)

@st.cache_resource
def load_model_and_scaler(model_name):
    file_key = model_name.lower().replace(" ", "_") + ".pkl"
    model = joblib.load(f"model/{file_key}")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📋 Dataset Preview")
    st.dataframe(data.head(5))

    if 'target' not in data.columns:
        st.error("Uploaded CSV must contain a 'target' column!")
    else:
        X_test = data.drop('target', axis=1)
        y_test = data['target']

        model, scaler = load_model_and_scaler(model_choice)

        if model_choice in ["Logistic Regression", "KNN", "Support Vector Machine"]:
            X_eval = scaler.transform(X_test)
        else:
            X_eval = X_test

        preds = model.predict(X_eval)
        probs = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else preds

        # Calculate metrics
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        mcc = matthews_corrcoef(y_test, preds)

        st.subheader(f"📈 Evaluation Metrics for {model_choice}")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        st.markdown("---")
        
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("🧩 Confusion Matrix")
            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        with c2:
            st.subheader("📝 Classification Report")
            report = classification_report(y_test, preds, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

else:
    st.info("👈 Please upload `test_data.csv` from the sidebar to display evaluations.")