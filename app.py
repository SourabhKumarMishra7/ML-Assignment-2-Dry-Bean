
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Dry Bean Classification", layout="wide")

st.title("Dry Bean Classification System")
st.write("Evaluate five trained machine learning models.")

# Load preprocessing objects
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# Available models
model_options = {
    "Logistic Regression": ("model/logistic_regression.pkl", True),
    "Decision Tree": ("model/decision_tree.pkl", False),
    "KNN": ("model/knn.pkl", True),
    "Gaussian Naive Bayes": ("model/naive_bayes.pkl", False),
    "SVM": ("model/svm.pkl", True)
}

selected_model = st.selectbox(
    "Select Machine Learning Model",
    list(model_options.keys())
)

uploaded_file = st.file_uploader(
    "Upload labelled test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(data.head())

    if "Class" not in data.columns:
        st.error("CSV must contain a target column named Class.")

    else:
        X = data.drop(columns=["Class"])
        y = data["Class"]

        if set(X.columns) != set(scaler.feature_names_in_):
            st.error("Uploaded CSV must contain the required 16 features.")

        else:
            X = X[scaler.feature_names_in_]

            try:
                y_encoded = label_encoder.transform(y)

                model_path, needs_scaling = model_options[selected_model]
                model = joblib.load(model_path)

                # Scale only models that require scaled features
                X_input = scaler.transform(X) if needs_scaling else X

                predictions = model.predict(X_input)
                probabilities = model.predict_proba(X_input)

                # Calculate evaluation metrics
                accuracy = accuracy_score(y_encoded, predictions)
                auc = roc_auc_score(
                    y_encoded,
                    probabilities,
                    multi_class="ovr",
                    average="weighted"
                )
                precision = precision_score(
                    y_encoded, predictions, average="weighted"
                )
                recall = recall_score(
                    y_encoded, predictions, average="weighted"
                )
                f1 = f1_score(
                    y_encoded, predictions, average="weighted"
                )
                mcc = matthews_corrcoef(y_encoded, predictions)

                st.subheader("Evaluation Results")

                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", round(accuracy, 4))
                col2.metric("AUC", round(auc, 4))
                col3.metric("Precision", round(precision, 4))

                col4, col5, col6 = st.columns(3)
                col4.metric("Recall", round(recall, 4))
                col5.metric("F1 Score", round(f1, 4))
                col6.metric("MCC", round(mcc, 4))

                st.subheader("Confusion Matrix")

                cm = confusion_matrix(y_encoded, predictions)
                st.dataframe(pd.DataFrame(
                    cm,
                    index=label_encoder.classes_,
                    columns=label_encoder.classes_
                ))

                st.subheader("Classification Report")

                report = classification_report(
                    y_encoded,
                    predictions,
                    target_names=label_encoder.classes_
                )

                st.text(report)

            except ValueError as error:
                st.error(f"Please check your uploaded CSV: {error}")
