
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib

# Title
st.title("🩺 Diabetes Prediction App (SVM + PCA)")
st.write("Upload your health data to get a diabetes prediction using an SVM model with PCA.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with your data", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.write(data.head())

    # Check if target column exists
    if 'Diabetes_binary' in data.columns:
        X = data.drop('Diabetes_binary', axis=1)
        y = data['Diabetes_binary']
    else:
        X = data

    # Pipeline steps (fit again or load pre-trained)
    scaler = StandardScaler()
    pca = PCA(n_components=0.95)  # retain 95% variance
    svm = SVC(C=1, gamma='scale', kernel='rbf')

    pipeline = Pipeline([
        ('scaler', scaler),
        ('pca', pca),
        ('svm', svm)
    ])

    if 'Diabetes_binary' in data.columns:
        pipeline.fit(X, y)
        st.success("Model trained on uploaded data.")

        # Predictions on training data
        preds = pipeline.predict(X)
        data['Prediction'] = preds
        st.subheader("Predictions")
        st.write(data[['Prediction']].value_counts())
    else:
        # Load model (optional: save and reuse trained model)
        st.warning("No 'Diabetes_binary' column found. Assuming inference mode.")

        try:
            model = joblib.load("svm_pca_model.pkl")
            preds = model.predict(X)
            st.subheader("Predicted Labels")
            st.write(preds)
        except:
            st.error("Pretrained model not found. Please upload labeled data first.")
else:
    st.info("Please upload a CSV file to begin.")



