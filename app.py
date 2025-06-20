import streamlit as st
import pandas as pd
import joblib
from utils.feature_extraction import extract_features

st.title("🛡️ Web Attack Detection Dashboard (SQLi, XSS, CMDi)")

model = joblib.load("model/rf_model.pkl")

attack_labels = {
    0: "🟢 Benign",
    1: "🔴 SQL Injection",
    2: "🟠 Cross-site Scripting (XSS)",
    3: "🔵 Command Injection"
}

option = st.radio("Choose Mode:", ("Upload URLs", "Enter Single URL"))

if option == "Upload URLs":
    uploaded_file = st.file_uploader("Upload CSV with column 'url'", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df["features"] = df["url"].apply(extract_features)
        df["prediction"] = df["features"].apply(lambda x: model.predict([x])[0])
        df["result"] = df["prediction"].map(attack_labels)
        st.dataframe(df[["url", "result"]])
else:
    url = st.text_input("Enter a URL:")
    if url:
        features = extract_features(url)
        pred = model.predict([features])[0]
        result = attack_labels.get(pred, "Unknown")
        st.markdown(f"### Prediction: {result}")
