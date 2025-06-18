import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("mentalhealth_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Prediksi Risiko Depresi Mahasiswa")

with st.form("mental_health_form"):
    st.subheader("Masukkan Data Responden")

    stress_level = st.slider("Skor Stres (1–5)", 1, 5)
    sleep_quality = st.slider("Kualitas Tidur (1–5)", 1, 5)
    academic_pressure = st.slider("Tekanan Akademik (1–5)", 1, 5)
    social_support = st.slider("Dukungan Sosial (1–5)", 1, 5)
    phone_usage = st.number_input("Jam Penggunaan HP per Hari", min_value=0.0, max_value=24.0)

    submit = st.form_submit_button("Prediksi")

if submit:
    features = np.array([[stress_level, sleep_quality, academic_pressure, social_support, phone_usage]])

    try:
        features_scaled = scaler.transform(features)
    except ValueError as e:
        st.error(f"Terjadi kesalahan saat transformasi fitur: {e}")
        st.stop()

    prediction = model.predict(features_scaled)[0]

    st.subheader("Hasil Prediksi:")
    if prediction == 1:
        st.error("Responden berpotensi mengalami depresi.")
    else:
        st.success("Responden tidak menunjukkan indikasi depresi.")
