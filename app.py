import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("mentalhealth_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Prediksi Risiko Depresi Mahasiswa")

with st.form("mental_health_form"):
    st.subheader("Masukkan Data Responden")

    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    age = st.number_input("Usia", min_value=10, max_value=100, step=1)
    cgpa = st.number_input("IPK", min_value=0.0, max_value=4.0, format="%.2f")
    sleep_quality = st.slider("Kualitas Tidur (1–5)", 1, 5)
    academic_pressure = st.slider("Tekanan Akademik (1–5)", 1, 5)
    social_support = st.slider("Dukungan Sosial (1–5)", 1, 5)
    phone_usage = st.number_input("Jam Penggunaan HP per Hari", min_value=0.0, max_value=24.0)

    submit = st.form_submit_button("Prediksi")

if submit:
    # Konversi input ke format numerik dan urutannya sesuai saat training
    gender_num = 1 if gender.lower() == "laki-laki" else 0

    features = np.array([[gender_num, age, cgpa, sleep_quality,
                          academic_pressure, social_support, phone_usage]])

    # Scaling
    try:
        features_scaled = scaler.transform(features)
    except ValueError as e:
        st.error(f"Terjadi kesalahan saat transformasi fitur: {e}")
        st.stop()

    # Prediksi
    prediction = model.predict(features_scaled)[0]

    # Output
    st.subheader("Hasil Prediksi:")
    if prediction == 1:
        st.error("⚠️ Responden berpotensi mengalami depresi.")
    else:
        st.success("✅ Responden tidak menunjukkan indikasi depresi.")
