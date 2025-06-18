import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("mentalhealth_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Prediksi Kesehatan Mental Mahasiswa")

with st.form("mental_health_form"):
    st.subheader("Masukkan data responden:")

    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    age = st.number_input("Usia", min_value=10, max_value=100)
    cgpa = st.number_input("IPK", min_value=0.0, max_value=4.0, format="%.2f")
    sleep_quality = st.slider("Kualitas Tidur (1: buruk - 5: sangat baik)", 1, 5)
    academic_pressure = st.slider("Tekanan Akademik (1: rendah - 5: tinggi)", 1, 5)
    social_support = st.slider("Dukungan Sosial (1: rendah - 5: tinggi)", 1, 5)
    phone_usage = st.number_input("Jam Penggunaan HP per Hari", min_value=0.0, max_value=24.0)

    submit = st.form_submit_button("Prediksi")

if submit:
    gender_num = 1 if gender == "Laki-laki" else 0

    # Susun fitur sesuai urutan training
    features = np.array([[gender_num, age, cgpa, sleep_quality,
                          academic_pressure, social_support, phone_usage]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    st.subheader("Hasil Prediksi:")
    if prediction == 1:
        st.error("Responden berpotensi mengalami depresi.")
    else:
        st.success("Responden tidak menunjukkan indikasi depresi.")
