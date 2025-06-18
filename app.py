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

    # Tambahan fitur sintetis yang kamu buat di .ipynb
    stress_level = st.slider("Skor Stres (1–5)", 1, 5)
    sleep_hours = st.number_input("Jam Tidur", min_value=0.0, max_value=24.0)
    anxiety_score = st.slider("Skor Kecemasan (1–10)", 1, 10)
    motivation = st.slider("Motivasi Akademik (1–5)", 1, 5)
    family_issues = st.slider("Permasalahan Keluarga (1–5)", 1, 5)
    financial_stress = st.slider("Tekanan Finansial (1–5)", 1, 5)
    campus_engagement = st.slider("Keterlibatan di Kampus (1–5)", 1, 5)
    workload = st.slider("Beban Tugas (1–5)", 1, 5)

    submit = st.form_submit_button("Prediksi")

if submit:
    gender_num = 1 if gender.lower() == "laki-laki" else 0

    features = np.array([[gender_num, age, cgpa, sleep_quality, academic_pressure,
                          social_support, phone_usage, stress_level, sleep_hours,
                          anxiety_score, motivation, family_issues, financial_stress,
                          campus_engagement, workload]])

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
