import streamlit as st
import numpy as np
import joblib

# Load model yang sudah dilatih dari notebook
model = joblib.load("mentalhealth_model.pkl")  # pastikan file ini hasil dump dari notebook

st.title("Prediksi Gangguan Kesehatan Mental")

with st.form("mental_health_form"):
    st.header("Masukkan data responden:")

    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    age = st.number_input("Usia", min_value=10, max_value=100, step=1)
    cgpa = st.number_input("IPK", min_value=0.0, max_value=4.0, format="%.2f")
    sleepinghours = st.number_input("Jam Tidur per Hari", min_value=0, max_value=24, step=1)
    stress = st.slider("Skor Stres (1-10)", 1, 10)
    anxiety = st.slider("Skor Kecemasan (1-10)", 1, 10)

    submit = st.form_submit_button("Prediksi")

if submit:
    # Proses encoding gender
    gender_num = 1 if gender.lower() == "laki-laki" else 0

    # Susun fitur
    features = np.array([[gender_num, age, cgpa, sleepinghours, stress, anxiety]])

    # Prediksi
    prediction = model.predict(features)[0]

    st.subheader("Hasil Prediksi:")
    if prediction == 1:
        st.error("Responden berpotensi mengalami depresi.")
    else:
        st.success("Responden tidak menunjukkan indikasi depresi.")
