import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Status Mahasiswa Jaya-Jaya Institut", page_icon="🎓", layout="wide")

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    model_path = "dropout_model_rf_final.joblib"
    le_path = "label_encoder.joblib"
    if all(os.path.isfile(path) for path in [model_path, le_path]):
        try:
            model = joblib.load(model_path)
            le = joblib.load(le_path)
            return model, le
        except Exception as e:
            st.error(f"Gagal memuat file: {e}")
            return None, None
    else:
        st.error("File model atau label encoder tidak ditemukan.")
        return None, None

model, le = load_model()

# Daftar fitur terpilih (23 fitur)
selected_features = [
    "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade",
    "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade",
    "Curricular_units_2nd_sem_evaluations",
    "Tuition_fees_up_to_date",
    "Curricular_units_1st_sem_evaluations",
    "Admission_grade",
    "Age_at_enrollment",
    "Course",
    "Previous_qualification_grade",
    "Fathers_occupation",
    "Mothers_occupation",
    "GDP",
    "Curricular_units_2nd_sem_enrolled",
    "Unemployment_rate",
    "Inflation_rate",
    "Mothers_qualification",
    "Fathers_qualification",
    "Application_mode",
    "Curricular_units_1st_sem_enrolled",
    "Scholarship_holder",
    "Application_order"
]

# Mapping kategorik
course_map = {
    "Biofuel Production Technologies": 33, "Animation and Multimedia Design": 171,
    "Social Service (evening attendance)": 8014, "Agronomy": 9003,
    "Communication Design": 9070, "Veterinary Nursing": 9085,
    "Informatics Engineering": 9119, "Equinculture": 9130, "Management": 9147,
    "Social Service": 9238, "Tourism": 9254, "Nursing": 9500, "Oral Hygiene": 9556,
    "Advertising and Marketing Management": 9670, "Journalism and Communication": 9773,
    "Basic Education": 9853, "Management (evening attendance)": 9991
}

binary_map = {"Yes": 1, "No": 0}

# Sidebar
st.sidebar.header("Tentang Aplikasi")
st.sidebar.info("""
Aplikasi ini memprediksi status mahasiswa (Dropout, Enrolled, Graduate) di Jaya-Jaya Institut menggunakan model Random Forest. Model dilatih dengan data 4000+ mahasiswa, mencapai akurasi 76%. Fitur utama meliputi:
- Prestasi akademik (unit disetujui, nilai semester)
- Status keuangan (biaya kuliah, beasiswa)
- Usia saat pendaftaran
- Indikator ekonomi (GDP, tingkat pengangguran)
""")

# Judul
st.title("🎓 Prediksi Status Mahasiswa Jaya-Jaya Institut")
st.write("Masukkan data mahasiswa untuk memprediksi status akademiknya.")

# Form input
with st.form("input_form"):
    st.header("Data Mahasiswa")
    col1, col2 = st.columns(2)

    # Akademik
    with col1:
        st.subheader("Prestasi Akademik")
        curricular_2nd_approved = st.number_input("Unit disetujui (Sem 2)", min_value=0, value=4)
        curricular_2nd_grade = st.slider("Rata-rata nilai (Sem 2)", 0.0, 20.0, 12.5, 0.1)
        curricular_1st_approved = st.number_input("Unit disetujui (Sem 1)", min_value=0, value=4)
        curricular_1st_grade = st.slider("Rata-rata nilai (Sem 1)", 0.0, 20.0, 13.0, 0.1)
        curricular_2nd_evals = st.number_input("Evaluasi (Sem 2)", min_value=0, value=6)
        curricular_1st_evals = st.number_input("Evaluasi (Sem 1)", min_value=0, value=5)
        curricular_2nd_enrolled = st.number_input("Unit diambil (Sem 2)", min_value=0, value=6)
        curricular_1st_enrolled = st.number_input("Unit diambil (Sem 1)", min_value=0, value=6)

    # Demografi dan keuangan
    with col2:
        st.subheader("Demografi dan Keuangan")
        age = st.number_input("Usia saat pendaftaran", min_value=17, max_value=70, value=20)
        tuition_fees_up_to_date = st.selectbox("Biaya kuliah lunas", options=["Yes", "No"], index=1)
        scholarship_holder = st.selectbox("Pemegang beasiswa", options=["Yes", "No"], index=0)
        admission_grade = st.slider("Nilai penerimaan", 0.0, 200.0, 120.0, 0.1)
        previous_qualification_grade = st.slider("Nilai kualifikasi sebelumnya", 0.0, 200.0, 125.0, 0.1)
        course = st.selectbox("Program studi", options=list(course_map.keys()))

    # Latar belakang keluarga
    st.subheader("Latar Belakang Keluarga")
    col3, col4 = st.columns(2)
    with col3:
        mothers_qualification = st.number_input("Kualifikasi ibu", min_value=1, value=3)
        fathers_qualification = st.number_input("Kualifikasi ayah", min_value=1, value=3)
    with col4:
        mothers_occupation = st.number_input("Pekerjaan ibu", min_value=0, value=4)
        fathers_occupation = st.number_input("Pekerjaan ayah", min_value=0, value=5)

    # Konteks ekonomi dan aplikasi
    st.subheader("Konteks Ekonomi dan Aplikasi")
    col5, col6, col7 = st.columns(3)
    with col5:
        unemployment_rate = st.slider("Tingkat pengangguran (%)", 0.0, 20.0, 9.0, 0.1)
        inflation_rate = st.slider("Tingkat inflasi (%)", -10.0, 10.0, 0.5, 0.1)
        gdp = st.slider("GDP", -10.0, 10.0, 1.74, 0.1)
    with col6:
        application_mode = st.number_input("Mode aplikasi", min_value=1, value=1)
        application_order = st.number_input("Urutan aplikasi", min_value=0, max_value=9, value=1)
    with col7:
        st.write("")  # Placeholder untuk keseimbangan layout

    # Tombol submit
    submitted = st.form_submit_button("Prediksi Status")

# Proses prediksi
if submitted and model:
    try:
        # Buat DataFrame dari input
        df_dummy = pd.DataFrame({
            "Curricular_units_2nd_sem_approved": [curricular_2nd_approved],
            "Curricular_units_2nd_sem_grade": [curricular_2nd_grade],
            "Curricular_units_1st_sem_approved": [curricular_1st_approved],
            "Curricular_units_1st_sem_grade": [curricular_1st_grade],
            "Curricular_units_2nd_sem_evaluations": [curricular_2nd_evals],
            "Tuition_fees_up_to_date": [binary_map[tuition_fees_up_to_date]],
            "Curricular_units_1st_sem_evaluations": [curricular_1st_evals],
            "Admission_grade": [admission_grade],
            "Age_at_enrollment": [age],
            "Course": [course_map[course]],
            "Previous_qualification_grade": [previous_qualification_grade],
            "Fathers_occupation": [fathers_occupation],
            "Mothers_occupation": [mothers_occupation],
            "GDP": [gdp],
            "Curricular_units_2nd_sem_enrolled": [curricular_2nd_enrolled],
            "Unemployment_rate": [unemployment_rate],
            "Inflation_rate": [inflation_rate],
            "Mothers_qualification": [mothers_qualification],
            "Fathers_qualification": [fathers_qualification],
            "Application_mode": [application_mode],
            "Curricular_units_1st_sem_enrolled": [curricular_1st_enrolled],
            "Scholarship_holder": [binary_map[scholarship_holder]],
            "Application_order": [application_order]
        })

        # Pastikan urutan kolom
        df_dummy = df_dummy[selected_features]

        # Prediksi
        prediction = model.predict(df_dummy)
        probabilities = model.predict_proba(df_dummy)
        proba_dropout = probabilities[0, 0] * 100
        proba_enrolled = probabilities[0, 1] * 100
        proba_graduate = probabilities[0, 2] * 100

        # Mapping
        mapping = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}

        # Tampilkan hasil
        st.header("Hasil Prediksi")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Status dan Probabilitas")
            st.success(f"**Prediksi**: {mapping[prediction[0]]}")
            proba_df = pd.DataFrame({
                "Status": ["Dropout", "Enrolled", "Graduate"],
                "Probabilitas (%)": [proba_dropout, proba_enrolled, proba_graduate]
            })
            fig = px.bar(proba_df, x="Status", y="Probabilitas (%)", title="Probabilitas Status",
                         color="Status", color_discrete_map={"Dropout": "#FF4B4B", "Enrolled": "#FFD700", "Graduate": "#4CAF50"})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Kategori Risiko dan Rekomendasi")
            dropout_risk = proba_dropout / 100
            if dropout_risk < 0.3:
                risk_category = "Rendah 🟢"
                recommendations = """
                - Mahasiswa menunjukkan performa akademik yang baik.
                - Pertahankan dengan pemantauan rutin.
                - Dorong partisipasi dalam kegiatan ekstrakurikuler.
                """
            elif dropout_risk < 0.7:
                risk_category = "Sedang 🟠"
                recommendations = """
                - Lakukan konseling akademik untuk mengatasi tantangan.
                - Pertimbangkan bimbingan belajar tambahan.
                - Tinjau jadwal kuliah untuk keseimbangan.
                """
            else:
                risk_category = "Tinggi 🔴"
                recommendations = """
                - Segera lakukan intervensi akademik dan konseling.
                - Jelajahi bantuan keuangan jika biaya kuliah menjadi kendala.
                - Sediakan dukungan kesehatan mental.
                - Buat rencana perbaikan akademik.
                """

            st.write(f"**Risiko Dropout**: {risk_category}")
            st.write("**Rekomendasi**:")
            st.markdown(recommendations)

            # Faktor risiko utama
            st.write("**Faktor Risiko Utama**:")
            factors = []
            if curricular_2nd_approved < 3:
                factors.append("Jumlah unit disetujui semester 2 rendah")
            if curricular_2nd_grade < 10:
                factors.append("Nilai rata-rata semester 2 rendah")
            if curricular_1st_approved < 3:
                factors.append("Jumlah unit disetujui semester 1 rendah")
            if tuition_fees_up_to_date == "No":
                factors.append("Biaya kuliah belum lunas")
            if age > 30:
                factors.append("Usia pendaftaran lebih tua")
            if factors:
                for i, factor in enumerate(factors[:3], 1):
                    st.write(f"{i}. {factor}")
            else:
                st.write("Tidak ada faktor risiko signifikan.")

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {str(e)}")
else:
    st.info("Isi form dan klik 'Prediksi Status' untuk melihat hasil.")