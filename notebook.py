# -*- coding: utf-8 -*-
"""Submission Akhir_Student DO Predictive.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Gj9RIUJdDewa79QDMnB2LiGrOJbUQQac

# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

- Nama: Dewi Rachmawati
- Email:rchmw.dewi10@gmail.com
- Id Dicoding:iam_dewir

## Persiapan

### Menyiapkan library yang dibutuhkan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sqlalchemy import create_engine
import os
import joblib

"""### Menyiapkan data yang akan diguankan"""

url = "https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/students_performance/data.csv"
df = pd.read_csv(url, sep=";")
print(f'Dataset memiliki {df.shape[0]} baris dan {df.shape[1]} kolom')

"""melihat 5 baris pertama dataset"""

df.head()

"""## Data Understanding

Analisis Struktur Dataset
"""

df.info()

"""**Informasi Dataset Mahasiswa**

Dataset ini berisi 4.424 entri mahasiswa dengan 37 atribut/kolom yang menggambarkan aspek demografis, latar belakang pendidikan, kondisi akademik, serta variabel makroekonomi saat pendaftaran. Dataset ini ditujukan untuk analisis status pendidikan mahasiswa, termasuk prediksi kelulusan dan dropout.

---

**Struktur Dataset Mahasiswa**

A. Atribut Demografis & Sosial

| Nama Kolom                  | Deskripsi                               | Tipe Data     |
|-----------------------------|------------------------------------------|---------------|
| Marital_status              | Status pernikahan mahasiswa              | Kategorikal   |
| Nationality                 | Kewarganegaraan                          | Kategorikal   |
| Gender                      | Jenis kelamin mahasiswa                  | Kategorikal   |
| Displaced                   | Status pengungsi/pemindahan              | Kategorikal   |
| Educational_special_needs  | Memiliki kebutuhan pendidikan khusus     | Kategorikal   |
| Debtor                     | Status memiliki utang                    | Kategorikal   |
| International               | Mahasiswa internasional                  | Kategorikal   |

B. Atribut Orang Tua

| Nama Kolom            | Deskripsi                    | Tipe Data   |
|------------------------|-------------------------------|-------------|
| Mothers_qualification  | Pendidikan ibu                | Kategorikal |
| Fathers_qualification  | Pendidikan ayah               | Kategorikal |
| Mothers_occupation     | Pekerjaan ibu                 | Kategorikal |
| Fathers_occupation     | Pekerjaan ayah                | Kategorikal |

C. Atribut Pendaftaran & Administrasi

| Nama Kolom                   | Deskripsi                               | Tipe Data     |
|------------------------------|------------------------------------------|---------------|
| Application_mode             | Jalur pendaftaran                        | Kategorikal   |
| Application_order            | Urutan pilihan studi                     | Numerik       |
| Course                       | Program studi                            | Kategorikal   |
| Daytime_evening_attendance  | Kehadiran siang/malam                    | Kategorikal   |
| Previous_qualification       | Pendidikan sebelumnya                    | Kategorikal   |
| Previous_qualification_grade | Nilai pendidikan sebelumnya              | Desimal       |
| Admission_grade              | Nilai seleksi masuk                      | Desimal       |
| Tuition_fees_up_to_date      | Status pembayaran kuliah                 | Kategorikal   |
| Scholarship_holder           | Status penerima beasiswa                 | Kategorikal   |
| Age_at_enrollment            | Usia saat mendaftar                      | Numerik       |

D. Atribut Akademik Semester 1

| Nama Kolom                                  | Deskripsi                       | Tipe Data   |
|---------------------------------------------|----------------------------------|-------------|
| Curricular_units_1st_sem_credited           | SKS dikreditkan semester 1       | Numerik     |
| Curricular_units_1st_sem_enrolled           | SKS diambil semester 1           | Numerik     |
| Curricular_units_1st_sem_evaluations        | Evaluasi semester 1              | Numerik     |
| Curricular_units_1st_sem_approved           | SKS lulus semester 1             | Numerik     |
| Curricular_units_1st_sem_grade              | Rata-rata nilai semester 1       | Desimal     |
| Curricular_units_1st_sem_without_evaluations| SKS tanpa evaluasi semester 1    | Numerik     |

E. Atribut Akademik Semester 2

| Nama Kolom                                  | Deskripsi                       | Tipe Data   |
|---------------------------------------------|----------------------------------|-------------|
| Curricular_units_2nd_sem_credited           | SKS dikreditkan semester 2       | Numerik     |
| Curricular_units_2nd_sem_enrolled           | SKS diambil semester 2           | Numerik     |
| Curricular_units_2nd_sem_evaluations        | Evaluasi semester 2              | Numerik     |
| Curricular_units_2nd_sem_approved           | SKS lulus semester 2             | Numerik     |
| Curricular_units_2nd_sem_grade              | Rata-rata nilai semester 2       | Desimal     |
| Curricular_units_2nd_sem_without_evaluations| SKS tanpa evaluasi semester 2    | Numerik     |

F. Atribut Makroekonomi

| Nama Kolom       | Deskripsi                            | Tipe Data   |
|------------------|----------------------------------------|-------------|
| Unemployment_rate| Tingkat pengangguran saat pendaftaran | Desimal     |
| Inflation_rate   | Tingkat inflasi saat pendaftaran      | Desimal     |
| GDP              | Produk Domestik Bruto per kapita      | Numerik     |

G. Variabel Target

| Nama Kolom | Deskripsi                    | Tipe Data   |
|------------|-------------------------------|-------------|
| Status     | Status akhir mahasiswa        | Kategorikal |
"""

# Nilai unik di kolom Status
print("\nNilai Unik di Kolom Status:")
print(df['Status'].unique())

"""Kolom **Status** merupakan **atribut/variabel target** dengan tipe data kategorikal: dropout (tidak menyelesaikan pendidikan), graduate (lulus), dan enrolled (aktif)

Cek Kualitas Data
"""

# cek missing values
print("\nJumlah nilai yang hilang dalam setiap kolom:")
print(df.isnull().sum())

"""Tidak ditemukan adanya nilai yang hilang sehingga perlu dicek apakah terdapat data yang terduplikasi"""

# Cek duplikasi data
print("\nBanyaknya data yang duplikat:", df.duplicated().sum())

"""Tidak ada data yang teduplikasi sehingga dataset mahasiswa ini dapat dibilang sudah bersih.

Analisis Statistik Deskriptif
"""

pd.set_option('display.max_columns', None)
df.describe(include='all')

"""Analisis Korelasi"""

# Daftar fitur numerik untuk Jaya-Jaya Institut
numerical_features = [
    'Age_at_enrollment', 'Application_order', 'Previous_qualification_grade',
    'Admission_grade', 'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
    'Unemployment_rate', 'Inflation_rate', 'GDP', 'International',
    'Fathers_occupation', 'Mothers_occupation'
]

# Filter dataset untuk fitur numerik
numeric_df = df[numerical_features]

# Hitung matriks korelasi
correlation_matrix = numeric_df.corr(method='pearson')

# Buat figure dengan ukuran besar
plt.figure(figsize=(18, 15))

# Buat heatmap dengan anotasi selektif
mask = np.abs(correlation_matrix) < 0.7  # Sembunyikan korelasi <0.7
sns.heatmap(
    correlation_matrix,
    annot=np.where(~mask, correlation_matrix.round(2), ''),  # Anotasi hanya untuk |r|>=0.7
    fmt='s',  # Format string untuk menangani anotasi kosong
    cmap='RdBu_r',  # Skema warna merah-biru
    center=0,  # Pusatkan skala di 0
    square=True,  # Sel persegi
    linewidths=0.5,  # Garis antar-sel
    linecolor='gray',  # Warna garis abu-abu
    cbar_kws={'label': 'Korelasi Pearson', 'shrink': 0.8}  # Label dan ukuran colorbar
)

# Judul dan pengaturan sumbu
plt.title('Matriks Korelasi Fitur Numerik - Jaya-Jaya Institut', fontsize=20, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# Atur margin untuk mencegah tumpang tindih
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.25)

# Tampilkan plot
plt.show()

# Cetak pasangan dengan korelasi tinggi (|r| > 0.7)
print("\nPasangan Fitur dengan Korelasi Tinggi (|r| > 0.7):")
corr_melt = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
high_corr = corr_melt[np.abs(corr_melt) > 0.7]
for col in high_corr.columns:
    for idx in high_corr.index:
        if pd.notna(high_corr.loc[idx, col]):
            print(f"{idx} vs {col}: {high_corr.loc[idx, col]:.2f}")

"""Eksplorasi Data Analysis (EDA)

1. Berapa jumlah dan persentase status mahasiswa pada Jaya-Jaya Institut?
"""

# Hitung Jumlah dan Persentase Status
status_counts = df['Status'].value_counts().reset_index()
status_counts.columns = ['Status', 'Jumlah']
status_counts['Persentase'] = (status_counts['Jumlah'] / status_counts['Jumlah'].sum() * 100).round(2)
print("\nJumlah dan Persentase Status Mahasiswa:")
print(status_counts)

fig = px.bar(status_counts, x='Status', y='Jumlah',
             title='Jumlah dan Persentase Mahasiswa Berdasarkan Status',
             color='Status', color_discrete_map={'Dropout': '#FF6361', 'Graduate': '#2ECC71', 'Enrolled': '#FFA500'},
             text='Persentase')
fig.update_layout(xaxis_title="Status Mahasiswa",
                 yaxis_title="Jumlah Mahasiswa",
                 showlegend=True)
fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
fig.show()

"""**Jawab:** Jumlah dan persentase status mahasiswa graduate, dropout, dan enrolled berturut turut adalah 2209 (49,93%), 1421 (32,12%), dan 794 (17,95%).

2. Apakah mahasiswa yang mendaftar melalui jalur aplikasi tertentu (Application_mode) memiliki tingkat kelulusan yang lebih tinggi dibandingkan yang dropout?
"""

# Pertanyaan 2: Pengaruh Application_mode terhadap Status
app_mode_status = df.groupby(['Application_mode', 'Status']).size().reset_index(name='Count')
app_mode_status['Percentage'] = app_mode_status.groupby('Application_mode')['Count'].transform(lambda x: x / x.sum() * 100)
fig1 = px.bar(app_mode_status, x='Application_mode', y='Percentage', color='Status',
              title='Persentase Status Mahasiswa Berdasarkan Jalur Aplikasi',
              color_discrete_map={'Dropout': '#FF6361', 'Graduate': '#2ECC71', 'Enrolled': '#FFA500'},
              text='Percentage')
fig1.update_layout(xaxis_title="Jalur Aplikasi", yaxis_title="Persentase (%)")
fig1.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
fig1.show()

"""Jalur aplikasi yang digunakan mahasiswa:

1 - 1st phase - general contingent\
2 - Ordinance No. 612/93\
5 - 1st phase - special contingent (Azores Island)\
7 - Holders of other higher courses
10 - Ordinance No. 854-B/99\
15 - International student (bachelor)\
16 - 1st phase - special contingent (Madeira Island)\
17 - 2nd phase - general contingent\
18 - 3rd phase - general contingent\
26 - Ordinance No. 533-A/99, item b2) (Different Plan)\
27 - Ordinance No. 533-A/99, item b3 (Other Institution)\
39 - Over 23 years old\
42 - Transfer\
43 - Change of course\
44 - Technological specialization diploma holders\
51 - Change of institution/course\
53 - Short cycle diploma holders\
57 - Change of institution/course (International)\

**Jawab:** \
Ya, mahasiswa yang mendaftar jalur aplikasi internasional memiliki persentase kelulusan (graduate) tertinggi sebesar 100%, diikuti jalur pendaftaran yang diatur spesifik pemerintah Portugal berdasarkan peraturan No. 854-B/99 sebesar 70% dan jalur aplikasi prioritas untuk warga Maidera sebesar 65,79%. Kemudian mahasiswa yang mendaftar jalur aplikasi peraturan No.533-A/99 item b2 dan b3 memiliki persentase dropout tertinggi sebesar 100%, diikuti jalur pendaftaran berdasarkan regulasi khusus dari Portugal No. 612/93 sebesar 66,66%.

3. Bagaimana pengaruh status beasiswa (Scholarship_holder) terhadap kemungkinan mahasiswa menyelesaikan studi tepat waktu?
"""

# Pertanyaan 3: Pengaruh status beasiswa terhadap status mahasiswa
# Hitung distribusi Status berdasarkan Scholarship_holder
scholarship_status = df.groupby(['Scholarship_holder', 'Status']).size().reset_index(name='Count')
scholarship_status['Percentage'] = scholarship_status.groupby('Scholarship_holder')['Count'].transform(lambda x: x / x.sum() * 100)

fig = px.bar(scholarship_status, x='Scholarship_holder', y='Percentage', color='Status',
             title='Persentase Status Mahasiswa Berdasarkan Status Beasiswa',
             color_discrete_map={'Dropout': '#FF6361', 'Graduate': '#2ECC71', 'Enrolled': '#FFA500'},
             text='Percentage')
fig.update_layout(xaxis_title="Status Beasiswa (0: Tidak, 1: Ya)", yaxis_title="Persentase (%)")
fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
fig.show()

"""**Jawab:** \
Persentase mahasiswa dropout bukan penerima beasiswa cenderung lebih besar yakni 38,7% ketimbang mahasiswa berstatus penerima beasiswa yakni 12,2%. Oleh karena itu, persentase kelulusan (graduate) mahasiswa lebih besar pada status penerima beasiswa yakni 76%.

4. Apakah mahasiswa internasional memiliki risiko dropout yang berbeda dibandingkan mahasiswa lokal?
"""

# Pertanyaan 4: Pengaruh International terhadap Status
international_status = df.groupby(['International', 'Status']).size().reset_index(name='Count')
international_status['Percentage'] = international_status.groupby('International')['Count'].transform(lambda x: x / x.sum() * 100)
fig3 = px.bar(international_status, x='International', y='Percentage', color='Status',
              title='Persentase Status Mahasiswa Berdasarkan Status Internasional',
              color_discrete_map={'Dropout': '#FF6361', 'Graduate': '#2ECC71', 'Enrolled': '#FFA500'},
              text='Percentage')
fig3.update_layout(xaxis_title="Status Internasional (0: Lokal, 1: Internasional)", yaxis_title="Persentase (%)")
fig3.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
fig3.show()

"""**Jawab:**
Perbedaan tidak terlalu signifikan, justru mahasiswa lokal cenderung memiliki risiko dropout lebih besar sekitar 32,2% daripada mahasiswa internasional sekitar 29,1%.

5. Bagaimana pola distribusi jumlah evaluasi semester kedua (Curricular_units_2nd_sem_evaluations) pada mahasiswa dengan status dropout dibandingkan dengan graduate dan enrolled?
"""

# Pertanyaan 5: Pola distribusi Curricular_units_2nd_sem_evaluations dengan Status
fig4 = px.box(df, x='Status', y='Curricular_units_2nd_sem_evaluations',
              title='Distribusi Jumlah Evaluasi Semester 2 Berdasarkan Status',
              color='Status', color_discrete_map={'Dropout': '#FF6361', 'Graduate': '#2ECC71', 'Enrolled': '#FFA500'})
fig4.update_layout(xaxis_title="Status", yaxis_title="Jumlah Evaluasi Semester 2")
fig4.show()

"""**Jawab:** \
Pada boxplot di atas, distribusi mahasiswa dropout memiliki jumlah evaluasi semester kedua lebih rendah (median=7) dibandingkan graduate (median=8) dan enrolled (median=9), dengan variabilitas yang lebih tinggi (IQR=5–10). Mahasiswa enrolled mengalami evaluasi lebih sering (IQR=8–11), menunjukkan pemantauan akademik lebih intensif dibandingkan kelompok lainnya. Mahasiswa graduate memiliki distribusi yang lebih stabil (IQR=6–9), yang mencerminkan jalur akademik yang lebih terstruktur menuju kelulusan. Temuan ini menunjukkan bahwa jumlah evaluasi semester kedua tidak secara langsung berkorelasi dengan status dropout, melainkan ada faktor lain seperti keterlibatan akademik atau kebijakan evaluasi yang berperan dalam retensi mahasiswa.

6. Apakah tingkat pengangguran (Unemployment_rate) saat pendaftaran memengaruhi kemungkinan mahasiswa untuk dropout?
"""

# Pertanyaan 6: Pengaruh Unemployment_rate terhadap Status
fig5 = px.box(df, x='Status', y='Unemployment_rate',
              title='Distribusi Tingkat Pengangguran Saat Pendaftaran Berdasarkan Status',
              color='Status', color_discrete_map={'Dropout': '#FF6361', 'Graduate': '#2ECC71', 'Enrolled': '#FFA500'})
fig5.update_layout(xaxis_title="Status", yaxis_title="Tingkat Pengangguran (%)")
fig5.show()

"""**Jawab:**\
Tidak terlihat pola signifikan yang menunjukkan bahwa tingkat pengangguran saat pendaftaran memengaruhi status dropout secara langsung. Jika tingkat pengangguran berperan, efeknya mungkin tidak linier atau dipengaruhi oleh variabel lain seperti kondisi ekonomi individu, dukungan finansial, atau motivasi akademik.
"""

# Visualisasi Tambahan: Scatter Plot untuk korelasi Unemployment_rate dan Curricular_units_2nd_sem_grade guna mengetahui hubungan ekonomi dengan performa akademik
fig6 = px.scatter(df, x='Unemployment_rate', y='Curricular_units_2nd_sem_grade', color='Status',
                  title='Hubungan Tingkat Pengangguran dan Nilai Semester 2',
                  color_discrete_map={'Dropout': '#FF6361', 'Graduate': '#2ECC71', 'Enrolled': '#FFA500'})
fig6.update_layout(xaxis_title="Tingkat Pengangguran (%)", yaxis_title="Nilai Semester 2")
fig6.show()

"""Scatter plot menunjukkan bahwa tidak ada pola korelasi yang jelas antara tingkat pengangguran saat pendaftaran (unemployment_rate) dan performa akademik mahasiswa pada semester kedua (curricular_units_2nd_sem_grade), karena distribusi data tersebar tanpa tren linier atau pola monotonic yang signifikan. Mayoritas mahasiswa memiliki nilai semester kedua dalam kisaran 10–15, terlepas dari tingkat pengangguran, yang menunjukkan bahwa faktor ekonomi saat pendaftaran tidak secara langsung mempengaruhi pencapaian akademik mereka. Selain itu, mahasiswa dengan status dropout tidak menunjukkan karakteristik khusus dalam distribusi nilai mereka yang berbeda dari graduate atau enrolled, mengindikasikan bahwa variabel lain, seperti motivasi akademik, dukungan finansial, atau keterlibatan dalam studi, mungkin lebih berperan dalam menentukan performa akademik dan kemungkinan dropout.

## Data Preparation / Preprocessing
"""

# Create Copies for Preprocessing
df_labeled = df.copy()  # For descriptive label (visualisasi)
df_numeric = df.copy()  # For numeric values (modeling)

# 4. Mapping kategori menjadi label deskriptif
mappings = {
    'Marital_status': {
        1: 'Single', 2: 'Married', 3: 'Widowed', 4: 'Divorced', 5: 'Common-Law', 6: 'Separated'
    },
    'Application_mode': {
        1: 'Phase 1 - General', 2: 'Ordinance 612/93', 5: 'Phase 1 - Azores', 7: 'Higher Course Holders',
        10: 'Ordinance 854-B/99', 15: 'International Student', 16: 'Phase 1 - Madeira', 17: 'Phase 2 - General',
        18: 'Phase 3 - General', 26: 'Ordinance 533-A/99 b2', 27: 'Ordinance 533-A/99 b3', 39: 'Over 23 Years',
        42: 'Transfer', 43: 'Course Change', 44: 'Tech Diploma', 51: 'Institution Change', 53: 'Short Cycle Diploma',
        57: 'International Transfer'
    },
    'Course': {
        33: 'Biofuel Tech', 171: 'Animation Design', 8014: 'Social Service Evening', 9003: 'Agronomy',
        9070: 'Communication Design', 9085: 'Veterinary Nursing', 9119: 'Informatics Engineering',
        9130: 'Equinculture', 9147: 'Management', 9238: 'Social Service', 9254: 'Tourism', 9500: 'Nursing',
        9556: 'Oral Hygiene', 9670: 'Marketing', 9773: 'Journalism', 9853: 'Basic Education',
        9991: 'Management Evening'
    },
    'Daytime_evening_attendance': {
        0: 'Evening', 1: 'Daytime'
    },
    'Previous_qualification': {
        1: 'Secondary Education', 2: 'Bachelor Degree', 3: 'Degree', 4: 'Master Degree', 5: 'Doctorate',
        6: 'Higher Ed Frequency', 9: '12th Incomplete', 10: '11th Incomplete', 12: 'Other 11th', 14: '10th Year',
        15: '10th Incomplete', 19: 'Basic Ed 3rd Cycle', 38: 'Basic Ed 2nd Cycle', 39: 'Tech Specialization',
        40: 'Degree 1st Cycle', 42: 'Professional Tech Course', 43: 'Master 2nd Cycle'
    },
    'Nacionality': {
        1: 'Portuguese', 2: 'German', 6: 'Spanish', 11: 'Italian', 13: 'Dutch', 14: 'English', 17: 'Lithuanian',
        21: 'Angolan', 22: 'Cape Verdean', 24: 'Guinean', 25: 'Mozambican', 26: 'Santomean', 32: 'Turkish',
        41: 'Brazilian', 62: 'Romanian', 100: 'Moldova', 101: 'Mexican', 103: 'Ukrainian', 105: 'Russian',
        108: 'Cuban', 109: 'Colombian'
    },
    'Mothers_qualification': {
        1: 'Secondary Ed', 2: 'Bachelor', 3: 'Degree', 4: 'Master', 5: 'Doctorate', 6: 'Higher Ed Frequency',
        9: '12th Incomplete', 10: '11th Incomplete', 11: '7th Year Old', 12: 'Other 11th', 14: '10th Year',
        18: 'Commerce Course', 19: 'Basic Ed 3rd Cycle', 22: 'Technical Course', 26: '7th Year',
        27: '2nd Cycle High School', 29: '9th Incomplete', 30: '8th Year', 34: 'Unknown',
        35: 'Illiterate', 36: 'Can Read', 37: 'Basic Ed 1st Cycle', 38: 'Basic Ed 2nd Cycle',
        39: 'Tech Course', 40: 'Degree 1st Cycle', 41: 'Specialized Course', 42: 'Professional Tech Course',
        43: 'Master 2nd Cycle', 44: 'Doctorate 3rd Cycle'
    },
    'Fathers_qualification': {
        1: 'Secondary Ed', 2: 'Bachelor', 3: 'Degree', 4: 'Master', 5: 'Doctorate', 6: 'Higher Ed Frequency',
        9: '12th Incomplete', 10: '11th Incomplete', 11: '7th Year Old', 12: 'Other 11th', 13: '2nd Year High School',
        14: '10th Year', 18: 'Commerce Course', 19: 'Basic Ed 3rd Cycle', 20: 'High School Course',
        22: 'Technical Course', 25: 'High School Incomplete', 26: '7th Year', 27: '2nd Cycle High School',
        29: '9th Incomplete', 30: '8th Year', 31: 'Admin Course', 33: 'Accounting Course', 34: 'Unknown',
        35: 'Illiterate', 36: 'Can Read', 37: 'Basic Ed 1st Cycle', 38: 'Basic Ed 2nd Cycle',
        39: 'Tech Course', 40: 'Degree 1st Cycle', 41: 'Specialized Course', 42: 'Professional Tech Course',
        43: 'Master 2nd Cycle', 44: 'Doctorate 3rd Cycle'
    },
    'Mothers_occupation': {
        0: 'Student', 1: 'Legislative', 2: 'Scientific', 3: 'Technician', 4: 'Admin', 5: 'Service',
        6: 'Agriculture', 7: 'Industry', 8: 'Machine Operator', 9: 'Unskilled', 10: 'Armed Forces',
        90: 'Other', 99: 'Unknown', 122: 'Health Professional', 123: 'Teacher', 125: 'ICT Specialist',
        131: 'Science Technician', 132: 'Health Technician', 134: 'Social Technician', 141: 'Office Worker',
        143: 'Accounting', 144: 'Admin Support', 151: 'Personal Service', 152: 'Sales', 153: 'Care Worker',
        171: 'Construction', 173: 'Precision Craft', 175: 'Food Processing', 191: 'Cleaning',
        192: 'Unskilled Agriculture', 193: 'Unskilled Industry', 194: 'Meal Prep'
    },
    'Fathers_occupation': {
        0: 'Student', 1: 'Legislative', 2: 'Scientific', 3: 'Technician', 4: 'Admin', 5: 'Service',
        6: 'Agriculture', 7: 'Industry', 8: 'Machine Operator', 9: 'Unskilled', 10: 'Armed Forces',
        90: 'Other', 99: 'Unknown', 101: 'Armed Forces Officer', 102: 'Armed Forces Sergeant',
        103: 'Armed Forces Personnel', 112: 'Admin Director', 114: 'Service Director', 121: 'Physical Scientist',
        122: 'Health Professional', 123: 'Teacher', 124: 'Finance Specialist', 131: 'Science Technician',
        132: 'Health Technician', 134: 'Social Technician', 135: 'ICT Technician', 141: 'Office Worker',
        143: 'Accounting', 144: 'Admin Support', 151: 'Personal Service', 152: 'Sales', 153: 'Care Worker',
        154: 'Security', 161: 'Farmer', 163: 'Subsistence Farmer', 171: 'Construction', 172: 'Metalworking',
        174: 'Electrical', 175: 'Food Processing', 181: 'Plant Operator', 182: 'Assembler', 183: 'Driver',
        192: 'Unskilled Agriculture', 193: 'Unskilled Industry', 194: 'Meal Prep', 195: 'Street Vendor'
    },
    'Gender': {
        1: 'Male', 0: 'Female'
    },
    'Scholarship_holder': {
        1: 'Yes', 0: 'No'
    },
    'Displaced': {
        1: 'Yes', 0: 'No'
    },
    'Educational_special_needs': {
        1: 'Yes', 0: 'No'
    },
    'Debtor': {
        1: 'Yes', 0: 'No'
    },
    'Tuition_fees_up_to_date': {
        1: 'Paid', 0: 'Unpaid'
    },
    'International': {
        1: 'Yes', 0: 'No'
    },
    'Status': {
        'Dropout': 'Dropout', 'Graduate': 'Graduate', 'Enrolled': 'Enrolled'
    }
}

# Check for unmapped values and apply mappings
for col, mapping in mappings.items():
    if col in df_labeled.columns:
        # Check unique values to ensure all are mapped
        unique_values = df_labeled[col].unique()
        unmapped = [val for val in unique_values if val not in mapping]
        if unmapped:
            print(f"\nWarning: Unmapped values in {col}: {unmapped}")
            # Add unmapped values to mapping as strings
            for val in unmapped:
                mapping[val] = str(val)
        # Apply mapping
        df_labeled[col] = df_labeled[col].map(mapping)
        print(f"Applied mapping to column {col}.")
    else:
        print(f"Column {col} not found for mapping.")

# Ubah menjadi tipe kategori setelah mapping
categorical_columns = [
    'Marital_status', 'Application_mode', 'Course', 'Daytime_evening_attendance',
    'Previous_qualification', 'Nacionality', 'Mothers_qualification', 'Fathers_qualification',
    'Mothers_occupation', 'Fathers_occupation', 'Gender', 'Scholarship_holder',
    'Displaced', 'Educational_special_needs', 'Debtor', 'Tuition_fees_up_to_date',
    'International', 'Status'
]
for col in categorical_columns:
    if col in df_labeled.columns:
        df_labeled[col] = df_labeled[col].astype('category')
    else:
        print(f"Column {col} not found in dataset.")

print("\nInfo Dataset setelah mapping dan konversi menjadi kategori:")
print(df_labeled.info())

df_labeled.head()

df_numeric.head()

# Save labeled dataset to CSV untuk keperluan dashboard
df_labeled.to_csv('processed_labeled_dataset.csv', index=False)
print("\nLabeled dataset saved to 'processed_labeled_dataset.csv'")

# Save numeric dataset to CSV untuk keperluan modelling
df_numeric.to_csv('processed_numeric_dataset.csv', index=False)
print("Numeric dataset saved to 'processed_numeric_dataset.csv'")

"""Import to Supabase"""

# Fetch variables
USER = "postgres.hidlckzzlhsgyjcrtfpu" #USERNAME
PASSWORD = "Root123#LaskarAi" #PASWORD
HOST = "aws-0-ap-southeast-1.pooler.supabase.com" #HOSTNAME
PORT = "6543" #PORT
DBNAME = "postgres" #DBNAME

# Construct the SQLAlchemy connection string
DATABASE_URL = "postgresql://postgres.hidlckzzlhsgyjcrtfpu:Root123#LaskarAi@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)


# Test the connection
try:
    with engine.connect() as connection:
        print("Connection successful!")
except Exception as e:
    print(f"Failed to connect: {e}")

"""## Feature Importance dengan Modelling"""

# Load Numeric Dataset
df_numeric = pd.read_csv('/content/processed_numeric_dataset.csv')

# Pisahkan Features dan Target
X = df_numeric.drop('Status', axis=1)
y = df_numeric['Status']

# Encode Target (Status: Dropout, Graduate, Enrolled)
le = LabelEncoder()
y = le.fit_transform(y)
print("\nEncoded Classes:", dict(zip(le.classes_, range(len(le.classes_)))))

# Normalisasi fitur numerik
scaler = StandardScaler()
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
print("\nNumeric Features Normalized.")

# Split data 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

# Latih Inisiasi Random Forest untuk mendapatkan feature importance
initial_model = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2,
    class_weight='balanced', random_state=42
)
initial_model.fit(X_train, y_train)

# identifikasi fitur yang dapat didrop
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': initial_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Visualisasi feature importance
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in Initial Random Forest Model')
plt.show()

# Pilih features dengan importance melebihi nilai threshold (top 20 atau importance > 0.01)
importance_threshold = 0.01
features_to_keep = feature_importance[feature_importance['Importance'] > importance_threshold]['Feature'].tolist()
if not features_to_keep:
    features_to_keep = feature_importance['Feature'].head(20).tolist()  # Fallback to top 20
features_to_drop = [col for col in X.columns if col not in features_to_keep]
print(f"\nFeatures to Keep ({len(features_to_keep)}):", features_to_keep)
print(f"Features to Drop ({len(features_to_drop)}):", features_to_drop)

# Filter feature_importance untuk hanya menampilkan features_to_keep
features_to_keep_df = feature_importance[feature_importance['Feature'].isin(features_to_keep)][['Feature', 'Importance']]

# Urutkan fitur berdasarkan importance (dari besar ke kecil)
features_to_keep_df = features_to_keep_df.sort_values(by='Importance', ascending=False)

# Buat visualisasi
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=features_to_keep_df, hue='Feature', palette='viridis', legend=False)

# Tambahkan judul dan label
plt.title('Fitur Penting untuk Prediksi Status Mahasiswa', fontsize=14, pad=15)
plt.xlabel('Feature Importance', fontsize=12)
plt.ylabel('Fitur', fontsize=12)

# Tambahkan nilai importance di ujung bar
for index, value in enumerate(features_to_keep_df['Importance']):
    plt.text(value, index, f'{value:.4f}', va='center', ha='left', fontsize=10, color='black')

# Atur layout agar tidak terpotong
plt.tight_layout()

# Cetak fitur yang ditampilkan
print(f"\nFeatures Displayed ({len(features_to_keep_df)}):", features_to_keep_df['Feature'].tolist())

"""## Modelling"""

# Update training dan testing set dengan fitur terpilih
X_train_selected = X_train[features_to_keep]
X_test_selected = X_test[features_to_keep]
print("\nSelected Training Set Shape:", X_train_selected.shape)
print("Selected Testing Set Shape:", X_test_selected.shape)

# Latih RF dengan fitur terpilih
model = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2,
    class_weight='balanced', random_state=42
)
model.fit(X_train_selected, y_train)
print("\nModel Training Completed with Selected Features.")

"""## Evaluation"""

# Evaluate Model
y_pred = model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix for Random Forest with Selected Features')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Simpan Model
joblib.dump(model, 'dropout_model_rf_final.joblib')
joblib.dump(le, 'label_encoder.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("\nModel, Label Encoder, and Scaler saved.")

"""Model, encoder, dan scaler disimpan untuk keperluan deployment pada file `app.py`.

## Interpretasi Hasil

Berdasarkan feature importance dari model Random Forest (akurasi 76%), berikut adalah analisis 10 fitur terpenting yang memengaruhi prediksi status mahasiswa (Dropout, Enrolled, Graduate) di Jaya-Jaya Institut:

1. Dominasi Performa Akademik Semester Kedua\
* Fitur Utama:
  - Curricular_units_2nd_sem_approved (0.1503): Jumlah mata kuliah yang lulus di semester kedua adalah prediktor terkuat.
  - Curricular_units_2nd_sem_grade (0.0983): Nilai rata-rata semester kedua menempati peringkat kedua.
* Insight: Semester kedua adalah periode kritis untuk keberhasilan akademik. Mahasiswa dengan <3 mata kuliah lulus atau nilai rata-rata <10 berisiko tinggi dropout. Korelasi kuat dengan semester pertama (Curricular_units_1st_sem_approved, r=0.90; Curricular_units_1st_sem_grade, r=0.84) menunjukkan pola performa yang konsisten, tetapi semester kedua lebih menentukan, mungkin karena mencerminkan stabilitas akademik setelah adaptasi.
2. Performa Semester Pertama sebagai Pendukung
* Fitur:
  - Curricular_units_1st_sem_approved (0.0838): Jumlah mata kuliah lulus di semester pertama.
  - Curricular_units_1st_sem_grade (0.0670): Nilai rata-rata semester pertama.
* Insight:
Performa awal kuliah penting, tetapi kurang dominan dibandingkan semester kedua, menunjukkan bahwa tantangan awal (misalnya, adaptasi kampus) dapat diatasi seiring waktu.
Mahasiswa dengan performa lemah di semester pertama tetap perlu pemantauan untuk mencegah risiko jangka panjang.
3. Faktor Keuangan yang Signifikan
* Fitur:
  - Tuition_fees_up_to_date (0.0455): Status pembayaran biaya kuliah.
* Insight:
Tunggakan biaya kuliah meningkatkan risiko dropout, menegaskan bahwa stabilitas finansial adalah faktor kunci. Fitur ini memiliki bobot lebih tinggi dari beberapa fitur akademik lainnya, menunjukkan urgensi dukungan keuangan.
4. Keterlibatan dalam Evaluasi Akademik
* Fitur:
  - Curricular_units_2nd_sem_evaluations (0.0529): Jumlah evaluasi di semester kedua.
  - Curricular_units_1st_sem_evaluations (0.0430): Jumlah evaluasi di semester pertama.
* Insight:
Jumlah evaluasi (ujian, tugas) mencerminkan tingkat keterlibatan mahasiswa. Evaluasi rendah dapat menandakan kurangnya motivasi atau kesulitan akademik.
Bobot lebih rendah dibandingkan hasil evaluasi (lulus/nilai) menunjukkan bahwa kualitas performa lebih penting daripada kuantitas.
5. Profil dan Kesiapan Mahasiswa
* Fitur:
  - Admission_grade (0.0422): Nilai penerimaan kuliah.
  - Age_at_enrollment (0.0396): Usia saat mendaftar.
  - Course (0.0345): Program studi yang diambil.
* Insight:
Nilai penerimaan yang tinggi mengindikasikan kesiapan akademik, tetapi efeknya lebih kecil dibandingkan performa kuliah saat ini. Mahasiswa yang lebih tua (>30 tahun) mungkin menghadapi tantangan tambahan, seperti keseimbangan kerja-kuliah. Sedangkan program studi tertentu (misalnya, dengan kurikulum berat) dapat memengaruhi risiko dropout, meskipun pengaruhnya moderat.
6. Konteks dan Keterbatasan
* Fitur Lain:
Fitur seperti Previous_qualification_grade (0.0344) dan GDP (0.0246) tidak masuk 10 teratas tetapi tetap relevan dalam 23 fitur terpilih.
Fitur dengan importance rendah (misalnya, Educational_special_needs, 0.0002) menunjukkan dampak minimal pada prediksi.

Keterbatasan:
- Data Imbalance: Kelas Enrolled (F1-score 0.46, support 151) jauh lebih sulit diprediksi dibandingkan Dropout (0.79) dan Graduate (0.84), membatasi keandalan untuk kelas ini.
- Akurasi 76%: Cukup untuk intervensi praktis, tetapi performa Enrolled perlu ditingkatkan.
- Fokus Semester Kedua: Ketergantungan pada semester kedua menyulitkan prediksi dini di semester pertama.

---

## Rekomendasi Action Items

Berdasarkan analisis model Random Forest (akurasi 76%), fitur utama seperti Curricular_units_2nd_sem_approved (0.1503), Curricular_units_2nd_sem_grade (0.0983), dan Tuition_fees_up_to_date (0.0455) menunjukkan bahwa performa akademik semester kedua dan stabilitas keuangan adalah prediktor kunci status mahasiswa. Berikut adalah rekomendasi tindakan umum untuk meningkatkan retensi mahasiswa dan mengurangi risiko dropout:

1. Intervensi akademik di semester kedua\
Prioritaskan evaluasi performa mahasiswa di semester kedua sebagai periode kritis untuk mencegah dropout. Langkah-langkah yang bisa diambil adalah seperti membentuk tim akademik untuk memantau jumlah mata kuliah yang lulus dan nilai rata-rata setiap mahasiswa di minggu ke-6 hingga ke-8 semester kedua. Kemudian identifikasi mahasiswa dengan < 3 mata kuliah lulus atau nilai rata-rata < 10 agar dapat memberikan tindakan berupa sesi konseling individu untuk mengevaluasi kendala akademik. Selain itu, juga dapat menyediakan program bimbingan belajar tambahan seperti kelompok studi atau tutor dengan teman sebaya untuk mahasiswa berisiko. Jika langkah-langkah ini dilakukan dengan bijak dapat mengurangi tingkat dropout hingga 10-15% dengan intervensi dini di periode kritis.

2. Perkuat program dukungan keuangan\
Kembangkan inisiatif untuk membantu mahasiswa dengan masalah pembayaran biaya kuliah. Hal ini dapat dilakukan dengan membuat sistem untuk mendeteksi mahasiswa yang menunggak biaya kuliah di awal setiap semester melalui data administrasi. Kemudian tawarkan opsi pembayaran fleksibel, seperti cicilan bulanan tanpa bunga atau beasiswa jangka pendek untuk mahasiswa dengan kesulitan finansial. Hal lain yang dapat dilakukan juga adalah mengadakan seminar literasi keuangan untuk mengedukasi mahasiswa tentang pengelolaan biaya kuliah dan sumber bantuan. Apabila langkah ini dilakukan dengan bijak dapat mengurangi dropout akibat masalah finansial hingga 15-20%.

3. Intervensi awal di semester pertama\
Deteksi dini mahasiswa dengan performa akademik lemah di semester pertama untuk mencegah risiko jangka panjang. Hal ini dapat dilakukan dengan meninjau data Curricular_units_1st_sem_approved dan Curricular_units_1st_sem_grade di pertengan semester pertama (minggu ke-8). Untuk mahasiswa dengan < 3 mata kuliah lulus atau nilai rata-rata < 10, adakan workshop keterampilan belajar (misalnya, manajemen waktu, teknik mencatat). Selanjutnya, dapat menetapkan dosen wali untuk memberikan mentoring bulanan kepada mahasiswa berisiko. Apabila langkah ini dilakukan dengan benar maka dapat mencegah hingga 10% mahasiswa berisiko berpindah ke status dropout di semester kedua.

4. Sediakan dukungan khusus untuk mahasiswa berumur\
Tangani tantangan unik mahasiswa dengan age_at_enrollment > 30 yang memiliki risiko dropout lebih tinggi. Caranya, identifikasi mahasiswa berusia > 30 tahun melalui data pendaftaran di awal semester. Kemudian sediakan konseling karir dan akademik khusus untuk membantu mereka menyeimbangkan kuliah dengan tanggung jawab lain (misalnya, pekerjaan, keluarga). Selanjutnya, tawarkan opsi jadwal kuliah fleksibel seperti kelas malam atau pembelajaran daring untuk meningkatkan aksesibilitas. Apabila langkah ini dilakukan dengan baik maka dapat meningkatkan retensi mahasiswa dewasa hingga 8-10%.

5. Sesuaikan kurikulum berdasarkan program studi\
Tinjau ulang program studi (Course) dengan tingkat dropout tinggi untuk menyesuaikan kurikulum atau dukungan akademik. Analisis data mahasiswa per program studi untuk mengidentifikasi pola dropout (misalnya, program dengan beban akademik tinggi). Untuk program berisiko tinggi, kurangi beban mata kuliah wajib di semester kedua atau tambahkan sesi orientasi khusus. Kemudian dapat melibatkan dosen untuk merancang ulang metode pengajaran agar lebih interaktif dan mendukung keterlibatan mahasiswa. Jika langkah ini diwujudkan segera dengan bijak maka dapat mengurangi dropout di program tertentu hingga 5-10%
"""