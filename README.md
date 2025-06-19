# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya-Jaya Institut, berdiri sejak tahun 2000, adalah perguruan tinggi ternama yang telah mencetak lulusan berkualitas dengan reputasi unggul. Namun, tingkat dropout yang tinggi menjadi tantangan besar, memengaruhi reputasi institusi dan keberlanjutan finansial. Dengan dataset performa mahasiswa yang tersedia, institusi bertujuan memanfaatkan machine learning untuk mendeteksi dini mahasiswa berisiko dropout, memungkinkan intervensi seperti bimbingan akademik atau dukungan keuangan. Proyek ini juga mencakup pembuatan visualisasi data untuk membantu manajemen memahami pola dropout dan memonitor performa mahasiswa secara efektif.

Tujuan Bisnis:
- Mengurangi tingkat dropout melalui identifikasi dini mahasiswa berisiko.
- Mengoptimalkan alokasi sumber daya untuk intervensi akademik dan keuangan.
- Meningkatkan retensi dan tingkat kelulusan untuk mempertahankan reputasi institusi.

### Permasalahan Bisnis
1. Tingkat Dropout Tinggi: Sejumlah mahasiswa tidak menyelesaikan studi, terutama di semester awal, menyebabkan kerugian reputasi dan pendapatan.
2. Keterlambatan Identifikasi Risiko: Kurangnya sistem untuk mendeteksi mahasiswa berisiko dropout secara dini menghambat intervensi tepat waktu.
3. Alokasi Sumber Daya Tidak Efisien: Tanpa wawasan data, institusi kesulitan menentukan prioritas bimbingan atau bantuan keuangan.
4. Variasi Performa Akademik: Performa mahasiswa di semester pertama dan kedua bervariasi, memerlukan strategi spesifik untuk mendukung keberhasilan.
5. Kendala Keuangan Mahasiswa: Tunggakan biaya kuliah berkontribusi signifikan terhadap dropout, membutuhkan solusi keuangan yang efektif.

### Cakupan Proyek
Proyek ini mencakup:
1. Membangun Model Prediksi: Mengembangkan model Random Forest Classifier untuk memprediksi status mahasiswa (Dropout, Enrolled, Graduate).
2. Menganalisis Faktor Utama: Menentukan 23 fitur kunci berdasarkan tingkat kepentingan (>0.01), seperti jumlah mata kuliah lulus di semester kedua (0.1503), nilai rata-rata semester kedua (0.0983), dan status pembayaran biaya kuliah (0.0455).
3. Memberikan Rekomendasi Bisnis: Menyusun saran intervensi berdasarkan performa akademik di semester kedua dan stabilitas keuangan mahasiswa untuk mencegah dropout.
4. Menyediakan Visualisasi Data: Membuat grafik, seperti hubungan antara jumlah mata kuliah lulus dengan risiko dropout dan distribusi status mahasiswa, untuk mempermudah pemahaman pola risiko.
5. Hasil Proyek: Model machine learning, laporan performa model (F1-score: Dropout 0.79, Enrolled 0.46, Graduate 0.84), visualisasi data, serta rekomendasi tindakan praktis untuk institusi.

### Persiapan
- Sumber data: Dataset ini berisi informasi mengenai 4424 mahasiswa dengan 37 variabel yang mencakup berbagai aspek latar belakang mahasiswa, performa akademik, dan status pendidikan. Dataset [students performance](https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/students_performance/data.csv) berasal dari Dicoding.

- File:
    - `processed_numeric_dataset.csv`: data numerik untuk pelatihan model, memiliki 4424 entri data dan 23 fitur terpilih.
    - `processed_labeled_dataset.csv`: data dengan label (Dropout, Enrolled, Graduate) untuk keperluan visualisasi.
      
- Penjelasan fitur:\
Fitur Utama:
  - Akademik: `Curricular_units_2nd_sem_approved`, `Curricular_units_2nd_sem_grade`, `Curricular_units_1st_sem_approved`.
  - Keuangan: `Tuition_fees_up_to_date`, `Scholarship_holder`.
  - Demografi: `Age_at_enrollment`, `Course`.
  - Ekonomi: `GDP`, `Unemployment_rate`.

  Fitur lengkap dapat diakses melalui [dokumentasi fitur](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)

- ⚙️ Setup environment:
```bash
# Membuat environment baru dan mengaktifkannya
conda create --name student_dropout python=3.9
conda activate student_dropout

# Instalasi dependencies
pip install streamlit pandas scikit-learn joblib plotly
pip install -r requirements.txt
```

## Business Dashboard
Dashboard visualisasi dirancang untuk membantu Jaya-Jaya Institut memantau risiko dropout dan performa mahasiswa secara efektif. Berdasarkan analisis dataset, dashboard ini menyediakan wawasan kunci untuk mendukung pengambilan keputusan strategis.

Visualisasi Dashboard:\
![alt text](https://github.com/dysthymicfact/student-dropout-prediction/blob/main/images/student-dropout-1.png?raw=true)
![alt text](https://github.com/dysthymicfact/student-dropout-prediction/blob/main/images/student-dropout-2.png?raw=true)
* Fitur Visualisasi:
  - Grafik Batang: Dropout Rate per Program Studi
    Menampilkan persentase dropout berdasarkan program studi (Course). Program dengan dropout tertinggi meliputi Biofuel Technology (~66%), Equiniculture (~60%), dan Informatics Engineering (~55%), menunjukkan kebutuhan akan intervensi khusus pada program ini.
  - Grafik Batang: Segmentasi Dropout Berdasarkan Performa Akademik
    Menunjukkan rata-rata jumlah mata kuliah lulus di semester 1 dan 2 (Curricular_units_1st_sem_approved, Curricular_units_2nd_sem_approved) untuk setiap status mahasiswa. Mahasiswa dropout memiliki rata-rata ~2 mata kuliah lulus per semester, jauh lebih rendah dibandingkan graduate (~6 mata kuliah).

* Statistik Dropout:
  - Total Mahasiswa: 4,424
  - Dropout: 1,421 mahasiswa (32.12%)
  - Enrolled: 794 mahasiswa (17.96%)
  - Graduate: 2,209 mahasiswa (49.93%)

* Distribusi Dropout:\
  Berdasarkan program studi dan performa akademik (jumlah mata kuliah lulus di semester 1 dan 2). Visualisasi menggunakan grafik batang untuk kejelasan dalam mengidentifikasi pola risiko.

* Key Insights:
  - Program Studi Berisiko Tinggi: Biofuel Technology, Equiniculture, dan Informatics Engineering memiliki tingkat dropout di atas 50%, memerlukan penyesuaian kurikulum atau dukungan akademik tambahan.
  - Performa Akademik: Mahasiswa dropout rata-rata hanya lulus ~2 mata kuliah per semester, dibandingkan ~6 untuk graduate, menegaskan pentingnya pemantauan semester kedua (Curricular_units_2nd_sem_approved, importance 0.1503).
  - Status Keuangan: Mahasiswa dengan tunggakan biaya kuliah memiliki risiko dropout hingga 80% (Tuition_fees_up_to_date, importance 0.0455).

 ### Cara Akses Dashboard Metabase

   * Buka browser ke [http://localhost:3000](http://localhost:3000)
   * **Login:**
     * Email: `ryorikim06@gmail.com`
     * Password: `190525LaskarAi`
   * Pilih menu **Your personal collection** > **Dewi Rachmawati's Personal Collection** > **student-dropout** untuk melihat visualisasi.

## Menjalankan Sistem Machine Learning
Sistem prediksi status mahasiswa telah dibuat menggunakan model Random Forest Classifier dengan akurasi 76% dan diimplementasikan dalam bentuk aplikasi web interaktif menggunakan Streamlit. Sistem ini memungkinkan pengguna untuk memasukkan data mahasiswa dan mendapatkan prediksi status (Dropout, Enrolled, atau Graduate) secara langsung. Berdasarkan temuan model, berikut adalah 23 faktor teratas yang memengaruhi status mahasiswa:
![alt text](https://github.com/dysthymicfact/student-dropout-prediction/blob/main/images/fitur%20selection.png?raw=true)

### Langkah-langkah Menjalankan Aplikasi
1. Clone Repository
   ```bash
   git clone https://github.com/dysthymicfact/student-dropout-prediction.git
   cd student_dropout_prediction

2. Persiapan file
   Pastikan file berikut ada di direktori proyek:
   - app.py: kode aplikasi streamlit
   - dropout_model_rf_final.joblib: model random forest yang telah dilatih
   - label_encoder.joblib: label encoder untuk prediksi
    
3. Jalankan Aplikasi Streamlit
   ```bash
   streamlit run app.py
  Aplikasi akan berjalan secara lokal dan otomatis terbuka di browser pada alamat `http://localhost:8501`.
  Atau juga bisa diakses secara online melalui [klik disini]()

### Cara Menggunakan Sistem Machine Learning
1. Masukkan Infromasi Mahasiswa
   - Prestasi Akademik: Masukkan jumlah unit yang disetujui (`Curricular_units_2nd_sem_approved`, `Curricular_units_1st_sem_approved`), nilai rata-rata semester, evaluasi, dan unit diambil.
   - Demografi dan Keuangan: Masukkan usia saat pendaftaran, status biaya kuliah (`Tuition_fees_up_to_date`), status beasiswa, nilai penerimaan, dan program studi.
   - Latar Belakang Keluarga: Masukkan kualifikasi dan pekerjaan ibu/ayah.
   - Konteks Ekonomi dan Aplikasi: Masukkan tingkat pengangguran, inflasi, GDP, mode aplikasi, dan urutan aplikasi.
2. Dapatkan Prediksi
   - Klik tombol **Prediksi Status** untuk melihat hasil.
   - Aplikasi akan menampilkan:
       - Prediksi Status: Dropout, Enrolled, atau Graduate.
       - Probabilitas: Grafik batang menunjukkan probabilitas setiap status (misalnya, 70% Dropout, 20% Enrolled, 10% Graduate).
       - Kategori Risiko: Rendah (🟢, <30%), Sedang (🟠, 30–70%), atau Tinggi (🔴, >70%).
       - Faktor Risiko: Faktor utama seperti jumlah unit disetujui rendah atau biaya kuliah belum lunas.
       - Rekomendasi: Tindakan seperti konseling akademik, bantuan keuangan, atau dukungan kesehatan mental untuk risiko tinggi.
3. Interpretasi Hasil
   - Risiko Rendah (<30%): Mahasiswa menunjukkan performa akademik baik; pertahankan dengan pemantauan rutin.
   - Risiko Sedang (30–70%): Perlu tindakan pencegahan, seperti bimbingan belajar atau konseling akademik.
   - Risiko Tinggi (>70%): Membutuhkan intervensi segera, seperti dukungan akademik intensif dan bantuan keuangan.

Aplikasi ini memungkinkan staf akademik Jaya-Jaya Institut untuk mengidentifikasi mahasiswa berisiko dropout sejak dini sehingga tindakan pencegahan yang tepat dapat diambil untuk meningkatkan retensi mahasiswa.

## Conclusion
Dari analisis data mahasiswa dan pengembangan model prediktif, kami mengidentifikasi beberapa faktor kunci yang memengaruhi risiko dropout:
1). Performa Akademik Semester Kedua: Jumlah mata kuliah lulus (Curricular_units_2nd_sem_approved, importance 0.1503) dan nilai rata-rata adalah prediktor terkuat.
2). Status Keuangan: Mahasiswa dengan tunggakan biaya kuliah memiliki risiko dropout hingga 80% (Tuition_fees_up_to_date, importance 0.0455).
3). Program Studi: Program seperti Biofuel Technology (66%), Equiniculture (60%), dan Informatics Engineering (55%) memiliki tingkat dropout tinggi.

Proyek ini menghasilkan model Random Forest dengan akurasi 76%, efektif untuk memprediksi Dropout (F1-score 0.79) dan Graduate (0.84), tetapi kurang optimal untuk Enrolled (0.46) karena data imbalance. Aplikasi Streamlit dan dashboard visualisasi mendukung identifikasi pola risiko, dan rekomendasi tindakan dapat mengurangi dropout hingga 15-20%. Untuk perbaikan, model dapat ditingkatkan pada kelas Enrolled, dan data diperbarui rutin untuk relevansi jangka panjang.

## Recommendations Action Items
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

