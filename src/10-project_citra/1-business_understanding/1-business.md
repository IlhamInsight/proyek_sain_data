# 1. Business Understanding: Klasifikasi Citra Kerapian Sandal & Sepatu (Alas Kaki)

**Metodologi:** *Cross-Industry Standard Process for Data Mining* (**CRISP-DM**) — 6 Tahapan Terpadu

---

### 1. Latar Belakang & Konteks Permasalahan Bisnis
Penataan alas kaki (sandal dan sepatu) yang tertib merupakan cerminan kedisiplinan, kebersihan, estetika, serta keselamatan akses ruang pada fasilitas bersama. Kebutuhan pemantauan kerapian ini sangat vital pada berbagai lingkungan:
* **Tempat Ibadah (Masjid / Musholla):** Penumpukan sandal jamaah yang berserakan di pintu masuk menghambat kelancaran arus keluar-masuk saat waktu salat berjamaah.
* **Sekolah, Pesantren, & Kampus:** Menjaga keteraturan rak sepatu di depan ruang kelas atau laboratorium untuk membangun budaya disiplin.
* **Perkantoran, Klinik, & Area Publik:** Memastikan area resepsionis atau loker tetap tertata rapi dan bebas dari risiko tersandung.

Selama ini, pemeriksaan kondisi alas kaki dilakukan **secara manual** melalui patroli fisik petugas kebersihan (*cleaning service*). Metode manual ini memiliki tiga keterbatasan mendasar:
1. **Tidak Waktu-Nyata (*Delayed Response*):** Area alas kaki sering kali langsung berantakan pada jam sibuk (*peak hours*) tanpa terdeteksi seketika oleh petugas.
2. **Inefisiensi Tenaga Kerja:** Petugas harus berulang kali berkeliling hanya untuk memeriksa kondisi area yang mungkin masih rapi.
3. **Subjektivitas Standar:** Penilaian "rapi" atau "berantakan" bergantung pada persepsi masing-masing petugas tanpa ukuran kuantitatif yang konsisten.

Dengan memanfaatkan **Computer Vision** dan **Machine Learning**, sistem dapat mengevaluasi citra visual dari kamera atau unggahan foto secara otomatis, cepat, dan objektif.

---

### 2. Tujuan Bisnis (*Business Objectives*)
1. **Otomasi Deteksi Kerapian:** Membangun sistem klasifikasi citra otomatis untuk membedakan status alas kaki **`Rapi`** dan **`Berantakan`**.
2. **Optimalisasi Tugas Petugas Operasional (*On-Demand Alert*):** Mengalihkan inspeksi keliling rutin menjadi sistem peringatan berbasis kondisi aktual (*hanya bertindak saat status terdeteksi Berantakan*).
3. **Standarisasi Keteraturan Lingkungan:** Mendukung tata kelola fasilitas publik agar selalu tertib, higienis, dan aman dilalui.

---

### 3. Sasaran Teknis Sains Data (*Data Science Objectives*)

#### A. Definisi 2 Kelas Target & Lokasi Dataset Terkini
Seluruh citra dikelola secara terpusat di dalam folder **`src/10-project_citra/2-data_understanding/data/`** yang terbagi ke dalam dua direktori kelas:
* **Kelas 1 — `RAPI`:** Alas kaki tertata sejajar, berpasangan teratur, atau tersusun rapi di dalam rak/kompartemen.
* **Kelas 0 — `BERANTAKAN`:** Alas kaki berserakan di lantai, posisinya saling bersilangan, bertumpuk acak, atau keluar dari area tertib.

#### B. Audit Kelemahan Baseline Awal & Strategi Peningkatan Akurasi (Bebas *Data Leakage*)
Pada eksperimen baseline awal (hanya menggunakan rata-rata RGB global dan HOG grid statis), model *Logistic Regression* memperoleh **Akurasi Test 60,0% (Macro-F1: 52,4%)**. Analisis ilmiah menunjukkan dua penyebab utama rendahnya performa awal:
1. **Bias Warna Lantai (*Domain Shift* Antar-Kelompok):** Setiap kelompok pengambil data memotret lokasi yang berbeda (ubin putih, karpet biru, kayu cokelat). Akibatnya, nilai rata-rata RGB global (`mean_red`, `mean_green`, `mean_blue`) justru membuat model menghafal warna lantai masing-masing kelompok, bukan kerapian sepatunya.
2. **Artefak Padding Abu-Abu:** Teknik `ImageOps.pad` dengan warna abu-abu solid `(127, 127, 127)` menciptakan garis tepi buatan yang tajam di pinggir foto potret maupun lanskap.

Untuk **meningkatkan akurasi secara ilmiah tanpa rekayasa curang (*no data leakage / tanpa menyuntik data uji*)**, proyek ini menerapkan 4 pembaruan standar sains data:
1. **Ekstraksi 65 Fitur Invarian Lantai & Pencahayaan:**
   * **Keragaman Warna Klasik (13 Fitur):** Tetap mempertahankan Standar Deviasi RGB/HSV (`std_red`, `std_green`, `std_blue`, `std_overall`, `hue_std`, `sat_std`, `val_std`) serta variansi warna oponen ($R-G$, $Y-B$) yang mengukur keberagaman warna objek tanpa terikat warna dasar lantai. *(Catatan: Nilai Mean RGB klasik tetap diekstrak dan disimpan ke `fitur_warna_klasik.csv` pada Tahap 2 untuk audit deskriptif).*
   * **Keteraturan Orientasi & Tensor Struktur (11 Fitur):** Menghitung entropi dan koefisien Gini histogram gradien Sobel multi-skala, rasio tepi ortogonal ($0^\circ/90^\circ$) vs diagonal ($45^\circ/135^\circ$), serta koherensi tensor struktur. Sepatu yang rapi cenderung sejajar (*entropi orientasi rendah, koherensi tinggi*).
   * **Spektrum Frekuensi 2D FFT Radial & Angular (9 Fitur):** Menangkap pola periodik barisan alas kaki dan konsentrasi sudut pada domain frekuensi.
   * **Sebaran Spasial Tepi & Profil Proyeksi (19 Fitur):** Mengukur keragaman kepadatan tepi pada grid $4 	imes 4$ dan $8 	imes 8$, profil proyeksi baris/kolom, kesimetrian cermin kiri-kanan dan atas-bawah, rasio pusat vs tepi, serta sebaran komponen *foreground*.
   * **Tekstur Lokal LBP & GLCM (8 Fitur):** *Local Binary Patterns* dan *Gray-Level Co-occurrence Matrix* dari 4 arah sudut ($0^\circ, 45^\circ, 90^\circ, 135^\circ$).
2. **Pemotongan Duplikat & Koreksi Label Terverifikasi:** Menghapus 3 berkas duplikat identik (berbasis hash SHA-256) dan menerapkan 4 koreksi label terverifikasi sehingga tersisa **150 citra unik murni**.
3. **Partisi Berbasis Kelompok (*Group-Independent Split*):** Memisahkan **`Kelompok 2`** (15 Rapi + 15 Berantakan = **30 citra**) sebagai **Unseen Test Set** murni, sementara 6 kelompok lain (**120 citra unik**) menjadi **Development Set**.
4. **Augmentasi Pencerminan Horizontal Khusus Data Latih (*Train-Only Augmentation*):** Pencerminan horizontal (*mirror*) mempertahankan status kerapian secara sempurna (citra rapi dicerminkan tetap rapi). Augmentasi ini diterapkan **hanya di dalam fold latih** pada *5-Fold StratifiedGroupKFold* ($120 	o 240$ sampel pengembangan), tanpa pernah menyentuh data validasi maupun data uji.

#### C. Kriteria & Target Keberhasilan
* **Metrik Utama Seleksi Model:** *Macro F1-Score* dan *Accuracy* pada *5-Fold StratifiedGroupKFold* (data pengembangan).
* **Target Jangka Panjang (*Strategic Benchmark*):** Akurasi $\ge 85,00\%$ dan Macro-F1 $\ge 83,00\%$ (sebagai tolok ukur pengembangan lanjutan ketika dataset diperluas $>500$ citra lintas lokasi menggunakan *Deep Learning / CNN*).
* **Pelaporan Objektif:** Jika pada 150 citra klasik saat ini target jangka panjang belum tercapai, hasil evaluasi dilaporkan apa adanya secara jujur dan transparan.
* **Latensi Inferensi Produksi:** $< 100	ext{ ms}$ per citra pada CPU standar.

---

### 4. Alur Pengolahan 6 Tahapan Metodologi CRISP-DM

```
[1. Business Understanding] ──► Perumusan masalah, 2 kelas (RAPI/BERANTAKAN), & strategi 65 fitur invarian
           │
           ▼
[2. Data Understanding]     ──► Audit 153 file di 2-data_understanding/data/, ekstraksi fitur_warna_klasik.csv
           │
           ▼
[3. Data Preparation]       ──► Deduplikasi hash (150 unik), koreksi label, ekstraksi 65 fitur, & Group Split
           │
           ▼
[4. Modeling]               ──► Komparasi 8 model (LightGBM, MLP_Ensemble, HybridVoting, LogReg, SVC, dll)
           │                    dengan 5-Fold StratifiedGroupKFold + Augmentasi Horizontal khusus Fold Latih
           ▼
[5. Evaluation]             ──► Pengujian objektif pada Kelompok 2 (30 citra unseen): Accuracy, Macro-F1, ROC-AUC
           │
           ▼
[6. Deployment]             ──► Microservice FastAPI (Port 8016), Web Dashboard Interaktif, & Docker Container
```

| Tahapan CRISP-DM | Direktori & Berkas Utama | Rincian Aktivitas & Luaran Terverifikasi |
| :--- | :--- | :--- |
| **1. Business Understanding** | `1-business_understanding/1-business.md` & `1-business.ipynb` | Penetapan sasaran bisnis, justifikasi pemindahan data, strategi fitur klasik invarian lantai, dan kriteria evaluasi jujur. |
| **2. Data Understanding** | `2-data_understanding/1-data_understanding.ipynb` | Memuat 153 citra (14 folder kelompok) dari `2-data_understanding/data/`, visualisasi sampel, dan ekspor `fitur_warna_klasik.csv`. |
| **3. Data Preparation** | `3-data_preparation/1-data_preparation.ipynb` | Koreksi orientasi EXIF, resize $224 	imes 224$ bebas padding abu-abu, deduplikasi ke 150 citra unik, ekstraksi 65 fitur visual + augmentasi flip latih, dan pemisahan Kelompok 2 sebagai Test Set. |
| **4. Modeling** | `4-modeling/1-modeling.ipynb` | Komparasi 8 algoritma pada *5-Fold StratifiedGroupKFold*; `StandardScaler` & augmentasi di-fit murni di dalam fold latih; menghasilkan `best_model.joblib` & `model_comparison.json`. |
| **5. Evaluation** | `5-evaluation/1-evaluation.ipynb` | Evaluasi objektif pada 30 citra uji Kelompok 2, menghasilkan `evaluation_metrics.json`, *Confusion Matrix*, kurva ROC-AUC, dan analisis *Feature Importance*. |
| **6. Deployment** | `6-deployment/1-deployment.ipynb` & `app.py` | Layanan REST API FastAPI pada **Port `8016`** (`/api/v1/predict`, `/health`), antarmuka web unggah foto, dan kontainerisasi Docker. |

---

### 5. Arsitektur Layanan Deployment (Microservice FastAPI)

* **Endpoint Web & API (`Port 8016`):**
  * `GET /`: Antarmuka web interaktif unggah foto alas kaki (*Dark Theme*).
  * `POST /api/v1/predict`: Menerima unggahan citra (JPEG/PNG), mengekstrak 65 fitur visual secara konsisten dengan mempertahankan orientasi EXIF, menerapkan `StandardScaler`, dan mengembalikan kelas prediksi (`Rapi` / `Tidak Rapi (Berantakan)`), skor keyakinan, probabilitas per kelas, serta latensi proses dalam milidetik.
  * `GET /health`: Pemeriksaan kesiapan model dan scaler di memori server.
