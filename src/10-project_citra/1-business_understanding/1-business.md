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

#### B. Evolusi Ilmiah Model: Menghilangkan *Overfitting* & Meningkatkan Generalisasi
Melalui iterasi metodologi sains data yang ketat (**0% *Data Leakage* / tanpa menyuntik data uji**), sistem mengalami 3 fase peningkatan terukur:
1. **Baseline Awal (`LogisticRegression` + RGB/HOG statis):** Mengalami bias warna lantai tiap kelompok dan artefak padding abu-abu $\rightarrow$ **Test Acc 60,0% (Macro-F1: 52,4%)**.
2. **Iterasi Fitur Visual Tanpa Regularisasi Ketat (`LGBMClassifier` + 65 fitur):** Meningkatkan Test Acc ke **66,7% (Macro-F1: 66,5%)**, namun mengalami *overfitting* karena pohon keputusan menghafal **100,0%** data latih (*Train-Test Gap = 33,3%*).
3. **Pipeline Final Anti-Overfitting (`PowerTransformer + PCA(10) + LogisticRegression(C=0.1)` pada 75 Fitur):**
   * **Ekstraksi 75 Fitur Visual Invarian Lantai & Pencahayaan:** Menggabungkan keragaman warna klasik (Std RGB/HSV), koherensi tensor struktur, Radon *rotated alignment* (tahan rotasi), spektrum 2D FFT (radial & angular), kesimetrian pusat/global, sebaran geometri objek *foreground* (`fg_dist_p75`, `fg_spread_diag`, `fg_elongation`), serta tekstur lokal LBP & GLCM. *(Nilai Mean RGB klasik tetap disimpan di `fitur_warna_klasik.csv` pada Tahap 2 untuk audit deskriptif).*
   * **Transformasi Non-Linier & Reduksi Dimensi Anti-Overfitting:** `PowerTransformer` (Yeo-Johnson) menormalkan fitur yang miring (*skewed*), lalu **PCA (10 komponen utama)** membuang variansi noise sisa dan multikolinearitas, sebelum dimodelkan oleh `LogisticRegression` dengan penalti L2 kuat (`C=0.1`).
   * **Hasil Terverifikasi & Bebas Overfitting:** **Train Acc = 85,0%**, **5-Fold CV Acc = 77,1% (Macro-F1: 74,3%, Gap hanya 7,5%)**, serta **Unseen Test Acc = 70,0% (Macro-F1: 68,3%, ROC-AUC: 0,733)** pada `Kelompok 2`.

#### C. Kriteria & Target Keberhasilan
* **Keseimbangan Train-Val-Test (Anti-Overfitting):** Menjaga *Train-Val Gap* $\le 10\%$ sehingga model tidak sekadar menghafal data latih.
* **Target Jangka Panjang (*Strategic Benchmark*):** Akurasi $\ge 85,00\%$ dan Macro-F1 $\ge 83,00\%$ (sebagai tolok ukur pengembangan lanjutan ketika dataset diperluas $>500$ citra lintas lokasi menggunakan *Deep Learning / CNN*).
* **Latensi Inferensi Produksi:** $< 100	ext{ ms}$ hingga ~250 ms pada CPU standar.

---

### 4. Alur Pengolahan 6 Tahapan Metodologi CRISP-DM

```
[1. Business Understanding] ──► Perumusan masalah, 2 kelas (RAPI/BERANTAKAN), & arsitektur anti-overfitting
           │
           ▼
[2. Data Understanding]     ──► Audit 153 file di 2-data_understanding/data/, ekspor fitur_warna_klasik.csv
           │
           ▼
[3. Data Preparation]       ──► Deduplikasi hash (150 unik), koreksi label, ekstraksi 75 fitur, & Group Split
           │
           ▼
[4. Modeling]               ──► Komparasi 8 model terregularisasi pada 5-Fold StratifiedGroupKFold
           │                    dengan PowerTransformer + Augmentasi Horizontal khusus Fold Latih
           ▼
[5. Evaluation]             ──► Pengujian objektif pada Kelompok 2 (30 citra unseen): Accuracy, Macro-F1, Gap
           │
           ▼
[6. Deployment]             ──► Microservice FastAPI (Port 8016), Web Dashboard Interaktif, & Docker Container
```

| Tahapan CRISP-DM | Direktori & Berkas Utama | Rincian Aktivitas & Luaran Terverifikasi |
| :--- | :--- | :--- |
| **1. Business Understanding** | `1-business_understanding/1-business.md` & `1-business.ipynb` | Penetapan sasaran bisnis, justifikasi pemindahan data, strategi anti-overfitting, dan kriteria evaluasi jujur. |
| **2. Data Understanding** | `2-data_understanding/1-data_understanding.ipynb` | Memuat 153 citra (14 folder kelompok) dari `2-data_understanding/data/`, visualisasi sampel, dan ekspor `fitur_warna_klasik.csv`. |
| **3. Data Preparation** | `3-data_preparation/1-data_preparation.ipynb` | Koreksi orientasi EXIF, resize $224 \times 224$ bebas padding abu-abu, deduplikasi ke 150 citra unik, ekstraksi 75 fitur visual + augmentasi flip latih, dan pemisahan Kelompok 2 sebagai Test Set. |
| **4. Modeling** | `4-modeling/1-modeling.ipynb` | Komparasi 8 model terregularisasi pada *5-Fold StratifiedGroupKFold*; `PowerTransformer` & augmentasi di-fit murni di dalam fold latih; memantau `train_val_gap`. |
| **5. Evaluation** | `5-evaluation/1-evaluation.ipynb` | Evaluasi objektif pada 30 citra uji Kelompok 2, audit `Train Acc` vs `Test Acc` (*Gap 15,0%*), *Confusion Matrix*, kurva ROC-AUC, dan *Feature Importance*. |
| **6. Deployment** | `6-deployment/1-deployment.ipynb` & `app.py` | Layanan REST API FastAPI pada **Port `8016`** (`/api/v1/predict`, `/health`), antarmuka web unggah foto, dan kontainerisasi Docker. |

---

### 5. Arsitektur Layanan Deployment (Microservice FastAPI)

* **Endpoint Web & API (`Port 8016`):**
  * `GET /`: Antarmuka web interaktif unggah foto alas kaki (*Dark Theme*).
  * `POST /api/v1/predict`: Menerima unggahan citra (JPEG/PNG), mengekstrak 75 fitur visual secara konsisten dengan mempertahankan orientasi EXIF, menerapkan `PowerTransformer` (`scaler.joblib`), melakukan proyeksi PCA 10-komponen dan klasifikasi `LogisticRegression`, serta mengembalikan respons JSON lengkap.
  * `GET /health`: Pemeriksaan kesiapan model dan scaler di memori server.
