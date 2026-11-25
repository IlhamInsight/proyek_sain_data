# Business Understanding
# Proyek Sains Data: Klasifikasi Citra Kerapian Sandal dan Sepatu (Alas Kaki)
## Dokumen Pemahaman Bisnis (CRISP-DM Tahap 1: Business Understanding)

---

### 1. Latar Belakang & Konteks Masalah
Penataan alas kaki (sandal dan sepatu) yang tertib dan teratur merupakan indikator penting dari kedisiplinan, keteraturan lingkungan, estetika ruang, serta standar higienitas. Kebutuhan akan keteraturan alas kaki ini sangat krusial di berbagai lokasi publik dan institusi, antara lain:
* **Tempat Ibadah (Masjid / Musholla):** Penumpukan sandal jamaah yang berserakan di depan pintu masuk sering menghambat mobilitas arus keluar-masuk jamaah saat jam salat berjamaah atau salat Jumat.
* **Institusi Pendidikan & Sekolah:** Menjaga kerapian rak alas kaki di depan kelas atau laboratorium komputer untuk melatih kedisiplinan siswa.
* **Gedung Perkantoran & Fasilitas Publik:** Menjaga estetika lobi dan lorong agar tetap rapi, nyaman dipandang, dan mencegah risiko tersandung atau kehilangan alas kaki.

Selama ini, proses pemantauan kerapian alas kaki dilakukan secara **konvensional / manual** oleh petugas kebersihan (*cleaning service*) atau staf operasional melalui patroli fisik berkala. Pendekatan manual ini memiliki sejumlah kelemahan mendasar:
1. **Tidak Waktu-Nyata (*Delayed / Reactive*):** Ketidakteraturan biasanya terjadi mendadak pada jam-jam sibuk (*peak hours*) dan baru diketahui setelah kondisi lantai atau rak sudah sangat berantakan.
2. **Pemborosan Biaya & Waktu Operasional:** Petugas harus berulang kali berkeliling hanya untuk mengecek apakah rak alas kaki sudah penuh dan berantakan.
3. **Subjektivitas Penilaian:** Penilaian kondisi "rapi" atau "berantakan" bersifat kualitatif dan berbeda-beda antar-petugas tanpa adanya standar numerik terukur.

Pemanfaatan teknologi **Computer Vision (Visi Komputer)** dan **Sains Data** memungkinkan pembuatan sistem pengawasan otomatis berbasis kamera CCTV atau kamera pintar (*Smart Camera*) yang mampu mendeteksi dan mengklasifikasikan status kerapian alas kaki secara instan dan objektif.

---

### 2. Tujuan Bisnis (*Business Objectives*)
1. **Otomatisasi Pemantauan Kerapian Fasilitas:** Menghadirkan sistem berbasis AI yang mampu menganalisis citra visual area rak/lantai alas kaki secara otomatis tanpa memerlukan inspeksi manual terus-menerus.
2. **Efisiensi Alokasi Tenaga Kerja Petugas:** Mengubah pola kerja dari *inspeksi rutin keliling* menjadi *intervensi berbasis kebutuhan (on-demand alert)*. Petugas hanya diarahkan menuju lokasi ketika sistem mendeteksi kondisi "Berantakan".
3. **Peningkatan Standar Kenyamanan & Keselamatan Lingkungan:** Meminimalkan hambatan evakuasi darurat pada pintu keluar-masuk serta menjaga citra kerapian institusi.

---

### 3. Sasaran Proyek Sains Data (*Data Science Objectives*)

#### 1. Definisi Tugas Klasifikasi Biner (*Binary Classification Task*)
Model sains data dirancang untuk mengklasifikasikan citra alas kaki ke dalam **2 kelas diskrit**:
* **Kelas 1 — RAPI (`Label 1`):** Citra yang memperlihatkan sandal/sepatu tertata sejajar, berada di dalam rak/kompartemen secara teratur, berpasangan rapi, dan tidak menghalangi jalur jalan.
* **Kelas 0 — BERANTAKAN (`Label 0`):** Citra yang memperlihatkan sandal/sepatu berserakan di lantai, posisinya saling bersilangan, bertumpuk tidak beraturan, atau berada di luar kompartemen yang ditentukan.

#### 2. Metode Ekstraksi Fitur Visual Klasik (*Classical Feature Extraction*)
Sesuai instruksi dan panduan teknis proyek, pendekatan yang digunakan adalah **ekstraksi fitur klasik berbasis statistik warna dan spasial**. Pendekatan ini dipilih karena memiliki keunggulan komputasi yang sangat ringan (*low resource*), waktu inferensi sangat cepat ($< 20\text{ ms}$), dan dapat dijalankan pada komputer tanpa GPU (*Edge Computing*):
* **Statistik Warna Kanal RGB (8 Fitur):**
  Menghitung nilai Rata-rata (*Mean* $\mu$) dan Simpangan Baku (*Standard Deviation* $\sigma$) untuk setiap kanal warna (Red, Green, Blue) serta nilai agregat keseluruhan:
  $$\mu_c = \frac{1}{N} \sum_{i=1}^N I_c(i), \quad \sigma_c = \sqrt{\frac{1}{N} \sum_{i=1}^N (I_c(i) - \mu_c)^2}$$
* **Statistik Warna Ruang HSV (6 Fitur):**
  Menghitung Mean dan Std untuk kanal Hue ($H$), Saturation ($S$), dan Value ($V$) guna menangkap saturasi warna alas kaki terhadap latar belakang lantai/rak.
* **Statistik Intensitas Grayscale & Deteksi Tepi (*Edge Filter*) (4 Fitur):**
  Menghitung Mean dan Std pada citra skala abu-abu dan hasil filter deteksi tepi (*edge magnitude*). Citra berantakan umumnya menghasilkan kerapatan tepi yang lebih acak dan tersebar tidak teratur.
* **Statistik Spasial 4 Kuadran (8 Fitur):**
  Membagi citra menjadi 4 area (Top-Left, Top-Right, Bottom-Left, Bottom-Right) untuk mengukur distribusi penyebaran alas kaki secara spasial.
* **Histogram of Oriented Gradients / HOG Lokal (144 Fitur):**
  Membagi citra ke dalam grid $4 \times 4$ dengan 9 orientasi gradien per sel untuk menangkap pola orientasi geometri alas kaki (sejajar vs acak).

#### 3. Kriteria Integritas Data (*No-Leakage Group-Based Partition*)
Dataset dikumpulkan oleh beberapa kelompok mahasiswa/pengambil data. Untuk menjamin keabsahan evaluasi ilmiah:
* Pembagian data latih dan data uji dilakukan secara **Group-Independent Split** (Kelompok 2 dijadikan *Unseen Test Set*).
* Hal ini menjamin bahwa model diuji pada sudut pandang, pencahayaan, dan kamera yang benar-benar baru (*0% Data Leakage*).

#### 4. Kriteria Keberhasilan Teknis (*Technical Success Metrics*)
* **Akurasi Pengujian (*Test Accuracy*):** Target $\ge 70,00\%$ pada kelompok penutur/fotografer baru (*unseen group test set*).
* **Macro F1-Score & Weighted F1-Score:** Memastikan performa seimbang antara kelas `Rapi` dan `Berantakan` tanpa bias terhadap kelas mayoritas.
* **Inference Latency:** Target $< 100\text{ ms}$ per citra pada CPU standar untuk kelayakan integrasi kamera waktu-nyata.

---

### 4. Rencana Kerja Metodologi CRISP-DM

| Tahapan CRISP-DM | Rincian Aktivitas & Luaran | Direktori Kerja |
| :--- | :--- | :--- |
| **1. Business Understanding** | Perumusan masalah penataan alas kaki, penentuan 2 kelas target (`Rapi` vs `Berantakan`), sasaran bisnis, kriteria ekstraksi fitur klasik, dan arsitektur alur kerja. | `1-business_understanding/` |
| **2. Data Understanding** | Eksplorasi dataset citra asli, verifikasi integritas 2 folder kelas (`RAPI` dan `BERANTAKAN`), analisis resolusi/aspek rasio, dan ekstraksi tabel fitur awal `fitur_warna_klasik.csv`. | `2-data_understanding/` |
| **3. Data Preparation** | Penyelarasan orientasi citra (EXIF transpose), normalisasi ukuran kanvas ($224 \times 224$), pembersihan/deduplikasi data, ekstraksi fitur visual komprehensif, dan partisi bebas kebocoran (*Group-based Split*). | `3-data_preparation/` |
| **4. Modeling** | Pelatihan dan komparasi 5 algoritma machine learning klasik: Logistic Regression, Support Vector Classifier (SVC), Random Forest, Extra Trees, dan Gradient Boosting menggunakan Stratified Group Cross-Validation. | `4-modeling/` |
| **5. Evaluation** | Pengujian komprehensif pada subset uji independen (Kelompok 2): kalkulasi matriks konfusi ($2 \times 2$), classification report (Precision, Recall, F1), kurva ROC-AUC, dan verifikasi kriteria keberhasilan bisnis. | `5-evaluation/` |
| **6. Deployment** | Pengemasan model terpilih ke dalam layanan REST API berbasis FastAPI (Port `8016`), pembuatan dashboard web interaktif unggah foto, serta kontainerisasi Docker & docker-compose. | `6-deployment/` |

---

### 5. Arsitektur Deployment & Integrasi Sistem Produksi

```
[ Kamera Pengawas / Upload Citra Foto ]
                  │
                  ▼
   [ FastAPI REST API (Port 8016) ]
                  │
                  ├─► 1. EXIF Orientation & Padding (224x224)
                  ├─► 2. Ekstraksi Fitur Klasik (RGB, HSV, Grayscale, Tepi, Kuadran, HOG)
                  ├─► 3. Standarisasi Fitur (StandardScaler Latih)
                  └─► 4. Inferensi Model Klasifikasi Terbaik
                  │
                  ▼
[ Output Respons JSON & Tampilan Web Interaktif ]
   • Status Kerapian: "RAPI" / "BERANTAKAN"
   • Skor Keyakinan (Confidence Score): misal 88.5%
   • Rekomendasi Tindakan Operasional untuk Petugas Kebersihan
```

Sistem menyediakan antarmuka web interaktif yang memungkinkan pengguna atau staf operasional untuk menguji citra secara langsung melalui peramban web (*browser*), menginspeksi nilai fitur warna yang diekstrak, serta melihat rekomendasi operasional secara waktu-nyata.
