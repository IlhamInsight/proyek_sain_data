# Klasifikasi Kerapian Alas Kaki (Sandal & Sepatu)

Pipeline menggunakan dataset citra yang berada di `src/10-project_citra/2-data_understanding/data/RAPI` dan `src/10-project_citra/2-data_understanding/data/BERANTAKAN`.

Jalankan seluruh tahapan dari folder `10-project_citra`:

```powershell
python run_pipeline.py
```

## Ringkasan Metodologi Sains Data (Tanpa Kebocoran Data)

1. **Ekstraksi Fitur Klasik & Invarian Lantai (65 Fitur):**
   * **Keragaman Warna Klasik:** Standar deviasi kanal RGB dan HSV serta keragaman warna lokal 4x4 (mengabaikan nilai rata-rata RGB global agar model tidak overfitting terhadap warna lantai masing-masing kelompok).
   * **Keteraturan Orientasi & Koherensi Tensor Struktur:** Entropi dan koefisien Gini histogram gradien Sobel multi-skala, rasio tepi ortogonal vs diagonal, serta koherensi anisotropi struktur.
   * **Spektrum Frekuensi 2D FFT (Radial & Angular):** Menangkap keteraturan jarak baris/pasangan alas kaki dan orientasi paralel pada domain frekuensi.
   * **Distribusi Spasial, Profil Proyeksi, & Simetri:** Variansi/Gini proyeksi baris & kolom, kesimetrian cermin kiri-kanan & atas-bawah, rasio pusat vs tepi, serta sebaran komponen tepi *foreground*.
   * **Tekstur Lokal:** *Local Binary Patterns (LBP)* dan *Gray-Level Co-occurrence Matrix (GLCM)* dari 4 sudut ($0^\circ, 45^\circ, 90^\circ, 135^\circ$).
2. **Pembagian Data Berbasis Kelompok (*Group-Independent Split*):**
   * **Pengujian (`Unseen Test Set`):** `Kelompok 2` (15 citra Rapi + 15 citra Berantakan = 30 citra).
   * **Pengembangan (`Development Set`):** Seluruh kelompok lainnya (120 citra unik setelah deduplikasi dan 4 koreksi label terverifikasi).
3. **Pelatihan & Validasi Bebas Kebocoran:**
   * Seleksi model memakai **5-Fold `StratifiedGroupKFold`** pada data pengembangan berdasarkan **Macro-F1**.
   * `StandardScaler` dan augmentasi pencerminan horizontal (*horizontal flip*) diterapkan **hanya di dalam fold latih**, tidak pernah menyentuh data validasi atau test.
   * Model yang dibandingkan: `MLP_Ensemble`, `HybridVoting`, `LogisticRegression`, `LightGBM`, `SVC`, `ExtraTrees`, `RandomForest`, dan `GradientBoosting`.
4. **Evaluasi Jujur & Transparan:**
   * Artefak fitur disimpan di `3-data_preparation/data/processed`, model di `4-modeling/models`, salinan deployment di `6-deployment`, dan metrik evaluasi objektif di `5-evaluation/results`.
   * Target akurasi ($\ge 85\%$) dan Macro-F1 ($\ge 83\%$) adalah sasaran strategis jangka panjang; hasil uji dilaporkan apa adanya tanpa *data leakage* atau *oversampling* buatan pada data uji.
