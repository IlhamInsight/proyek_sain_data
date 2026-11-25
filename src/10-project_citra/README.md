# Klasifikasi Kerapian Alas Kaki (Sandal & Sepatu)

Pipeline menggunakan dataset citra yang berada di `src/10-project_citra/2-data_understanding/data/RAPI` dan `src/10-project_citra/2-data_understanding/data/BERANTAKAN`.

Jalankan seluruh tahapan dari folder `10-project_citra`:

```powershell
python run_pipeline.py
```

## Ringkasan Metodologi Sains Data (Anti-Overfitting & Tanpa Kebocoran Data)

1. **Ekstraksi 75 Fitur Visual Invarian Lantai & Pencahayaan:**
   * **Keragaman Warna Klasik (13 Fitur):** Standar deviasi kanal RGB/HSV, warna oponen ($R-G$, $Y-B$), dan keragaman warna lokal 4x4 (mengabaikan nilai rata-rata RGB global agar model tidak menghafal warna lantai masing-masing kelompok).
   * **Keteraturan Orientasi & Tensor Struktur (12 Fitur):** Entropi dan koefisien Gini histogram gradien Sobel multi-skala, rasio tepi ortogonal vs diagonal, serta koherensi anisotropi struktur.
   * **Sebaran Tepi Spasial & Spektrum 2D FFT (16 Fitur):** Kepadatan tepi grid 4x4/8x8, energi 6 pita frekuensi radial, dan konsentrasi sudut (*FFT Angular Entropy & Gini*).
   * **Radon Rotated Alignment, Profil Proyeksi, & Simetri (17 Fitur):** Keteraturan proyeksi garis multi-sudut (*rotation-invariant*), proyeksi baris/kolom, kesimetrian cermin, dan rasio pusat-vs-tepi.
   * **Geometri Objek *Foreground* & Tekstur LBP/GLCM (17 Fitur):** Jarak radial titik tepi (`fg_dist_mean`, `fg_dist_p75`, `fg_dist_iqr`), elongasi PCA (`fg_elongation`), *Local Binary Patterns*, dan GLCM 4 sudut.
2. **Pembagian Data Berbasis Kelompok (*Group-Independent Split*):**
   * **Pengujian (`Unseen Test Set`):** `Kelompok 2` (15 citra Rapi + 15 citra Berantakan = 30 citra).
   * **Pengembangan (`Development Set`):** Seluruh kelompok lainnya (120 citra unik setelah deduplikasi dan 4 koreksi label terverifikasi).
3. **Pelatihan & Kontrol *Overfitting* (0% Kebocoran):**
   * Seleksi model menggunakan **5-Fold `StratifiedGroupKFold`** pada data pengembangan berdasarkan **Macro-F1** serta audit *Train-Val Gap*.
   * `PowerTransformer` (Yeo-Johnson) dan augmentasi pencerminan horizontal (*horizontal flip*) diterapkan **hanya di dalam fold latih**.
   * Kandidat model terregularisasi yang dibandingkan: `LogReg_PCA10`, `LogReg_PCA8`, `LogReg_KBest25`, `SVC_PCA10`, `SVC_KBest20`, `MLP_Regularized`, `LGBM_Regularized`, dan `RF_Regularized`.
4. **Hasil Terverifikasi:**
   * Model terpilih: **`LogReg_PCA10`** (`PowerTransformer + PCA(10) + LogisticRegression(C=0.1)`).
   * **Train Accuracy:** `85,0%` | **5-Fold CV Accuracy:** `77,1%` (**Macro-F1:** `74,3%`, **Train-Val Gap:** `7,5%`).
   * **Unseen Test Set (`Kelompok 2`):** **Accuracy:** `70,0%` | **Macro-F1:** `68,3%` | **ROC-AUC:** `0,733`.
