# -*- coding: utf-8 -*-
"""
Generator script for 6-deployment.ipynb
Adheres strictly to CRISP-DM 
Deployment phase requirements.
"""

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Markdown Title & Introduction
cells.append(nbf.v4.new_markdown_cell("""# 6. Deployment: Implementasi dan Diseminasi Sistem Geospasial Tutupan Lahan Sentinel-2 Jawa Timur

**Mata Kuliah**: Proyek Sains Data | **Fase CRISP-DM**: Phase 6 - Deployment  
**Sensor Citra**: Satelit Multi-Spektral Copernicus Sentinel-2 Level-2A (Reflektansi Permukaan / BOA)  
**Wilayah Studi**: Jawa Timur (Scene Surabaya, Sidoarjo, Gresik, dsk.)  
**Model Utama**: LightGBM Pixel Classifier (`model_lightgbm_piksel.pkl`)  
**Dimensi Fitur**: 13 Fitur Spektral Pasti (7 Band BOA + 6 Indeks Biofisik Teruji)  
**Aplikasi Produksi**: Streamlit Interactive GIS Dashboard ([GitHub Repo](https://github.com/IlhamInsight/sentinel2-jatim-lulc-app))  

---

## 6.1 Tujuan dan Peran Tahap Deployment dalam CRISP-DM

Tahap **Deployment** merupakan fase pamungkas dalam metodologi **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*). Pada tahap ini, model prediktif terbaik yang telah dilatih pada Tahap 4 (Modeling) dan divalidasi secara ilmiah pada Tahap 5 (Evaluation) ditransformasikan menjadi **produk operasional yang siap pakai (*production-ready system*)**.

### Prinsip Pemisahan Dokumentasi & Aplikasi Interaktif:
1. **Jupyter Notebook (`6-deployment.ipynb`)**:
   - Berfungsi sebagai **dokumentasi teknis dan saintifik** tahap deployment pada buku laporan Jupyter Book / CRISP-DM.
   - Mendokumentasikan verifikasi model, arsitektur modul bersama (`shared/`), proses inferensi *full-raster* dari citra mentah Sentinel-2, pemotongan wilayah spasial (AOI windowing), penanganan ambang batas keyakinan (*confidence threshold*), dan validasi output GeoTIFF.
2. **Aplikasi Streamlit Interaktif (`app.py`)**:
   - Berfungsi sebagai **antarmuka pengguna interaktif (*interactive operational deployment*)** yang dapat diakses oleh praktisi GIS, akademisi, dan pembuat kebijakan.
   - Memungkinkan eksplorasi interaktif peta Leaflet, pengaturan ambang batas keyakinan, pemilihan wilayah spasial, serta inspeksi profil spektral per piksel secara *real-time*.
   - **Repositori Produksi Mandiri**: Sesuai standar arsitektur perangkat lunak, aplikasi Streamlit production dipisahkan ke repositori tersendiri di [https://github.com/IlhamInsight/sentinel2-jatim-lulc-app](https://github.com/IlhamInsight/sentinel2-jatim-lulc-app).
"""))

# Cell 2: Code - Imports & Environment
cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns

# Pastikan modul bersama dapat diimpor dari root Tugas_UTS
BASE_DIR = os.path.abspath('')
UTS_DIR = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == '6-deployment' else os.path.join(BASE_DIR, 'src', 'Tugas_UTS')
if UTS_DIR not in sys.path:
    sys.path.insert(0, UTS_DIR)

from shared.lulc_features import (
    FEATURE_COLS, BAND_NAMES, CLASS_CONFIG, VALID_CLASSES,
    REGIONS_AOI, compute_spectral_indices, extract_13_features, validate_model
)
from shared.lulc_inference import (
    check_raster_coverage, read_sentinel2_raster,
    run_full_raster_inference, save_inference_geotiffs
)

print('[OK] Modul geospasial dan dependensi tahap Deployment berhasil dimuat.')
"""))

# Cell 3: Markdown - 6.2 Model Validation
cells.append(nbf.v4.new_markdown_cell("""## 6.2 Pemuatan dan Validasi Integritas Model Final LightGBM

Sesuai metodologi CRISP-DM, model utama yang digunakan untuk inferensi raster geospasial resolusi tinggi 10 meter adalah **LightGBM Pixel Classifier** (`model_lightgbm_piksel.pkl`).

Sebelum model digunakan untuk inferensi seluruh piksel raster, dilakukan validasi ketat terhadap:
1. `model.classes_`: Memastikan model memiliki tepat 5 kelas target `[0, 1, 2, 3, 4]`.
2. `model.n_features_in_`: Memastikan model menerima tepat 13 fitur masukan tanpa perubahan skema.
"""))

# Cell 4: Code - Load & Validate Model
cells.append(nbf.v4.new_code_cell("""model_dir = os.path.join(UTS_DIR, '4-modeling', 'models')
model_path = os.path.join(model_dir, 'model_lightgbm_piksel.pkl')
meta_path = os.path.join(model_dir, 'metadata_lightgbm_piksel.json')

# Pemuatan artefak model
model = joblib.load(model_path)
validate_model(model)

with open(meta_path, 'r', encoding='utf-8') as f:
    model_metadata = json.load(f)

print('=' * 75)
print('STATUS VALIDASI INTEGRITAS MODEL FINAL (LIGHTGBM PIKSEL 10M):')
print(f'• File Model             : {os.path.basename(model_path)} ({os.path.getsize(model_path)/1024:.1f} KB)')
print(f'• Nama Model             : {model_metadata.get("nama_model")}')
print(f'• Kelas Model (classes_) : {list(model.classes_)} (Tepat 5 Kelas Target)')
print(f'• Jumlah Fitur Masukan   : {model.n_features_in_} Fitur Spektral Pasti')
print(f'• Daftar 13 Fitur Pasti  : {FEATURE_COLS}')
print('=' * 75)
"""))

# Cell 5: Markdown - 6.3 Raw Raster Input
cells.append(nbf.v4.new_markdown_cell("""## 6.3 Pemuatan Citra Geospasial Asli Sentinel-2 Menggunakan Rasterio

Sumber data inferensi adalah citra raster multi-spektral asli **Sentinel-2 Level-2A** (`sentinel2_jatim_scene.tif`), bukan raster klasifikasi lama.
Citra dibaca menggunakan pustaka standar geospasial **`rasterio`** guna mempertahankan metadata proyeksi, koordinat, transform affine, dan resolusi spasial asli.
"""))

# Cell 6: Code - Read Raster with Rasterio
cells.append(nbf.v4.new_code_cell("""raw_raster_path = os.path.join(UTS_DIR, '2-data_understanding', 'extracted_data', 'sentinel2_jatim_scene.tif')

with rasterio.open(raw_raster_path) as src:
    raster_crs = src.crs
    raster_bounds = src.bounds
    raster_transform = src.transform
    raster_shape = (src.count, src.height, src.width)
    raster_res = src.res

print('=' * 75)
print('METADATA RASTER CITRA SENTINEL-2 ASLI (sentinel2_jatim_scene.tif):')
print(f'• Driver Format          : GeoTIFF')
print(f'• Dimensi Piksel         : {raster_shape[2]} x {raster_shape[1]} piksel ({raster_shape[2]*raster_shape[1]:,} piksel total)')
print(f'• Jumlah Band Spektral   : {raster_shape[0]} Band (B02, B03, B04, B08, B8A, B11, B12)')
print(f'• Sistem Koordinat (CRS) : {raster_crs}')
print(f'• Bounding Box Geografis : {raster_bounds}')
print(f'• Resolusi Spasial Piksel: {raster_res[0]:.6f}° x {raster_res[1]:.6f}° (~10 meter)')
print('=' * 75)

# Visualisasi Komposit Warna Citra Asli (True Color vs False Color CIR)
bands_all, prof_all, bounds_all, trans_all = read_sentinel2_raster(raw_raster_path)

def norm_band(b):
    p2, p98 = np.nanpercentile(b, (2, 98))
    return np.clip((b - p2) / (p98 - p2 + 1e-6), 0, 1)

rgb_true = np.dstack([norm_band(bands_all[2]), norm_band(bands_all[1]), norm_band(bands_all[0])])
rgb_cir = np.dstack([norm_band(bands_all[3]), norm_band(bands_all[2]), norm_band(bands_all[1])])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.imshow(rgb_true)
ax1.set_title('Komposit True Color (B04-Red, B03-Green, B02-Blue)', fontsize=11, fontweight="bold")
ax1.axis('off')

ax2.imshow(rgb_cir)
ax2.set_title('Komposit False Color CIR (B08-NIR, B04-Red, B03-Green)', fontsize=11, fontweight="bold")
ax2.axis('off')
plt.tight_layout()
plt.show()
plt.close()
"""))

# Cell 7: Markdown - 6.4 Full-Raster Inference Pipeline
cells.append(nbf.v4.new_markdown_cell("""## 6.4 Pipeline Inferensi Full-Raster dan Penanganan Ambang Batas Keyakinan

Tahap inferensi mengeksekusi urutan pipeline ilmiah terpadu:
$$\\text{GeoTIFF (7 Band)} \\xrightarrow{\\text{6 Indeks}} \\text{13 Fitur Spektral} \\xrightarrow{\\text{Reshape } N \\times 13} \\text{LightGBM.predict\\_proba()} \\xrightarrow{\\text{Confidence Filtering}} \\text{Reshape } H \\times W$$

### Penanganan Ambang Batas Keyakinan (Confidence Threshold):
- Probabilitas maksimum tiap piksel dievaluasi: $\\max_c P(y = c \\mid \\mathbf{x})$.
- Jika $\\max_c P < \\tau$ (di mana $\\tau = 0.40$), piksel ditetapkan sebagai **Kelas 255** (*Unknown / Low Confidence / NoData*).
- Kode numerik kelas tutupan lahan:
  - `0`: Bangunan (*Built-up*)
  - `1`: Lahan Pertanian (*Farmland*)
  - `2`: Perairan (*Water Body*)
  - `3`: Hutan Daratan (*Terrestrial Forest*)
  - `4`: Hutan Mangrove (*Mangrove Forest*)
  - `255`: Unknown / Low Confidence (bukan kelas LULC keenam).
"""))

# Cell 8: Code - Execute Full-Raster Inference
cells.append(nbf.v4.new_code_cell("""# Eksekusi inferensi penuh pada 1,048,576 piksel
t_start = time.time()
infer_result = run_full_raster_inference(model, bands_all, confidence_threshold=0.40)
t_elapsed = time.time() - t_start

classified_raster = infer_result['classified_raster']
confidence_raster = infer_result['confidence_raster']
stats = infer_result['class_stats']

print(f'[SUKSES] Inferensi Full-Raster selesai dalam {t_elapsed:.2f} detik.')
print(f'• Total Piksel Diproses : {infer_result["total_pixels"]:,} piksel')
print(f'• Resolusi Grid Keluaran: {classified_raster.shape[0]} baris x {classified_raster.shape[1]} kolom')
print(f'• Nilai Unik Piksel     : {np.unique(classified_raster).tolist()} (0..4, 255)')
"""))

# Cell 9: Markdown - 6.5 Distribution & Maps
cells.append(nbf.v4.new_markdown_cell("""## 6.5 Analisis Distribusi Tutupan Lahan & Visualisasi Peta Spasial

Hasil prediksi model disajikan melalui visualisasi spasial peta tematik klasifikasi LULC dan peta distribusi keyakinan (*confidence map*), dilengkapi dengan tabel proporsi luas tutupan lahan.
"""))

# Cell 10: Code - Plot Results
cells.append(nbf.v4.new_code_cell("""# 1. Tabel Distribusi 5 Kelas LULC
dist_records = []
for c in [0, 1, 2, 3, 4, 255]:
    st_c = stats[c]
    dist_records.append({
        'Kode': c,
        'Nama Kelas': st_c['name'],
        'Alias Internasional': st_c['alias'],
        'Jumlah Piksel': f"{st_c['count']:,}",
        'Proporsi Luas (%)': f"{st_c['percent']:.2f}%"
    })

dist_df = pd.DataFrame(dist_records)
print('DISTRIBUSI PREDIKSI TUTUPAN LAHAN (FULL SCENE SENTINEL-2):')
display(dist_df)

# 2. Visualisasi Peta Klasifikasi & Peta Keyakinan
cmap_colors = ['#EF4444', '#D97706', '#0284C7', '#16A34A', '#0D9488', '#000000']
cmap = ListedColormap(cmap_colors)
bounds_bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 255.5]
norm = BoundaryNorm(bounds_bins, cmap.N)

fig, (ax_lulc, ax_conf) = plt.subplots(1, 2, figsize=(16, 7))

# Peta LULC
im_lulc = ax_lulc.imshow(classified_raster, cmap=cmap, norm=norm)
ax_lulc.set_title('Peta Klasifikasi Tutupan Lahan Sentinel-2 (LightGBM 10m)', fontsize=12, fontweight="bold", pad=10)
ax_lulc.axis('off')

# Legenda LULC
cbar = fig.colorbar(im_lulc, ax=ax_lulc, fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3, 4, 255])
cbar.ax.set_yticklabels(['Bangunan', 'Pertanian', 'Perairan', 'Hutan', 'Mangrove', 'Low Conf / NoData'])

# Peta Keyakinan (Confidence)
im_conf = ax_conf.imshow(confidence_raster, cmap='viridis', vmin=0.4, vmax=1.0)
ax_conf.set_title('Peta Keyakinan Prediksi Model (Posterior Confidence Score)', fontsize=12, fontweight="bold", pad=10)
ax_conf.axis('off')
cbar_conf = fig.colorbar(im_conf, ax=ax_conf, fraction=0.046, pad=0.04)
cbar_conf.set_label('Confidence (0.40 - 1.00)', fontsize=10)

plt.tight_layout()
plt.show()
plt.close()
"""))

# Cell 11: Markdown - 6.6 Spatial Export
cells.append(nbf.v4.new_markdown_cell("""## 6.6 Ekspor Hasil Inferensi Menjadi GeoTIFF Terkalibrasi Spasial

Hasil prediksi langsung diekspor ke file GeoTIFF dengan mempertahankan parameter georeferensi asli:
1. `classified_lulc.tif`: Raster kelas tutupan lahan (`uint8`, nodata=`255`).
2. `confidence.tif`: Raster tingkat keyakinan posterior (`float32`, nodata=`0.0`).
"""))

# Cell 12: Code - Save GeoTIFFs
cells.append(nbf.v4.new_code_cell("""outputs_dir = os.path.join(UTS_DIR, '6-deployment', 'outputs')
class_path, conf_path = save_inference_geotiffs(
    classified_raster, confidence_raster, prof_all, outputs_dir, prefix='classified'
)

print('=' * 75)
print('VERIFIKASI OUTPUT RASTER GEOSPASIAL:')
print(f'1. Classified Raster : {class_path} ({os.path.getsize(class_path)/1024:.1f} KB)')
print(f'2. Confidence Raster : {conf_path} ({os.path.getsize(conf_path)/1024:.1f} KB)')

with rasterio.open(class_path) as chk:
    print(f'• Validasi CRS       : {chk.crs}')
    print(f'• Validasi Bounding  : {chk.bounds}')
    print(f'• Validasi Resolusi  : {chk.res}')
    print(f'• Validasi NoData    : {chk.nodata}')
    assert chk.nodata == 255, "NoData GeoTIFF harus 255!"
print('[OK] Seluruh file GeoTIFF terkalibrasi spasial dengan sempurna.')
print('=' * 75)
"""))

# Cell 13: Markdown - 6.7 AOI Windowing & Coverage Check
cells.append(nbf.v4.new_markdown_cell("""## 6.7 Pemotongan Wilayah (AOI Windowing) dan Pengecekan Cakupan Spasial

Untuk menguji generalisasi dan kepatuhan sistem pada batas wilayah:
1. **Wilayah Tercakup (Contoh: Surabaya)**: Sistem memotong jendela spasial (*window clipping*) dari GeoTIFF asli dan melakukan inferensi khusus pada koordinat tersebut.
2. **Wilayah di Luar Cakupan (Contoh: Malang / Batu)**: Sistem mendeteksi tidak adanya irisan koordinat dan menampilkan pesan:
   `"Wilayah tidak tercakup oleh citra Sentinel-2 yang tersedia."` tanpa melakukan klasifikasi palsu.
"""))

# Cell 14: Code - AOI Windowing & Coverage Check
cells.append(nbf.v4.new_code_cell("""# 1. Wilayah Tercakup: Surabaya Metropolitan & Pesisir Timur
surabaya_aoi = REGIONS_AOI['Surabaya (Metropolitan & Wonorejo)']['bounds']
surabaya_bands, surabaya_prof, surabaya_bounds, surabaya_trans = read_sentinel2_raster(raw_raster_path, aoi_bounds=surabaya_aoi)

surabaya_infer = run_full_raster_inference(model, surabaya_bands, confidence_threshold=0.40)
print(f'• Sub-wilayah Surabaya : Diproses {surabaya_infer["total_pixels"]:,} piksel ({surabaya_infer["inference_time_sec"]:.2f} detik)')
print(f'  - Bangunan (Built-up) : {surabaya_infer["class_stats"][0]["percent"]:.2f}%')
print(f'  - Mangrove Pesisir    : {surabaya_infer["class_stats"][4]["percent"]:.2f}%')

# 2. Wilayah di Luar Cakupan: Kawasan Malang & Batu
malang_aoi = REGIONS_AOI['Malang & Batu (Dataran Tinggi & Pegunungan)']['bounds']
is_covered, inter_box, coverage_msg = check_raster_coverage(raster_bounds, malang_aoi)

print(f'\\n• Pengecekan Wilayah Malang & Batu:')
print(f'  - Status Cakupan     : {is_covered}')
print(f'  - Respon Sistem      : "{coverage_msg}"')
if not is_covered:
    print('  [BENAR] Sistem menolak klasifikasi palsu pada area yang tidak tercakup citra.')
"""))

# Cell 15: Markdown - 6.8 Streamlit Documentation & Repo Link
cells.append(nbf.v4.new_markdown_cell("""## 6.8 Dokumentasi Aplikasi Streamlit & Tautan Repositori Produksi

Aplikasi interaktif utama yang digunakan oleh pengguna akhir dikembangkan menggunakan **Streamlit** dan **Leaflet.js** pada file [`src/Tugas_UTS/6-deployment/app.py`](file:///d:/university/semester%205_book/proyek%20sain%20data/src/Tugas_UTS/6-deployment/app.py).

### Fitur Utama Aplikasi Streamlit:
1. **Full-Raster Trigger Button**: Inferensi hanya dijalankan ketika pengguna mengklik tombol *'Jalankan Inferensi Raster'*.
2. **Pengecekan Cakupan Wilayah Otomatis**: Menolak klasifikasi jika wilayah berada di luar batas citra.
3. **Peta Interaktif Leaflet**: Memproyeksikan layer raster hasil prediksi terbaru langsung di atas basemap satelit resolusi tinggi.
4. **Kontrol Layer Dinamis**: Checkbox kelas tutupan lahan untuk menyembunyikan/menampilkan kelas tanpa mengubah hasil prediksi model.
5. **Single-Pixel Profiler**: Memungkinkan pengguna memasukkan nilai reflektansi 7 band untuk melihat kurva spektral dan probabilitas posterior secara *real-time*.
6. **Unduh GeoTIFF**: Tombol unduh file `classified_lulc.tif` dan `confidence.tif`.

---

### Tautan Repositori Produksi Mandiri:
Aplikasi Streamlit production dipisahkan ke repositori independen yang siap dideploy ke Streamlit Cloud:
* **Tautan GitHub**: [https://github.com/IlhamInsight/sentinel2-jatim-lulc-app](https://github.com/IlhamInsight/sentinel2-jatim-lulc-app)
* **Perintah Menjalankan Aplikasi Lokal**:
  ```bash
  streamlit run src/Tugas_UTS/6-deployment/app.py
  ```

![Antarmuka Aplikasi Streamlit](streamlit_app_screenshot.png)
"""))

nb.cells = cells

# Save notebook
notebook_path = 'src/Tugas_UTS/6-deployment/6-deployment.ipynb'
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'[OK] File {notebook_path} berhasil dibuat dengan {len(cells)} cells.')
