# -*- coding: utf-8 -*-
"""
Sistem Klasifikasi Tutupan Lahan Sentinel-2 Jawa Timur
CRISP-DM Phase 6: Deployment
Model   : LightGBM Pixel Classifier (7 band BOA + 6 indeks biofisik = 13 fitur)
Revisi  : - Masking piksel tidak valid (NoData, saturasi, awan terang) sebelum dan sesudah inferensi
          - Filter mayoritas (post-processing) untuk mengurangi efek salt-and-pepper
          - Diagnostik kualitas hasil dan metrik per kelas dari matriks konfusi
          - Tampilan profesional tanpa emoji, fitur tidak esensial dihapus
          - Seluruh path file TIDAK diubah
"""

import os
import sys
import json
import time
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Resolusi direktori
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UTS_DIR = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "6-deployment" else CURRENT_DIR
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import io
import base64
from PIL import Image
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

# ==============================================================================
# Konstanta Spektral & 13 Fitur Prediktor (Urutan Pasti Training)
# ==============================================================================
BAND_NAMES = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
INDEX_NAMES = ["NDVI", "MNDWI", "NDBI", "Mangrove_Index_8A_11_4", "EVI", "BSI"]
FEATURE_COLS = BAND_NAMES + INDEX_NAMES

# Definisi Kelas Tutupan Lahan (LULC) - Warna Kontras Tinggi untuk Peta Satelit
CLASS_CONFIG = {
    0: {"key": "built", "name": "Bangunan", "alias": "Built-up", "color": "#ef4444", "rgb": (239, 68, 68), "desc": "Permukiman, industri, jalan, dan perkerasan beton."},
    1: {"key": "farm", "name": "Lahan Pertanian", "alias": "Farmland", "color": "#eab308", "rgb": (234, 179, 8), "desc": "Sawah irigasi, tegalan, ladang semusim, dan hortikultura."},
    2: {"key": "water", "name": "Perairan", "alias": "Water Body", "color": "#0284c7", "rgb": (2, 132, 199), "desc": "Selat Madura, laut pesisir, muara sungai, dan waduk."},
    3: {"key": "forest", "name": "Hutan Daratan", "alias": "Terrestrial Forest", "color": "#22c55e", "rgb": (34, 197, 94), "desc": "Hutan alami dan tutupan kanopi lebat dataran tinggi."},
    4: {"key": "mangrove", "name": "Hutan Mangrove", "alias": "Mangrove Forest", "color": "#06b6d4", "rgb": (6, 182, 212), "desc": "Vegetasi bakau pasang-surut pantai dan muara berlumpur."},
    255: {"key": "nodata", "name": "Unknown / Low Confidence", "alias": "Unknown/NoData", "color": "#000000", "rgb": (0, 0, 0), "desc": "Piksel NoData, awan/saturasi, atau di bawah ambang keyakinan."}
}
VALID_CLASSES = [0, 1, 2, 3, 4]

# Batas Wilayah Studi (EPSG:4326)
REGIONS_AOI = {
    "Seluruh Citra Scene (Surabaya - Sidoarjo - Gresik)": {
        "bounds": (112.50, -7.60, 113.00, -7.00),
        "center": (-7.30, 112.75),
        "zoom": 10,
        "title": "Scene Sentinel-2 Jawa Timur (Penuh)",
        "desc": "Wilayah cakupan penuh citra Sentinel-2 mencakup Surabaya, Sidoarjo, dan pesisir Gresik.",
    },
    "Surabaya (Metropolitan & Wonorejo)": {
        "bounds": (112.60, -7.36, 112.85, -7.18),
        "center": (-7.26, 112.75),
        "zoom": 12,
        "title": "Kota Surabaya (Urban & Pesisir Timur)",
        "desc": "Pusat bisnis perkotaan, kawasan industri, pelabuhan, dan mangrove pesisir Wonorejo.",
    },
    "Sidoarjo (Delta Brantas & Tambak)": {
        "bounds": (112.55, -7.58, 112.88, -7.36),
        "center": (-7.45, 112.72),
        "zoom": 12,
        "title": "Kabupaten Sidoarjo (Delta & Perikanan)",
        "desc": "Hamparan persawahan delta sungai Brantas, kawasan industri, dan tambak pesisir timur.",
    },
    "Gresik (Kawasan Pesisir & Industri)": {
        "bounds": (112.50, -7.22, 112.70, -7.02),
        "center": (-7.12, 112.60),
        "zoom": 12,
        "title": "Gresik (Kawasan Industri & Pesisir)",
        "desc": "Kawasan industri pesisir Selat Madura dan muara sungai utara.",
    },
}

# Rekayasa 6 Indeks Biofisik Teruji
def compute_spectral_indices(b02, b03, b04, b08, b8a, b11, b12, eps=1e-6):
    b02, b03, b04 = np.asarray(b02, dtype=np.float32), np.asarray(b03, dtype=np.float32), np.asarray(b04, dtype=np.float32)
    b08, b8a = np.asarray(b08, dtype=np.float32), np.asarray(b8a, dtype=np.float32)
    b11, b12 = np.asarray(b11, dtype=np.float32), np.asarray(b12, dtype=np.float32)

    ndvi = np.clip((b08 - b04) / (b08 + b04 + eps), -1.0, 1.0)
    mndwi = np.clip((b03 - b11) / (b03 + b11 + eps), -1.0, 1.0)
    ndbi = np.clip((b11 - b08) / (b11 + b08 + eps), -1.0, 1.0)
    mangrove = np.clip((b8a - b04) / (b11 + eps), -10.0, 10.0)

    evi_denom = b08 + 6.0 * b04 - 7.5 * b02 + 1.0
    safe_denom = np.where(np.abs(evi_denom) < 1e-4, 1e-4, evi_denom)
    evi = np.clip(2.5 * (b08 - b04) / safe_denom, -3.0, 3.0)
    bsi = np.clip(((b11 + b04) - (b08 + b02)) / ((b11 + b04) + (b08 + b02) + eps), -1.0, 1.0)

    return {"NDVI": ndvi, "MNDWI": mndwi, "NDBI": ndbi, "Mangrove_Index_8A_11_4": mangrove, "EVI": evi, "BSI": bsi}

# Ekstraksi 13 Fitur Pasti
def extract_13_features(bands_array, eps=1e-6):
    arr = np.asarray(bands_array, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 7:
        _, h, w = arr.shape
        idx = compute_spectral_indices(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], eps=eps)
        feature_stack = np.stack([
            arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6],
            idx["NDVI"], idx["MNDWI"], idx["NDBI"], idx["Mangrove_Index_8A_11_4"], idx["EVI"], idx["BSI"]
        ], axis=-1)
        return feature_stack.reshape(-1, 13), (h, w)
    elif arr.ndim == 2 and arr.shape[1] == 7:
        idx = compute_spectral_indices(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5], arr[:, 6], eps=eps)
        X = np.column_stack([
            arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5], arr[:, 6],
            idx["NDVI"], idx["MNDWI"], idx["NDBI"], idx["Mangrove_Index_8A_11_4"], idx["EVI"], idx["BSI"]
        ])
        return X, None
    elif arr.ndim == 1 and arr.shape[0] == 7:
        idx = compute_spectral_indices(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], eps=eps)
        row = [float(arr[i]) for i in range(7)] + [
            float(idx["NDVI"]), float(idx["MNDWI"]), float(idx["NDBI"]),
            float(idx["Mangrove_Index_8A_11_4"]), float(idx["EVI"]), float(idx["BSI"])
        ]
        return np.array([row], dtype=np.float32), None
    raise ValueError(f"Bentuk bands_array tidak valid: {arr.shape}")

# Validasi Integritas Model
def validate_model(mdl):
    if not hasattr(mdl, "classes_") or list(mdl.classes_) != VALID_CLASSES:
        raise ValueError(f"Kelas model tidak sesuai! Ekspektasi {VALID_CLASSES}")
    if not hasattr(mdl, "n_features_in_") or mdl.n_features_in_ != len(FEATURE_COLS):
        raise ValueError(f"Jumlah fitur tidak sesuai! Ekspektasi {len(FEATURE_COLS)}")
    if not hasattr(mdl, "predict_proba"):
        raise ValueError("Model tidak mendukung fungsi predict_proba!")
    return True

# Geospasial: Pembacaan & Potong Window Rasterio
def check_raster_coverage(raster_bounds, aoi_bounds):
    r_left, r_bottom, r_right, r_top = raster_bounds
    aoi_left, aoi_bottom, aoi_right, aoi_top = aoi_bounds
    if aoi_left >= r_right or aoi_right <= r_left or aoi_bottom >= r_top or aoi_top <= r_bottom:
        return False, None, "Wilayah tidak tercakup oleh citra Sentinel-2 yang tersedia."
    inter = (max(r_left, aoi_left), max(r_bottom, aoi_bottom), min(r_right, aoi_right), min(r_top, aoi_top))
    if inter[0] >= inter[2] or inter[1] >= inter[3]:
        return False, None, "Wilayah tidak tercakup oleh citra Sentinel-2 yang tersedia."
    return True, inter, "Wilayah berhasil tercakup oleh citra Sentinel-2."

@st.cache_data(show_spinner=False)
def read_sentinel2_raster(tif_path, aoi_bounds=None):
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"File GeoTIFF tidak ditemukan: {tif_path}")
    with rasterio.open(tif_path) as src:
        crs, bounds, prof = src.crs, src.bounds, src.profile.copy()
        if crs and crs.to_epsg() != 4326:
            r_box = transform_bounds(crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top)
        else:
            r_box = (bounds.left, bounds.bottom, bounds.right, bounds.top)
        if aoi_bounds is not None:
            ok, inter, _ = check_raster_coverage(r_box, aoi_bounds)
            if not ok:
                raise ValueError("Wilayah tidak tercakup oleh citra Sentinel-2 yang tersedia.")
            win = from_bounds(*inter, transform=src.transform).round_offsets().round_lengths()
            bands = src.read(window=win)
            prof.update({"height": bands.shape[1], "width": bands.shape[2], "transform": rasterio.windows.transform(win, src.transform)})
            return bands.astype(np.float32), prof, rasterio.windows.bounds(win, src.transform), src.transform
        return src.read().astype(np.float32), prof, r_box, src.transform

# Inferensi Full-Raster Cepat (Multi-Thread & Seleksi Piksel Valid)
def run_full_raster_inference(model=None, bands=None, confidence_threshold=0.60, mdl=None):
    active_mdl = model if model is not None else mdl
    if active_mdl is None or bands is None:
        raise ValueError("Model dan array bands harus disediakan.")
    validate_model(active_mdl)
    t0 = time.time()
    _, h, w = bands.shape
    X, _ = extract_13_features(bands)
    
    # Deteksi piksel invalid (NoData / saturasi)
    invalid_mask = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | (X[:, :7] <= 0.0).all(axis=1) | (X[:, :7] > 1.4).any(axis=1)
    valid_idx = np.where(~invalid_mask)[0]
    
    max_idx = np.zeros(len(X), dtype=int)
    max_prob = np.zeros(len(X), dtype=np.float32)
    
    # Jalankan prediksi hanya pada piksel yang valid untuk efisiensi komputasi
    if len(valid_idx) > 0:
        X_valid = np.nan_to_num(X[valid_idx], nan=0.0, posinf=1.0, neginf=0.0)
        try:
            active_mdl.set_params(n_jobs=-1)
        except Exception:
            pass
        probs_valid = active_mdl.predict_proba(X_valid)
        max_idx[valid_idx] = np.argmax(probs_valid, axis=1)
        max_prob[valid_idx] = np.max(probs_valid, axis=1)
    
    preds = active_mdl.classes_[max_idx].copy()
    preds[max_prob < confidence_threshold] = 255
    preds[invalid_mask] = 255
    max_prob[invalid_mask] = 0.0
    
    cls_raster = preds.reshape(h, w).astype(np.uint8)
    conf_raster = max_prob.reshape(h, w).astype(np.float32)
    return {
        "classified_raster": cls_raster,
        "confidence_raster": conf_raster,
        "inference_time_sec": time.time() - t0,
        "shape": (h, w)
    }

# Simpan GeoTIFF & Overlay RGBA
def save_inference_geotiffs(classified_raster, confidence_raster, base_profile, output_dir, prefix="classified"):
    os.makedirs(output_dir, exist_ok=True)
    h, w = classified_raster.shape
    c_prof = base_profile.copy()
    c_prof.update({"driver": "GTiff", "height": h, "width": w, "count": 1, "dtype": rasterio.uint8, "nodata": 255, "compress": "lzw"})
    c_path = os.path.join(output_dir, f"{prefix}_lulc.tif")
    with rasterio.open(c_path, "w", **c_prof) as dst:
        dst.write(classified_raster.astype(np.uint8), 1)

    cf_prof = base_profile.copy()
    cf_prof.update({"driver": "GTiff", "height": h, "width": w, "count": 1, "dtype": rasterio.float32, "nodata": 0.0, "compress": "lzw"})
    cf_path = os.path.join(output_dir, "confidence.tif")
    with rasterio.open(cf_path, "w", **cf_prof) as dst:
        dst.write(confidence_raster.astype(np.float32), 1)
    return c_path, cf_path

def create_raster_rgba_overlay(classified_raster, active_classes=None, opacity=1.0):
    if active_classes is None:
        active_classes = VALID_CLASSES
    active_set = set(active_classes)
    h, w = classified_raster.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    alpha_val = int(np.clip(opacity, 0.0, 1.0) * 255)
    for c in VALID_CLASSES:
        if c in active_set:
            mask = (classified_raster == c)
            rgb = CLASS_CONFIG[c]["rgb"]
            rgba[mask, 0] = rgb[0]
            rgba[mask, 1] = rgb[1]
            rgba[mask, 2] = rgb[2]
            rgba[mask, 3] = alpha_val
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    # Kompresi level 1 untuk performa kilat (10x lebih cepat daripada optimize=True)
    img.save(buf, format="PNG", compress_level=1)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}", img

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from scipy.ndimage import uniform_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==============================================================================
# 1. Konfigurasi Halaman
# ==============================================================================
st.set_page_config(
    page_title="Klasifikasi Tutupan Lahan Sentinel-2 Jawa Timur",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Path file (tidak diubah)
MODEL_PATH = os.path.join(UTS_DIR, "4-modeling", "models", "model_lightgbm_piksel.pkl")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(CURRENT_DIR, "models", "model_lightgbm_piksel.pkl")

RAW_SCENE_TIF = os.path.join(UTS_DIR, "2-data_understanding", "extracted_data", "sentinel2_jatim_scene.tif")
if not os.path.exists(RAW_SCENE_TIF):
    RAW_SCENE_TIF = os.path.join(CURRENT_DIR, "prepared_data", "sentinel2_jatim_scene.tif")

METRICS_PATH = os.path.join(UTS_DIR, "5-evaluation", "metrics_evaluation.json")
if not os.path.exists(METRICS_PATH):
    METRICS_PATH = os.path.join(CURRENT_DIR, "prepared_data", "metrics_evaluation.json")

GEOJSON_DIR = os.path.join(UTS_DIR, "2-data_understanding", "sampling_geojson")
if not os.path.exists(GEOJSON_DIR):
    GEOJSON_DIR = os.path.join(CURRENT_DIR, "sampling_geojson")

OUTPUTS_DIR = os.path.join(CURRENT_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

NODATA_CLASS = 255
# Batas reflektansi valid. Data latih bernilai maksimum sekitar 0.95, sehingga piksel > 1.0
# berada di luar distribusi latih (saturasi/awan/artefak) dan tidak boleh diklasifikasikan.
MAX_VALID_REFLECTANCE = 1.0
# Ambang awan terang: seluruh band tampak (B02, B03, B04) sangat tinggi dan hampir rata (spektrum putih).
CLOUD_VIS_MIN = 0.35
CLOUD_FLATNESS = 0.12

# ==============================================================================
# 2. Pemuatan & Validasi Model
# ==============================================================================
@st.cache_resource(show_spinner="Memuat dan memvalidasi model LightGBM...")
def load_and_validate_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File model tidak ditemukan pada: {path}")
    mdl = joblib.load(path)
    validate_model(mdl)
    return mdl


try:
    model = load_and_validate_model(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal memuat model klasifikasi: {e}")
    st.stop()

# ==============================================================================
# 3. Metadata Evaluasi & Poligon Acuan
# ==============================================================================
@st.cache_data
def load_evaluation_metrics(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


evaluation_metrics = load_evaluation_metrics(METRICS_PATH)


@st.cache_data
def load_all_ground_truth_polygons(g_dir):
    """Muat poligon sampel lapangan 5 kelas. Ekstensi .geojson/.json dicari otomatis
    agar kelas yang nama filenya berbeda (mis. mangrove.json) tetap terbaca."""
    file_map = {
        0: ("bangunan", "Bangunan", "#ef4444", "BGN"),
        1: ("pertanian", "Lahan Pertanian", "#eab308", "PRT"),
        2: ("perairan", "Perairan", "#0284c7", "AIR"),
        3: ("hutan", "Hutan Daratan", "#15803d", "HTN"),
        4: ("mangrove", "Hutan Mangrove", "#0d9488", "MGV"),
    }
    all_features, missing = [], []
    for cid, (stem, cname, col, pfx) in file_map.items():
        candidates = []
        for ext in (".geojson", ".json"):
            candidates += glob.glob(os.path.join(g_dir, stem + ext))
        if not candidates:
            candidates = glob.glob(os.path.join(g_dir, f"*{stem}*"))
        if not candidates:
            missing.append(cname)
            continue
        try:
            with open(candidates[0], "r", encoding="utf-8") as f:
                data = json.load(f)
            for idx, feat in enumerate(data.get("features", [])):
                geom = feat.get("geometry")
                if geom and geom.get("coordinates"):
                    all_features.append({
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {
                            "id": f"{pfx}-{idx + 1:02d}",
                            "class_id": cid,
                            "className": cname,
                            "color": col,
                        },
                    })
        except Exception:
            missing.append(cname)
    return {"type": "FeatureCollection", "features": all_features}, missing


ground_truth_polygons, gt_missing = load_all_ground_truth_polygons(GEOJSON_DIR)


@st.cache_data
def get_scene_bounds_4326(tif_path):
    """Batas sebenarnya citra Sentinel-2 dalam EPSG:4326 (min_lon, min_lat, max_lon, max_lat)."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        with rasterio.open(tif_path) as src:
            return tuple(transform_bounds(src.crs, "EPSG:4326", *src.bounds))
    except Exception:
        return None


scene_bounds = get_scene_bounds_4326(RAW_SCENE_TIF)

# ==============================================================================
# 4. Fungsi Kualitas Data & Post-processing
# ==============================================================================
def _to_7hw(bands):
    """Ubah stack band ke bentuk (7, H, W) float32 beserta mask bawaan jika ada."""
    arr = bands
    pre_mask = None
    if isinstance(arr, np.ma.MaskedArray):
        pre_mask = np.ma.getmaskarray(arr)
        arr = np.ma.getdata(arr)
        if pre_mask is not None:
            pre_mask = pre_mask if pre_mask.ndim == 3 else None
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Bentuk array band tidak dikenali: {arr.shape}")
    if arr.shape[0] == 7:
        return arr, pre_mask
    if arr.shape[-1] == 7:
        arr = np.moveaxis(arr, -1, 0)
        if pre_mask is not None:
            pre_mask = np.moveaxis(pre_mask, -1, 0)
        return arr, pre_mask
    raise ValueError(f"Tidak ditemukan sumbu 7 band pada array berbentuk {arr.shape}")


def build_invalid_mask(bands, mask_cloud=True):
    """Kembalikan mask (H, W) piksel yang tidak boleh diklasifikasikan.

    Alasan temuan: model overconfident pada piksel di luar distribusi latih.
    NoData (semua band 0) diprediksi Perairan 100%, awan diprediksi Bangunan 100%,
    sehingga ambang confidence saja tidak bisa menyaringnya."""
    arr, pre_mask = _to_7hw(bands)
    _, h, w = arr.shape
    invalid = np.zeros((h, w), dtype=bool)
    all_zero = np.ones((h, w), dtype=bool)
    for i in range(7):
        b = arr[i].astype(np.float32, copy=False)
        invalid |= ~np.isfinite(b)
        invalid |= b > MAX_VALID_REFLECTANCE
        invalid |= b < 0
        all_zero &= (b == 0)
    invalid |= all_zero
    if pre_mask is not None:
        invalid |= pre_mask.any(axis=0)

    n_nodata = int(invalid.sum())
    n_cloud = 0
    if mask_cloud:
        b2, b3, b4 = (arr[i].astype(np.float32, copy=False) for i in range(3))
        vis_min = np.minimum(np.minimum(b2, b3), b4)
        vis_max = np.maximum(np.maximum(b2, b3), b4)
        cloud = (vis_min > CLOUD_VIS_MIN) & ((vis_max - vis_min) < CLOUD_FLATNESS) & ~invalid
        n_cloud = int(cloud.sum())
        invalid |= cloud
    return invalid, {"nodata_or_saturated": n_nodata, "cloud_like": n_cloud}


def majority_filter(cls, size=3, chunk=1024):
    """Filter mayoritas per blok baris. Piksel 255 tidak diisi ulang."""
    if size <= 1 or not HAS_SCIPY:
        return cls
    h, _ = cls.shape
    pad = size // 2 + 1
    out = cls.copy()
    for r0 in range(0, h, chunk):
        r1 = min(h, r0 + chunk)
        a0, a1 = max(0, r0 - pad), min(h, r1 + pad)
        blk = cls[a0:a1]
        best_score = np.full(blk.shape, -1.0, dtype=np.float32)
        best_cls = np.full(blk.shape, NODATA_CLASS, dtype=np.uint8)
        for c in VALID_CLASSES:
            sc = uniform_filter((blk == c).astype(np.float32), size=size, mode="constant")
            upd = sc > best_score
            best_score[upd] = sc[upd]
            best_cls[upd] = c
        res = best_cls[r0 - a0: r0 - a0 + (r1 - r0)]
        out[r0:r1] = np.where(cls[r0:r1] == NODATA_CLASS, NODATA_CLASS, res)
    return out


def compute_class_stats(cls, n_invalid, n_lowconf):
    """Statistik kelas. Persentase kelas dihitung terhadap piksel terklasifikasi."""
    counts = np.bincount(cls.ravel(), minlength=256)
    total = int(cls.size)
    classified = int(sum(counts[c] for c in VALID_CLASSES))
    stats = {}
    for c in VALID_CLASSES:
        stats[c] = {
            "count": int(counts[c]),
            "percent": (counts[c] / classified * 100.0) if classified else 0.0,
        }
    stats["summary"] = {
        "total": total,
        "classified": classified,
        "classified_pct": classified / total * 100.0 if total else 0.0,
        "invalid": int(n_invalid),
        "invalid_pct": n_invalid / total * 100.0 if total else 0.0,
        "lowconf": int(n_lowconf),
        "lowconf_pct": n_lowconf / total * 100.0 if total else 0.0,
    }
    return stats


def pct(v):
    """Normalisasi nilai metrik ke persen (menerima 0-1 maupun 0-100)."""
    if isinstance(v, (int, float)):
        return v * 100.0 if v <= 1.0 else float(v)
    return None


# ==============================================================================
# 5. Styling
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif !important; }
.stApp { background-color: #0e1117 !important; color: #e4e4e7 !important; }
section[data-testid="stSidebar"] { background-color: #161920 !important; border-right: 1px solid #262730 !important; }
h1, h2, h3, h4, h5 { color: #f4f4f5 !important; font-weight: 600 !important; letter-spacing: -0.01em !important; }
.app-header { border-bottom: 1px solid #262730; padding-bottom: 0.9rem; margin-bottom: 1rem; }
.app-title { font-size: 1.45rem; font-weight: 700; color: #f4f4f5; }
.app-sub { font-size: 0.82rem; color: #a1a1aa; margin-top: 0.3rem; max-width: 980px; line-height: 1.55; }
.tag { display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 600;
       padding: 0.18rem 0.55rem; border-radius: 3px; background: #262730; color: #d4d4d8;
       border: 1px solid #3f3f46; margin-right: 0.35rem; }
.panel { background: #161920; border: 1px solid #262730; border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 0.9rem; }
.panel-title { font-size: 0.95rem; font-weight: 600; color: #f4f4f5; }
.panel-sub { font-size: 0.76rem; color: #a1a1aa; margin-top: 2px; }
.class-cell { background: #161920; border: 1px solid #262730; border-radius: 6px; padding: 0.55rem 0.7rem; }
.class-cell .nm { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
.class-cell .vl { font-size: 1.05rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; margin: 2px 0; color: #f4f4f5; }
.class-cell .pc { font-size: 0.7rem; color: #a1a1aa; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; background-color: #161920; padding: 4px; border-radius: 6px; border: 1px solid #262730; }
.stTabs [data-baseweb="tab"] { background-color: transparent; border-radius: 4px; color: #a1a1aa !important; font-weight: 600; font-size: 0.82rem; padding: 6px 14px; }
.stTabs [aria-selected="true"] { background-color: #27272a !important; color: #f4f4f5 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <span class="tag">SENTINEL-2 L2A</span><span class="tag">10 METER</span>
    <span class="tag">LIGHTGBM</span><span class="tag">13 FITUR</span>
    <div class="app-title" style="margin-top:0.6rem;">Klasifikasi Tutupan Lahan Sentinel-2 Jawa Timur</div>
    <div class="app-sub">
        Inferensi tingkat piksel dari GeoTIFF reflektansi Sentinel-2 (7 band BOA dan 6 indeks biofisik).
        Piksel NoData, saturasi, dan awan terang dikeluarkan dari klasifikasi, kemudian hasil dihaluskan
        dengan filter mayoritas dan disaring berdasarkan ambang keyakinan model.
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 6. Sidebar
# ==============================================================================
with st.sidebar:
    st.markdown("### Parameter Analisis")

    region_options = list(REGIONS_AOI.keys())
    selected_region_name = st.selectbox(
        "Wilayah analisis",
        region_options,
        index=0,
        help="Pilih wilayah analisis yang tercakup dalam citra Sentinel-2.",
    )
    selected_region_meta = REGIONS_AOI[selected_region_name]

    run_inference_btn = st.button("Jalankan Inferensi", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("#### Tampilan Layer")
    show_built = st.checkbox("Bangunan", value=True)
    show_farm = st.checkbox("Lahan Pertanian", value=True)
    show_water = st.checkbox("Perairan", value=True)
    show_forest = st.checkbox("Hutan Daratan", value=True)
    show_mangrove = st.checkbox("Hutan Mangrove", value=True)
    active_classes = [c for c, f in enumerate([show_built, show_farm, show_water, show_forest, show_mangrove]) if f]

    show_polygons = st.checkbox("Poligon sampel acuan", value=False,
                                help="Garis batas sampel lapangan untuk kelas yang aktif.")
    basemap_choice = st.selectbox(
        "Basemap",
        ["Esri World Imagery", "Esri Dark Gray Canvas", "OpenStreetMap"],
        index=0,
    )
    overlay_opacity = st.slider("Opasitas prediksi", 0.10, 1.00, 0.75, 0.05)

# Pipeline inferensi per-wilayah dengan caching performa tinggi
@st.cache_data(show_spinner=False)
def execute_region_pipeline(region_name, _model_obj, raw_scene_path):
    region_meta = REGIONS_AOI[region_name]
    aoi = region_meta["bounds"] if "Penuh" not in region_meta["title"] else None
    bands, prof, bounds_4326, win_transform = read_sentinel2_raster(
        raw_scene_path,
        aoi_bounds=aoi,
    )
    invalid_mask, mask_info = build_invalid_mask(bands, mask_cloud=True)
    infer_result = run_full_raster_inference(
        model=_model_obj,
        bands=bands,
        confidence_threshold=0.60,
    )
    cls_raster = np.asarray(infer_result["classified_raster"]).astype(np.uint8, copy=True)
    conf_raster = np.asarray(infer_result["confidence_raster"]).copy()
    lowconf_mask = (cls_raster == NODATA_CLASS) & ~invalid_mask
    cls_raster[invalid_mask] = NODATA_CLASS
    conf_raster[invalid_mask] = 0
    stats = compute_class_stats(cls_raster, int(invalid_mask.sum()), int(lowconf_mask.sum()))

    class_path, conf_path = save_inference_geotiffs(
        classified_raster=cls_raster,
        confidence_raster=conf_raster,
        base_profile=prof,
        output_dir=OUTPUTS_DIR,
        prefix="classified",
    )

    stride = int(np.ceil(max(cls_raster.shape) / 2500.0))
    display_raster = cls_raster[::stride, ::stride] if stride > 1 else cls_raster

    return {
        "region_name": region_name,
        "region_meta": region_meta,
        "bounds_4326": bounds_4326,
        "display_raster": display_raster,
        "class_stats": stats,
        "mask_info": mask_info,
        "inference_time_sec": infer_result["inference_time_sec"],
        "class_geotiff_path": class_path,
        "conf_geotiff_path": conf_path,
    }

# ==============================================================================
# 7. Eksekusi Inferensi
# ==============================================================================
if run_inference_btn:
    with st.spinner(f"Memproses inferensi pada {selected_region_name}..."):
        try:
            res = execute_region_pipeline(selected_region_name, model, RAW_SCENE_TIF)
            st.session_state["inference_result"] = res
            st.success(f"Inferensi selesai dalam {res['inference_time_sec']:.2f} detik.")
        except Exception as ex:
            st.session_state.pop("inference_result", None)
            st.error(f"Terjadi kesalahan saat memproses inferensi: {ex}")

# ==============================================================================
# 8. Antarmuka Utama
# ==============================================================================
tab_map, tab_profiler, tab_eval = st.tabs([
    "Peta Klasifikasi",
    "Inspeksi Piksel",
    "Evaluasi Model",
])

# ------------------------------------------------------------------------------
# TAB 1: PETA
# ------------------------------------------------------------------------------
with tab_map:
    if "inference_result" not in st.session_state:
        st.info("Pilih wilayah analisis lalu klik Jalankan Inferensi pada sidebar.")
    else:
        res = st.session_state["inference_result"]
        stats = res["class_stats"]
        summ = stats["summary"]
        bounds_4326 = res["bounds_4326"]

        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">{res['region_name']}</div>
            <div class="panel-sub">{res['region_meta']['desc']} &nbsp;|&nbsp; {summ['total']:,} piksel
            &nbsp;|&nbsp; {res['inference_time_sec']:.2f} detik</div>
        </div>
        """, unsafe_allow_html=True)

        q1, q2, q3 = st.columns(3)
        q1.metric("Piksel terklasifikasi", f"{summ['classified_pct']:.1f}%")
        q2.metric("Dikeluarkan (NoData, saturasi, awan)", f"{summ['invalid_pct']:.1f}%")
        q3.metric("Keyakinan rendah", f"{summ['lowconf_pct']:.1f}%")

        if summ["classified_pct"] < 60:
            st.warning(
                "Kurang dari 60% piksel dapat diklasifikasikan dengan pasti. "
                "Periksa kualitas citra (awan, NoData) atau turunkan ambang keyakinan."
            )

        st.caption("Persentase kelas dihitung terhadap piksel terklasifikasi saja.")
        grid_cols = st.columns(5)
        for cid in VALID_CLASSES:
            c_info = CLASS_CONFIG[cid]
            c_stat = stats[cid]
            is_active = cid in active_classes
            color = c_info["color"] if is_active else "#52525b"
            status = "Ditampilkan" if is_active else "Disembunyikan"
            with grid_cols[cid]:
                st.markdown(f"""
                <div class="class-cell" style="border-top: 3px solid {color};">
                    <div class="nm" style="color:{color};">{c_info['name']}</div>
                    <div class="vl">{c_stat['count']:,} px</div>
                    <div class="pc">{c_stat['percent']:.2f}% &nbsp;|&nbsp; {status}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

        # Opasitas diterapkan sekali di Leaflet agar tidak terjadi pengalian opasitas ganda
        dynamic_scene_uri, _ = create_raster_rgba_overlay(
            classified_raster=res["display_raster"],
            active_classes=active_classes,
            opacity=1.0,
        )

        if show_polygons:
            active_gt = [f for f in ground_truth_polygons.get("features", [])
                         if f.get("properties", {}).get("class_id") in active_classes]
        else:
            active_gt = []
        gt_json = json.dumps({"type": "FeatureCollection", "features": active_gt})

        leaflet_bounds = [
            [float(bounds_4326[1]), float(bounds_4326[0])],
            [float(bounds_4326[3]), float(bounds_4326[2])],
        ]

        if "Dark" in basemap_choice:
            tile_url = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
        elif "OpenStreet" in basemap_choice:
            tile_url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        else:
            tile_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

        leaflet_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8"/>
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <style>
                html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; background: #0e1117; overflow: hidden; }}
                #wrap {{ position: relative; width: 100%; height: 560px; border: 1px solid #262730; border-radius: 6px; overflow: hidden; }}
                #map {{ width: 100%; height: 100%; background: #0e1117; }}
                .hud {{ position: absolute; z-index: 500; background: rgba(22,25,32,0.92); border: 1px solid #27272a;
                        padding: 5px 10px; border-radius: 4px; font-family: monospace; font-size: 11px; color: #d4d4d8; }}
                .hud-b {{ bottom: 10px; left: 10px; }}
                .hud-l {{ bottom: 10px; right: 10px; display: flex; gap: 10px; align-items: center; }}
                .dot {{ width: 9px; height: 9px; border-radius: 2px; display: inline-block; margin-right: 4px; }}
            </style>
        </head>
        <body>
            <div id="wrap">
                <div id="map"></div>
                <div class="hud hud-b">EPSG:4326 | [{bounds_4326[1]:.3f}, {bounds_4326[0]:.3f}] - [{bounds_4326[3]:.3f}, {bounds_4326[2]:.3f}]</div>
                <div class="hud hud-l">
                    <span><span class="dot" style="background:#ef4444"></span>Bangunan</span>
                    <span><span class="dot" style="background:#eab308"></span>Pertanian</span>
                    <span><span class="dot" style="background:#0284c7"></span>Perairan</span>
                    <span><span class="dot" style="background:#22c55e"></span>Hutan</span>
                    <span><span class="dot" style="background:#06b6d4"></span>Mangrove</span>
                </div>
            </div>
            <script>
                const map = L.map('map', {{ zoomControl: true, attributionControl: false }});
                L.tileLayer('{tile_url}', {{ maxZoom: 18 }}).addTo(map);
                const rasterBounds = {leaflet_bounds};
                const rasterUri = '{dynamic_scene_uri}';
                if (rasterUri && rasterUri.length > 50) {{
                    L.imageOverlay(rasterUri, rasterBounds, {{ opacity: {overlay_opacity}, interactive: false }}).addTo(map);
                }}
                const polyData = {gt_json};
                if (polyData.features && polyData.features.length > 0) {{
                    L.geoJSON(polyData, {{
                        style: f => ({{ color: f.properties.color, weight: 2, opacity: 0.95,
                                        fillColor: f.properties.color, fillOpacity: 0.12, dashArray: '4,4' }}),
                        onEachFeature: (f, layer) => layer.bindTooltip(
                            "<b>" + f.properties.className + "</b><br>ID: " + f.properties.id, {{ sticky: true }})
                    }}).addTo(map);
                }}
                map.fitBounds(rasterBounds);
            </script>
        </body>
        </html>
        """
        st.components.v1.html(leaflet_html, height=580)

        if show_polygons and gt_missing:
            st.caption("File poligon tidak ditemukan untuk kelas: " + ", ".join(gt_missing))

        col_dl1, col_dl2 = st.columns(2)
        slug = res["region_name"].lower().replace(" ", "_")
        with col_dl1:
            if os.path.exists(res["class_geotiff_path"]):
                with open(res["class_geotiff_path"], "rb") as f_class:
                    st.download_button("Unduh raster klasifikasi (GeoTIFF)", data=f_class,
                                       file_name=f"{slug}_classified_lulc.tif", mime="image/tiff",
                                       use_container_width=True)
        with col_dl2:
            if os.path.exists(res["conf_geotiff_path"]):
                with open(res["conf_geotiff_path"], "rb") as f_conf:
                    st.download_button("Unduh peta keyakinan (GeoTIFF)", data=f_conf,
                                       file_name=f"{slug}_confidence.tif", mime="image/tiff",
                                       use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: INSPEKSI PIKSEL
# ------------------------------------------------------------------------------
with tab_profiler:
    st.markdown("### Inspeksi Piksel Tunggal")
    st.caption("Masukkan reflektansi BOA tujuh band untuk melihat prediksi, probabilitas per kelas, dan indeks biofisik. "
               "Nilai contoh bersifat ilustratif, bukan sampel terverifikasi.")

    col_presets, col_inputs = st.columns([1, 2])
    with col_presets:
        preset_choice = st.selectbox(
            "Contoh spektrum",
            ["Input manual", "Bangunan", "Pertanian", "Perairan", "Hutan daratan", "Mangrove"],
        )
        preset_vals = {
            "Bangunan": [0.18, 0.19, 0.22, 0.25, 0.26, 0.31, 0.29],
            "Pertanian": [0.08, 0.11, 0.12, 0.25, 0.26, 0.23, 0.17],
            "Perairan": [0.10, 0.11, 0.09, 0.06, 0.06, 0.05, 0.04],
            "Hutan daratan": [0.04, 0.06, 0.05, 0.28, 0.29, 0.18, 0.10],
            "Mangrove": [0.03, 0.06, 0.04, 0.30, 0.31, 0.13, 0.06],
        }
        default_b = preset_vals.get(preset_choice, [0.10, 0.12, 0.11, 0.25, 0.26, 0.20, 0.15])

    with col_inputs:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            p_b02 = st.slider("B02 (Blue)", 0.0, 1.0, float(default_b[0]), 0.01, key=f"b02_{preset_choice}")
            p_b8a = st.slider("B8A (NIR sempit)", 0.0, 1.0, float(default_b[4]), 0.01, key=f"b8a_{preset_choice}")
        with c2:
            p_b03 = st.slider("B03 (Green)", 0.0, 1.0, float(default_b[1]), 0.01, key=f"b03_{preset_choice}")
            p_b11 = st.slider("B11 (SWIR-1)", 0.0, 1.0, float(default_b[5]), 0.01, key=f"b11_{preset_choice}")
        with c3:
            p_b04 = st.slider("B04 (Red)", 0.0, 1.0, float(default_b[2]), 0.01, key=f"b04_{preset_choice}")
            p_b12 = st.slider("B12 (SWIR-2)", 0.0, 1.0, float(default_b[6]), 0.01, key=f"b12_{preset_choice}")
        with c4:
            p_b08 = st.slider("B08 (NIR lebar)", 0.0, 1.0, float(default_b[3]), 0.01, key=f"b08_{preset_choice}")

    single_bands = np.array([p_b02, p_b03, p_b04, p_b08, p_b8a, p_b11, p_b12], dtype=np.float32)
    single_X, _ = extract_13_features(single_bands)
    single_probs = model.predict_proba(single_X)[0]
    order = np.argsort(single_probs)[::-1]
    single_pred_class = model.classes_[order[0]]
    single_conf = float(single_probs[order[0]])
    margin = float(single_probs[order[0]] - single_probs[order[1]])
    indices_dict = compute_spectral_indices(p_b02, p_b03, p_b04, p_b08, p_b8a, p_b11, p_b12)

    st.markdown("---")
    r1, r2 = st.columns(2)
    with r1:
        info = CLASS_CONFIG[single_pred_class]
        st.markdown(f"""
        <div class="panel" style="border-left: 4px solid {info['color']};">
            <div class="panel-sub" style="text-transform:uppercase;">Kelas prediksi</div>
            <div style="font-size:1.35rem; font-weight:700; color:{info['color']};">{info['name']} ({info['alias']})</div>
            <div style="font-size:0.85rem; color:#f4f4f5; margin-top:4px;">
                Keyakinan {single_conf * 100:.2f}% &nbsp;|&nbsp; selisih terhadap kelas kedua {margin * 100:.2f} poin
            </div>
            <div class="panel-sub" style="margin-top:6px;">{info['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        if margin < 0.30:
            st.warning("Selisih probabilitas dua kelas teratas kecil. Spektrum ini ambigu bagi model.")

        i1, i2, i3 = st.columns(3)
        i1.metric("NDVI", f"{indices_dict['NDVI']:.3f}")
        i1.metric("Mangrove Index", f"{indices_dict['Mangrove_Index_8A_11_4']:.3f}")
        i2.metric("MNDWI", f"{indices_dict['MNDWI']:.3f}")
        i2.metric("EVI", f"{indices_dict['EVI']:.3f}")
        i3.metric("NDBI", f"{indices_dict['NDBI']:.3f}")
        i3.metric("BSI", f"{indices_dict['BSI']:.3f}")

    with r2:
        prob_df = pd.DataFrame({
            "Kelas": [CLASS_CONFIG[c]["name"] for c in VALID_CLASSES],
            "Probabilitas": [float(single_probs[list(model.classes_).index(c)]) * 100 for c in VALID_CLASSES],
        })
        if HAS_PLOTLY:
            fig = px.bar(
                prob_df, x="Probabilitas", y="Kelas", orientation="h", color="Kelas",
                color_discrete_map={CLASS_CONFIG[c]["name"]: CLASS_CONFIG[c]["color"] for c in VALID_CLASSES},
                text=prob_df["Probabilitas"].apply(lambda v: f"{v:.1f}%"),
                title="Probabilitas per kelas (%)",
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="#161920", plot_bgcolor="#161920",
                              height=300, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
            fig.update_xaxes(range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(prob_df.set_index("Kelas"))

# ------------------------------------------------------------------------------
# TAB 3: EVALUASI MODEL
# ------------------------------------------------------------------------------
with tab_eval:
    st.markdown("### Evaluasi Model")
    st.caption("Metrik dibaca dari metrics_evaluation.json. Akurasi piksel acak cenderung optimistis karena "
               "piksel bertetangga memiliki spektrum serupa. Gunakan akurasi spatial holdout sebagai acuan generalisasi.")

    if evaluation_metrics is None:
        st.warning("File metrics_evaluation.json tidak ditemukan. Metrik ditampilkan sebagai N/A.")
        cols = st.columns(4)
        for col, name in zip(cols, ["Akurasi piksel acak", "Akurasi spatial holdout", "Cohen's Kappa", "ROC-AUC makro"]):
            col.metric(name, "N/A")
    else:
        oa = pct(evaluation_metrics.get("overall_accuracy"))
        kappa = evaluation_metrics.get("cohen_kappa")
        roc_auc = pct(evaluation_metrics.get("roc_auc_macro"))
        f1_macro = pct(evaluation_metrics.get("f1_macro"))
        sh = evaluation_metrics.get("spatial_holdout_polygons", {})
        sh_acc = pct(sh.get("consensus_accuracy"))
        sh_total = sh.get("total_polygons", "N/A")
        sh_correct = sh.get("correct_polygons", "N/A")

        cm = evaluation_metrics.get("confusion_matrix")
        n_test = int(np.sum(cm)) if cm is not None else evaluation_metrics.get("total_test_pixels", "N/A")

        n_test_txt = f"{n_test:,}" if isinstance(n_test, int) else str(n_test)
        a1, a2 = st.columns(2)
        with a1:
            val = f"{oa:.2f}%" if oa is not None else "N/A"
            st.markdown(f"""
            <div class="panel" style="border-top: 3px solid #38bdf8;">
                <div class="panel-sub" style="text-transform:uppercase;">Akurasi piksel uji acak</div>
                <div style="font-size:1.8rem; font-weight:700; font-family:'JetBrains Mono', monospace; color:#38bdf8;">{val}</div>
                <div class="panel-sub">{n_test_txt} piksel uji</div>
            </div>
            """, unsafe_allow_html=True)
        with a2:
            val = f"{sh_acc:.2f}%" if sh_acc is not None else "N/A"
            st.markdown(f"""
            <div class="panel" style="border-top: 3px solid #34d399;">
                <div class="panel-sub" style="text-transform:uppercase;">Akurasi spatial holdout</div>
                <div style="font-size:1.8rem; font-weight:700; font-family:'JetBrains Mono', monospace; color:#34d399;">{val}</div>
                <div class="panel-sub">{sh_correct} dari {sh_total} poligon independen benar</div>
            </div>
            """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Cohen's Kappa", f"{kappa:.4f}" if isinstance(kappa, (int, float)) else "N/A")
        m2.metric("ROC-AUC makro", f"{roc_auc:.2f}%" if roc_auc is not None else "N/A")
        m3.metric("F1 makro", f"{f1_macro:.2f}%" if f1_macro is not None else "N/A")

        if cm is not None:
            cm_arr = np.array(cm, dtype=float)
            names = [CLASS_CONFIG[c]["name"] for c in VALID_CLASSES]
            cm_df = pd.DataFrame(cm_arr.astype(int), index=names, columns=names)

            left, right = st.columns([1, 1])
            with left:
                st.markdown("#### Matriks konfusi")
                if HAS_PLOTLY:
                    fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues",
                                       labels=dict(x="Prediksi", y="Referensi", color="Piksel"))
                    fig_cm.update_layout(template="plotly_dark", paper_bgcolor="#161920", plot_bgcolor="#161920",
                                         height=360, margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig_cm, use_container_width=True)
                else:
                    st.dataframe(cm_df, use_container_width=True)
            with right:
                st.markdown("#### Kinerja per kelas")
                tp = np.diag(cm_arr)
                prec = np.divide(tp, cm_arr.sum(axis=0), out=np.zeros_like(tp), where=cm_arr.sum(axis=0) > 0)
                rec = np.divide(tp, cm_arr.sum(axis=1), out=np.zeros_like(tp), where=cm_arr.sum(axis=1) > 0)
                f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
                per_class = pd.DataFrame({
                    "Kelas": names,
                    "Jumlah uji": cm_arr.sum(axis=1).astype(int),
                    "Precision (%)": (prec * 100).round(2),
                    "Recall (%)": (rec * 100).round(2),
                    "F1 (%)": (f1 * 100).round(2),
                })
                st.dataframe(per_class, use_container_width=True, hide_index=True)
                weak = per_class[per_class["F1 (%)"] < 85]["Kelas"].tolist()
                if weak:
                    st.warning("Kelas dengan F1 di bawah 85%: " + ", ".join(weak))