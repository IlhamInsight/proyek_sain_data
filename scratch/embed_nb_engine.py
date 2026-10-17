# -*- coding: utf-8 -*-
import json

path = 'src/Tugas_UTS/6-deployment/6-deployment.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_1_code = """import os
import sys
import time
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns

BASE_DIR = os.path.abspath('')
UTS_DIR = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == '6-deployment' else os.path.join(BASE_DIR, 'src', 'Tugas_UTS')

BAND_NAMES = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
INDEX_NAMES = ["NDVI", "MNDWI", "NDBI", "Mangrove_Index_8A_11_4", "EVI", "BSI"]
FEATURE_COLS = BAND_NAMES + INDEX_NAMES

CLASS_CONFIG = {
    0: {"key": "built", "name": "Bangunan", "alias": "Built-up", "color": "#ef4444", "rgb": (239, 68, 68), "desc": "Permukiman, industri, jalan, dan perkerasan beton."},
    1: {"key": "farm", "name": "Lahan Pertanian", "alias": "Farmland", "color": "#eab308", "rgb": (234, 179, 8), "desc": "Sawah irigasi, tegalan, ladang semusim, dan hortikultura."},
    2: {"key": "water", "name": "Perairan", "alias": "Water Body", "color": "#0284c7", "rgb": (2, 132, 199), "desc": "Selat Madura, laut pesisir, muara sungai, dan waduk."},
    3: {"key": "forest", "name": "Hutan Daratan", "alias": "Terrestrial Forest", "color": "#22c55e", "rgb": (34, 197, 94), "desc": "Hutan alami dan tutupan kanopi lebat dataran tinggi."},
    4: {"key": "mangrove", "name": "Hutan Mangrove", "alias": "Mangrove Forest", "color": "#06b6d4", "rgb": (6, 182, 212), "desc": "Vegetasi bakau pasang-surut pantai dan muara berlumpur."},
    255: {"key": "nodata", "name": "Unknown / Low Confidence", "alias": "Unknown/NoData", "color": "#000000", "rgb": (0, 0, 0), "desc": "Piksel NoData, awan/saturasi, atau di bawah ambang keyakinan."}
}
VALID_CLASSES = [0, 1, 2, 3, 4]

REGIONS_AOI = {
    "Seluruh Citra Scene (Surabaya - Sidoarjo - Gresik)": {
        "bounds": (112.50, -7.60, 113.00, -7.00), "center": (-7.30, 112.75), "zoom": 10,
        "title": "Scene Sentinel-2 Jawa Timur (Penuh)",
        "desc": "Wilayah cakupan penuh citra Sentinel-2 mencakup Surabaya, Sidoarjo, dan pesisir Gresik."
    },
    "Surabaya (Metropolitan & Wonorejo)": {
        "bounds": (112.60, -7.36, 112.85, -7.18), "center": (-7.26, 112.75), "zoom": 12,
        "title": "Kota Surabaya (Urban & Pesisir Timur)",
        "desc": "Pusat bisnis perkotaan, kawasan industri, pelabuhan, dan mangrove pesisir Wonorejo."
    },
    "Malang & Batu (Dataran Tinggi & Pegunungan)": {
        "bounds": (112.45, -8.05, 112.75, -7.80), "center": (-7.95, 112.63), "zoom": 11,
        "title": "Kawasan Dataran Tinggi Malang & Kota Batu",
        "desc": "Pegunungan Arjuno-Semeru dan hortikultura (di luar cakupan scene citra ini)."
    }
}

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
    raise ValueError(f"Bentuk bands_array tidak valid: {arr.shape}")

def validate_model(mdl):
    if not hasattr(mdl, "classes_") or list(mdl.classes_) != VALID_CLASSES:
        raise ValueError(f"Kelas model tidak sesuai! Ekspektasi {VALID_CLASSES}")
    if not hasattr(mdl, "n_features_in_") or mdl.n_features_in_ != len(FEATURE_COLS):
        raise ValueError(f"Jumlah fitur tidak sesuai! Ekspektasi {len(FEATURE_COLS)}")
    if not hasattr(mdl, "predict_proba"):
        raise ValueError("Model tidak mendukung fungsi predict_proba!")
    return True

def check_raster_coverage(raster_bounds, aoi_bounds):
    r_left, r_bottom, r_right, r_top = raster_bounds
    aoi_left, aoi_bottom, aoi_right, aoi_top = aoi_bounds
    if aoi_left >= r_right or aoi_right <= r_left or aoi_bottom >= r_top or aoi_top <= r_bottom:
        return False, None, "Wilayah tidak tercakup oleh citra Sentinel-2 yang tersedia."
    inter = (max(r_left, aoi_left), max(r_bottom, aoi_bottom), min(r_right, aoi_right), min(r_top, aoi_top))
    if inter[0] >= inter[2] or inter[1] >= inter[3]:
        return False, None, "Wilayah tidak tercakup oleh citra Sentinel-2 yang tersedia."
    return True, inter, "Wilayah berhasil tercakup oleh citra Sentinel-2."

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

def run_full_raster_inference(mdl, bands, confidence_threshold=0.65):
    validate_model(mdl)
    t0 = time.time()
    _, h, w = bands.shape
    X, _ = extract_13_features(bands)
    invalid_mask = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | (X[:, :7] <= 0.0).all(axis=1)
    X_clean = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    probs = mdl.predict_proba(X_clean)
    max_idx = np.argmax(probs, axis=1)
    max_prob = np.max(probs, axis=1)
    preds = mdl.classes_[max_idx].copy()
    preds[max_prob < confidence_threshold] = 255
    preds[invalid_mask] = 255
    max_prob[invalid_mask] = 0.0
    cls_raster = preds.reshape(h, w).astype(np.uint8)
    conf_raster = max_prob.reshape(h, w).astype(np.float32)
    
    unique, counts = np.unique(cls_raster, return_counts=True)
    count_dict = dict(zip(unique.tolist(), counts.tolist()))
    class_stats = {}
    total_pixels = h * w
    for c in [0, 1, 2, 3, 4, 255]:
        cnt = count_dict.get(c, 0)
        class_stats[c] = {
            "name": CLASS_CONFIG[c]["name"],
            "alias": CLASS_CONFIG[c]["alias"],
            "color": CLASS_CONFIG[c]["color"],
            "count": cnt,
            "percent": (cnt / total_pixels) * 100.0
        }
        
    return {
        "classified_raster": cls_raster,
        "confidence_raster": conf_raster,
        "class_stats": class_stats,
        "inference_time_sec": time.time() - t0,
        "total_pixels": total_pixels,
        "shape": (h, w)
    }

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

print('[OK] Modul inferensi geospasial tahap Deployment berhasil diinisialisasi.')
"""

# Update cell 1
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('compute_spectral_indices' in line for line in cell.get('source', [])):
        cell['source'] = [line + '\n' for line in cell_1_code.split('\n')]
        break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated 6-deployment.ipynb with complete self-contained engine!')
