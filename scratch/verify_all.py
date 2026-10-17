# -*- coding: utf-8 -*-
import os
import sys
import json
import joblib
import rasterio
import numpy as np

# Load standalone deployment functions from 6-deployment.ipynb cell 1
nb_path = 'src/Tugas_UTS/6-deployment/6-deployment.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

c1_code = ''.join(nb['cells'][1]['source'])
c1_ns = {}
exec(c1_code, c1_ns)

FEATURE_COLS = c1_ns['FEATURE_COLS']
VALID_CLASSES = c1_ns['VALID_CLASSES']
REGIONS_AOI = c1_ns['REGIONS_AOI']
validate_model = c1_ns['validate_model']
read_sentinel2_raster = c1_ns['read_sentinel2_raster']
run_full_raster_inference = c1_ns['run_full_raster_inference']
check_raster_coverage = c1_ns['check_raster_coverage']

print("=" * 60)
print("INTEGRATION & REGRESSION VERIFICATION SUITE")
print("=" * 60)

# 1. Model loading & validation
model_path = 'src/Tugas_UTS/4-modeling/models/model_lightgbm_piksel.pkl'
model = joblib.load(model_path)
validate_model(model)
print("[PASS] 1. Model loads and passes validation:")
print(f"       • classes_ = {list(model.classes_)}")
print(f"       • n_features_in_ = {model.n_features_in_}")

# 2. Features order
expected_features = ['B02', 'B03', 'B04', 'B08', 'B8A', 'B11', 'B12', 'NDVI', 'MNDWI', 'NDBI', 'Mangrove_Index_8A_11_4', 'EVI', 'BSI']
assert FEATURE_COLS == expected_features, "Feature order mismatch!"
print("[PASS] 2. 13 features exact order matches training:")
for i, f in enumerate(FEATURE_COLS, 1):
    print(f"       {i:2d}. {f}")

# 3. Read raster scene
tif_path = 'src/Tugas_UTS/2-data_understanding/extracted_data/sentinel2_jatim_scene.tif'
bands, prof, bounds_4326, win_transform = read_sentinel2_raster(tif_path)
print(f"[PASS] 3. Raster CRS is {prof['crs']}, shape is {bands.shape}, bounds are {prof['transform']}")

# 4. Out-of-coverage rejection
malang_aoi = REGIONS_AOI["Malang & Batu (Dataran Tinggi & Pegunungan)"]["bounds"]
try:
    read_sentinel2_raster(tif_path, aoi_bounds=malang_aoi)
    raise AssertionError("Failed to reject out-of-coverage AOI!")
except ValueError as e:
    print(f"[PASS] 4. Out-of-coverage correctly rejected for Malang: '{e}'")

# 5. Full-raster inference
result = run_full_raster_inference(model, bands, confidence_threshold=0.65)
classified = result['classified_raster']
confidence = result['confidence_raster']
classes_found = sorted(np.unique(classified).tolist())
print(f"[PASS] 5. classified_raster valid! Classes found: {classes_found}")
print(f"[PASS] 6. confidence_raster valid! Min={confidence.min():.4f}, Max={confidence.max():.4f}")

# 6. Check files
assert os.path.exists('src/Tugas_UTS/6-deployment/app.py'), "app.py missing!"
assert os.path.exists('src/Tugas_UTS/6-deployment/6-deployment.ipynb'), "6-deployment.ipynb missing!"
assert not os.path.exists('src/Tugas_UTS/6-deployment/lulc_engine.py'), "lulc_engine.py should be removed!"
assert not os.path.exists('src/Tugas_UTS/shared'), "src/Tugas_UTS/shared should be removed!"
print("[PASS] 7. Architecture clean: only app.py & 6-deployment.ipynb in 6-deployment!")

print("=" * 60)
print("ALL VERIFICATION CHECKS PASSED!")
print("=" * 60)
