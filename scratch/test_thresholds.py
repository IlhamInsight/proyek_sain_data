# -*- coding: utf-8 -*-
import sys, joblib, rasterio
sys.path.insert(0, 'src/Tugas_UTS')
from shared.lulc_inference import run_full_raster_inference

model = joblib.load('src/Tugas_UTS/4-modeling/models/model_lightgbm_piksel.pkl')
with rasterio.open('src/Tugas_UTS/2-data_understanding/extracted_data/sentinel2_jatim_scene.tif') as src:
    bands = src.read()

res_40 = run_full_raster_inference(model, bands, confidence_threshold=0.40)
res_65 = run_full_raster_inference(model, bands, confidence_threshold=0.65)
res_70 = run_full_raster_inference(model, bands, confidence_threshold=0.70)

print("--- THRESHOLD 40% ---")
for c, s in res_40['class_stats'].items():
    print(f"Class {c:3d} ({s['name']:24s}): {s['percent']:6.2f}% ({s['count']} px)")

print("\n--- THRESHOLD 65% ---")
for c, s in res_65['class_stats'].items():
    print(f"Class {c:3d} ({s['name']:24s}): {s['percent']:6.2f}% ({s['count']} px)")

print("\n--- THRESHOLD 70% ---")
for c, s in res_70['class_stats'].items():
    print(f"Class {c:3d} ({s['name']:24s}): {s['percent']:6.2f}% ({s['count']} px)")
