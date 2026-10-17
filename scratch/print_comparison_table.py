# -*- coding: utf-8 -*-
import os, sys, rasterio, numpy as np, pandas as pd
sys.path.insert(0, 'src/Tugas_UTS')
from shared.lulc_features import FEATURE_COLS, extract_13_features

train_df = pd.read_csv('src/Tugas_UTS/3-data_preparation/prepared_data/train_pixels.csv')
tif_path = 'src/Tugas_UTS/2-data_understanding/extracted_data/sentinel2_jatim_scene.tif'
with rasterio.open(tif_path) as src:
    bands = src.read()

X_raster, _ = extract_13_features(bands)

print(f"{'FITUR':24s} | {'--- TRAINING (8,375 px) ---':28s} | {'--- RASTER DEPLOYMENT (1,048,576 px) ---':28s}")
print(f"{' ':24s} | {'min':8s} {'mean':8s} {'max':8s}   | {'min':10s} {'mean':8s} {'max':10s}")
print("-" * 92)

for i, f in enumerate(FEATURE_COLS):
    tr_col = train_df[f].values
    tr_min, tr_mean, tr_max = np.min(tr_col), np.mean(tr_col), np.max(tr_col)
    
    r_col = X_raster[:, i]
    fin = r_col[np.isfinite(r_col)]
    r_min, r_mean, r_max = np.min(fin), np.mean(fin), np.max(fin)
    
    print(f"{f:24s} | {tr_min:8.4f} {tr_mean:8.4f} {tr_max:8.4f}   | {r_min:10.4f} {r_mean:8.4f} {r_max:10.4f}")
