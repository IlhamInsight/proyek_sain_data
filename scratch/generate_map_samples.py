import os
import json
import pandas as pd

# Load the real sample data
df_samples = pd.read_csv("src/Tugas_UTS/3-data_preparation/prepared_data/data_prepared_all.csv")
print(f"Total samples loaded: {len(df_samples)}")

sample_records = []
for idx, r in df_samples.iterrows():
    cid = int(r.get("class_id", 0))
    cat_keys = {0: "built", 1: "farm", 2: "water", 3: "forest", 4: "mangrove"}
    cat = cat_keys.get(cid, "built")
    
    b2 = float(r.get("B02", 0.05))
    b3 = float(r.get("B03", 0.08))
    b4 = float(r.get("B04", 0.06))
    b8 = float(r.get("B08", 0.35))
    b11 = float(r.get("B11", 0.15))
    b12 = float(r.get("B12", 0.08))
    
    spec = [
        min(100, max(5, int(b2 * 250))),
        min(100, max(5, int(b3 * 250))),
        min(100, max(5, int(b4 * 250))),
        min(100, max(5, int(b8 * 250))),
        min(100, max(5, int(b11 * 250))),
        min(100, max(5, int(b12 * 250)))
    ]
    
    conf = round(float(r.get("pred_confidence", 0.95)) * 100, 1)
    sid = str(r.get("sample_id", f"{idx:03d}"))
    if not sid.startswith("JTM"):
        sid = f"JTM-{cat[:3].upper()}-{idx+1:03d}"
    
    rec = {
        "id": sid,
        "class": cat,
        "class_id": cid,
        "name": str(r.get("place_name", f"Titik #{idx+1}")),
        "city": str(r.get("city", "Jawa Timur")),
        "coords": [round(float(r["centroid_lat"]), 5), round(float(r["centroid_lon"]), 5)],
        "conf": conf,
        "ndvi": round(float(r.get("NDVI", 0.0)), 2),
        "ndwi": round(float(r.get("MNDWI", 0.0)), 2),
        "ndbi": round(float(r.get("NDBI", 0.0)), 2),
        "mangrove": round(float(r.get("Mangrove_Index_8A_11_4", 0.0)), 2),
        "evi": round(float(r.get("EVI", 0.0)), 2),
        "bsi": round(float(r.get("BSI", 0.0)), 2),
        "spec": spec
    }
    sample_records.append(rec)

print(f"Generated {len(sample_records)} map sample records.")

# Save to deployment folder
os.makedirs("src/Tugas_UTS/6-deployment/prepared_data", exist_ok=True)
with open("src/Tugas_UTS/6-deployment/prepared_data/map_samples.json", "w", encoding="utf-8") as f:
    json.dump(sample_records, f, indent=2)

os.makedirs("src/Tugas_UTS/6-deployment/standalone_streamlit_app/prepared_data", exist_ok=True)
with open("src/Tugas_UTS/6-deployment/standalone_streamlit_app/prepared_data/map_samples.json", "w", encoding="utf-8") as f:
    json.dump(sample_records, f, indent=2)

print("Saved map_samples.json successfully!")
