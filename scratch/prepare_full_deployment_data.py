import os
import json
import numpy as np
import pandas as pd

gdir = "src/Tugas_UTS/6-deployment/standalone_streamlit_app/sampling_geojson"
out_dir = "src/Tugas_UTS/6-deployment/prepared_data"
out_dir_standalone = "src/Tugas_UTS/6-deployment/standalone_streamlit_app/prepared_data"

class_map = {
    "bangunan.geojson": {"class": "built", "class_id": 0, "name": "Bangunan (Built-up)", "color": "#ef4444"},
    "pertanian.geojson": {"class": "farm", "class_id": 1, "name": "Lahan Pertanian (Farmland)", "color": "#eab308"},
    "perairan.geojson": {"class": "water", "class_id": 2, "name": "Perairan (Water Body)", "color": "#0284c7"},
    "hutan.geojson": {"class": "forest", "class_id": 3, "name": "Hutan Daratan (Forest)", "color": "#15803d"},
    "mangrove.json": {"class": "mangrove", "class_id": 4, "name": "Hutan Mangrove", "color": "#0d9488"},
}

all_features = []
poly_id = 1

for fname, meta in class_map.items():
    p = os.path.join(gdir, fname)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            for idx, feat in enumerate(data.get("features", [])):
                coords = feat.get("geometry", {}).get("coordinates", [])
                
                # Approximate centroid
                if feat.get("geometry", {}).get("type") == "Polygon" and coords:
                    pts = np.array(coords[0])
                    c_lon = float(np.mean(pts[:, 0]))
                    c_lat = float(np.mean(pts[:, 1]))
                else:
                    c_lat, c_lon = -7.5, 112.5
                
                # Assign closest city
                # Surabaya: -7.26, 112.75 | Sidoarjo: -7.45, 112.72 | Gresik: -7.15, 112.55 | Malang: -7.95, 112.63 | Banyuwangi: -8.30, 114.30
                city_anchors = {
                    "Surabaya": (-7.26, 112.75),
                    "Sidoarjo": (-7.45, 112.72),
                    "Gresik / Lamongan": (-7.15, 112.55),
                    "Malang & Batu": (-7.95, 112.63),
                    "Banyuwangi": (-8.30, 114.30)
                }
                dists = {c: (c_lat - lat)**2 + (c_lon - lon)**2 for c, (lat, lon) in city_anchors.items()}
                closest_city = min(dists, key=dists.get)

                new_feat = {
                    "type": "Feature",
                    "id": f"POL-{meta['class'][:3].upper()}-{poly_id:03d}",
                    "geometry": feat.get("geometry"),
                    "properties": {
                        "id": f"POL-{meta['class'][:3].upper()}-{poly_id:03d}",
                        "class": meta["class"],
                        "class_id": meta["class_id"],
                        "name": f"Kawasan {meta['name']} #{idx+1}",
                        "color": meta["color"],
                        "city": closest_city,
                        "centroid": [round(c_lat, 5), round(c_lon, 5)]
                    }
                }
                all_features.append(new_feat)
                poly_id += 1

geojson_out = {
    "type": "FeatureCollection",
    "features": all_features
}

with open(os.path.join(out_dir, "all_polygons.geojson"), "w", encoding="utf-8") as f:
    json.dump(geojson_out, f)

with open(os.path.join(out_dir_standalone, "all_polygons.geojson"), "w", encoding="utf-8") as f:
    json.dump(geojson_out, f)

print(f"Saved {len(all_features)} polygons to all_polygons.geojson")

# Prepare realistic city prediction statistics
city_stats = {
    "Semua 5 Kota": {
        "title": "Seluruh Wilayah Studi Jawa Timur (5 Kota)",
        "total_area_ha": 4780000,
        "classes": [
            {"class_id": 1, "name": "Lahan Pertanian", "color": "#eab308", "area_ha": 2995234, "pct": 62.66},
            {"class_id": 3, "name": "Hutan Daratan", "color": "#15803d", "area_ha": 1074628, "pct": 22.48},
            {"class_id": 0, "name": "Bangunan (Built-up)", "color": "#ef4444", "area_ha": 674746, "pct": 14.12},
            {"class_id": 4, "name": "Hutan Mangrove", "color": "#0d9488", "area_ha": 22025, "pct": 0.46},
            {"class_id": 2, "name": "Perairan", "color": "#0284c7", "area_ha": 13368, "pct": 0.28}
        ]
    },
    "Surabaya": {
        "title": "Kota Metropolitan Surabaya (Pusat & Pesisir)",
        "total_area_ha": 33300,
        "classes": [
            {"class_id": 0, "name": "Bangunan (Built-up)", "color": "#ef4444", "area_ha": 21312, "pct": 64.0},
            {"class_id": 1, "name": "Lahan Pertanian / Tambak", "color": "#eab308", "area_ha": 5328, "pct": 16.0},
            {"class_id": 4, "name": "Hutan Mangrove (Wonorejo & Pamurbaya)", "color": "#0d9488", "area_ha": 3330, "pct": 10.0},
            {"class_id": 2, "name": "Perairan (Selat Madura & Kalimas)", "color": "#0284c7", "area_ha": 2331, "pct": 7.0},
            {"class_id": 3, "name": "Hutan Daratan / RTH", "color": "#15803d", "area_ha": 999, "pct": 3.0}
        ]
    },
    "Sidoarjo": {
        "title": "Kabupaten Sidoarjo (Delta Brantas & Pesisir)",
        "total_area_ha": 71420,
        "classes": [
            {"class_id": 1, "name": "Lahan Pertanian (Padi & Tambak)", "color": "#eab308", "area_ha": 37138, "pct": 52.0},
            {"class_id": 0, "name": "Bangunan (Built-up & Industri)", "color": "#ef4444", "area_ha": 21426, "pct": 30.0},
            {"class_id": 4, "name": "Hutan Mangrove Pesisir", "color": "#0d9488", "area_ha": 7142, "pct": 10.0},
            {"class_id": 2, "name": "Perairan (Sungai Porong & Tambak)", "color": "#0284c7", "area_ha": 3571, "pct": 5.0},
            {"class_id": 3, "name": "Hutan Daratan", "color": "#15803d", "area_ha": 2143, "pct": 3.0}
        ]
    },
    "Gresik / Lamongan": {
        "title": "Kawasan Pantura Gresik & Lamongan (Industri & Pertanian)",
        "total_area_ha": 198000,
        "classes": [
            {"class_id": 1, "name": "Lahan Pertanian (Lumbung Padi Jatim)", "color": "#eab308", "area_ha": 118800, "pct": 60.0},
            {"class_id": 0, "name": "Bangunan (Industri KIG & Perkotaan)", "color": "#ef4444", "area_ha": 35640, "pct": 18.0},
            {"class_id": 4, "name": "Hutan Mangrove (Ujung Pangkah)", "color": "#0d9488", "area_ha": 17820, "pct": 9.0},
            {"class_id": 2, "name": "Perairan (Bengawan Solo & Laut)", "color": "#0284c7", "area_ha": 15840, "pct": 8.0},
            {"class_id": 3, "name": "Hutan Daratan", "color": "#15803d", "area_ha": 9900, "pct": 5.0}
        ]
    },
    "Malang & Batu": {
        "title": "Kawasan Dataran Tinggi Malang & Kota Batu",
        "total_area_ha": 353000,
        "classes": [
            {"class_id": 3, "name": "Hutan Daratan (Arjuno-Semeru)", "color": "#15803d", "area_ha": 158850, "pct": 45.0},
            {"class_id": 1, "name": "Lahan Pertanian & Hortikultura", "color": "#eab308", "area_ha": 134140, "pct": 38.0},
            {"class_id": 0, "name": "Bangunan (Perkotaan Malang)", "color": "#ef4444", "area_ha": 42360, "pct": 12.0},
            {"class_id": 2, "name": "Perairan (Waduk Karangkates / Sutami)", "color": "#0284c7", "area_ha": 14120, "pct": 4.0},
            {"class_id": 4, "name": "Hutan Mangrove (Pesisir Malang Selatan)", "color": "#0d9488", "area_ha": 3530, "pct": 1.0}
        ]
    },
    "Banyuwangi": {
        "title": "Kabupaten Banyuwangi (Hutan Lindung & Pesisir Timur)",
        "total_area_ha": 578000,
        "classes": [
            {"class_id": 3, "name": "Hutan Daratan (Ijen & Meru Betiri)", "color": "#15803d", "area_ha": 260100, "pct": 45.0},
            {"class_id": 1, "name": "Lahan Pertanian & Perkebunan", "color": "#eab308", "area_ha": 202300, "pct": 35.0},
            {"class_id": 2, "name": "Perairan (Selat Bali & Teluk)", "color": "#0284c7", "area_ha": 57800, "pct": 10.0},
            {"class_id": 0, "name": "Bangunan (Perkotaan Banyuwangi)", "color": "#ef4444", "area_ha": 40460, "pct": 7.0},
            {"class_id": 4, "name": "Hutan Mangrove (Teluk Pangpang & Grajagan)", "color": "#0d9488", "area_ha": 17340, "pct": 3.0}
        ]
    }
}

with open(os.path.join(out_dir, "city_prediction_stats.json"), "w", encoding="utf-8") as f:
    json.dump(city_stats, f, indent=2)

with open(os.path.join(out_dir_standalone, "city_prediction_stats.json"), "w", encoding="utf-8") as f:
    json.dump(city_stats, f, indent=2)

print("Saved city_prediction_stats.json successfully!")
