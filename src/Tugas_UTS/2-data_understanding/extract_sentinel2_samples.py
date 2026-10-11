import os
import glob
import time
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from dotenv import load_dotenv
import requests

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)
client_id = os.getenv('CDSE_CLIENT_ID')
client_secret = os.getenv('CDSE_CLIENT_SECRET')

if not client_id or not client_secret:
    raise ValueError("CDSE_CLIENT_ID atau CDSE_CLIENT_SECRET tidak ditemukan di file .env!")

class CDSEClient:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.token_expiry = 0
        self._authenticate()

    def _authenticate(self):
        token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        resp = requests.post(token_url, data=data, timeout=30)
        if resp.status_code == 200:
            res_json = resp.json()
            self.token = res_json['access_token']
            self.token_expiry = time.time() + res_json.get('expires_in', 300) - 30
            print(" Berhasil terhubung & autentikasi ke Copernicus CDSE Sentinel Hub API.")
        else:
            raise RuntimeError(f"Gagal autentikasi CDSE: {resp.text}")

    def get_token(self):
        if time.time() >= self.token_expiry:
            self._authenticate()
        return self.token

    def fetch_sentinel2_bands(self, geometry_geojson, time_from='2024-05-01T00:00:00Z', time_to='2024-09-30T23:59:59Z'):
        token = self.get_token()
        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: [{
              bands: ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"],
              units: "REFLECTANCE"
            }],
            output: {
              id: "default",
              bands: 7,
              sampleType: "FLOAT32"
            }
          };
        }

        function evaluatePixel(sample) {
          return [sample.B02, sample.B03, sample.B04, sample.B08, sample.B8A, sample.B11, sample.B12];
        }
        """
        process_url = 'https://sh.dataspace.copernicus.eu/api/v1/process'
        payload = {
            'input': {
                'bounds': {
                    'geometry': geometry_geojson,
                    'properties': {'crs': 'http://www.opengis.net/def/crs/EPSG/0/4326'}
                },
                'data': [{
                    'type': 'sentinel-2-l2a',
                    'dataFilter': {
                        'timeRange': {
                            'from': time_from,
                            'to': time_to
                        },
                        'maxCloudCoverage': 20
                    }
                }]
            },
            'output': {
                'responses': [{
                    'identifier': 'default',
                    'format': {'type': 'image/tiff'}
                }]
            },
            'evalscript': evalscript
        }
        headers = {
            'Authorization': f'Bearer {token}',
            'Accept': 'image/tiff'
        }

        for attempt in range(3):
            try:
                resp = requests.post(process_url, json=payload, headers=headers, timeout=60)
                if resp.status_code == 200:
                    return resp.content
                elif resp.status_code in [429, 502, 503, 504]:
                    time.sleep(2 ** attempt + 1)
                else:
                    print(f" Warning ({resp.status_code}): {resp.text[:120]}")
                    return None
            except Exception as e:
                time.sleep(2)
        return None

def process_all_samples():
    base_dir = os.path.dirname(__file__)
    geojson_dir = os.path.join(base_dir, 'sampling_geojson')
    output_dir = os.path.join(base_dir, 'extracted_data')
    os.makedirs(output_dir, exist_ok=True)

    client = CDSEClient(client_id, client_secret)

    class_files = {
        'bangunan': os.path.join(geojson_dir, 'bangunan.geojson'),
        'hutan': os.path.join(geojson_dir, 'hutan.geojson'),
        'mangrove': os.path.join(geojson_dir, 'mangrove.json'),
        'perairan': os.path.join(geojson_dir, 'perairan.geojson'),
        'pertanian': os.path.join(geojson_dir, 'pertanian.geojson')
    }

    class_label_map = {
        'bangunan': 0,
        'pertanian': 1,
        'perairan': 2,
        'hutan': 3,
        'mangrove': 4
    }

    band_names = ['B02', 'B03', 'B04', 'B08', 'B8A', 'B11', 'B12']
    polygon_records = []
    pixel_records = []

    total_classes = len(class_files)
    print(f"\n Memulai ekstraksi data citra satelit Sentinel-2A untuk 5 kelas...")

    for class_idx, (class_name, filepath) in enumerate(class_files.items(), 1):
        if not os.path.exists(filepath):
            print(f" File {filepath} tidak ditemukan!")
            continue

        gdf = gpd.read_file(filepath)
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')

        print(f"\n[{class_idx}/{total_classes}] Memproses kelas: {class_name.upper()} ({len(gdf)} sampel poligon)...")

        for i, row in gdf.iterrows():
            geom = row.geometry
            geom_json = geom.__geo_interface__

            tiff_bytes = client.fetch_sentinel2_bands(geom_json)
            if tiff_bytes is None:
                print(f"   Sampel #{i+1} gagal diambil dari CDSE, melewatinya...")
                continue

            try:
                with MemoryFile(tiff_bytes) as memfile:
                    with memfile.open() as src:
                        # Masking array strictly to the polygon geometry
                        masked_image, _ = mask(src, [geom], crop=True, nodata=np.nan)
                        
                        # Valid pixels across all 7 bands
                        valid_mask = ~np.isnan(masked_image).any(axis=0)
                        
                        if np.sum(valid_mask) == 0:
                            # Fallback if masking produced all NaNs
                            pixels_data = masked_image.reshape(7, -1)
                            valid_cols = ~np.isnan(pixels_data).any(axis=0)
                            if np.sum(valid_cols) > 0:
                                pixels = pixels_data[:, valid_cols]
                            else:
                                pixels = src.read().reshape(7, -1)
                        else:
                            pixels = masked_image[:, valid_mask]

                        # Calculate polygon summary stats
                        mean_bands = np.nanmean(pixels, axis=1)
                        median_bands = np.nanmedian(pixels, axis=1)
                        std_bands = np.nanstd(pixels, axis=1)

                        b2, b3, b4, b8, b8a, b11, b12 = mean_bands

                        # Calculate spectral indices
                        ndvi = (b8 - b4) / (b8 + b4 + 1e-8)
                        mndwi = (b3 - b11) / (b3 + b11 + 1e-8)
                        ndbi = (b11 - b8) / (b11 + b8 + 1e-8)
                        b8a_11_ratio = b8a / (b11 + 1e-8)
                        mvi_mangrove = (b8a - b4) / (b11 + 1e-8)

                        polygon_records.append({
                            'sample_id': f"{class_name}_{i+1:03d}",
                            'class_name': class_name,
                            'class_id': class_label_map[class_name],
                            'B02': b2,
                            'B03': b3,
                            'B04': b4,
                            'B08': b8,
                            'B8A': b8a,
                            'B11': b11,
                            'B12': b12,
                            'NDVI': ndvi,
                            'MNDWI': mndwi,
                            'NDBI': ndbi,
                            'B8A_B11_Ratio': b8a_11_ratio,
                            'Mangrove_Index_8A_11_4': mvi_mangrove,
                            'centroid_lon': geom.centroid.x,
                            'centroid_lat': geom.centroid.y,
                            'pixel_count': pixels.shape[1]
                        })

                        # Sample up to 50 representative pixels per polygon for high-res dataset
                        n_subsample = min(pixels.shape[1], 50)
                        chosen_indices = np.random.choice(pixels.shape[1], n_subsample, replace=False)
                        sub_pixels = pixels[:, chosen_indices]

                        for p_idx in range(n_subsample):
                            pb2, pb3, pb4, pb8, pb8a, pb11, pb12 = sub_pixels[:, p_idx]
                            p_ndvi = (pb8 - pb4) / (pb8 + pb4 + 1e-8)
                            p_mndwi = (pb3 - pb11) / (pb3 + pb11 + 1e-8)
                            p_ndbi = (pb11 - pb8) / (pb11 + pb8 + 1e-8)
                            p_b8a_11_ratio = pb8a / (pb11 + 1e-8)
                            p_mvi = (pb8a - pb4) / (pb11 + 1e-8)

                            pixel_records.append({
                                'sample_id': f"{class_name}_{i+1:03d}",
                                'class_name': class_name,
                                'class_id': class_label_map[class_name],
                                'B02': pb2,
                                'B03': pb3,
                                'B04': pb4,
                                'B08': pb8,
                                'B8A': pb8a,
                                'B11': pb11,
                                'B12': pb12,
                                'NDVI': p_ndvi,
                                'MNDWI': p_mndwi,
                                'NDBI': p_ndbi,
                                'B8A_B11_Ratio': p_b8a_11_ratio,
                                'Mangrove_Index_8A_11_4': p_mvi
                            })

            except Exception as e:
                print(f"   Error saat memproses raster sampel #{i+1}: {e}")

            # Small sleep to be polite to API rate limits
            time.sleep(0.1)

        print(f" Berhasil mengekstrak kelas {class_name.upper()}.")

    df_poly = pd.DataFrame(polygon_records)
    df_pixels = pd.DataFrame(pixel_records)

    poly_csv = os.path.join(output_dir, 'sentinel2_sample_polygons.csv')
    pixels_csv = os.path.join(output_dir, 'sentinel2_pixels_dataset.csv')

    df_poly.to_csv(poly_csv, index=False)
    df_pixels.to_csv(pixels_csv, index=False)

    print("\n" + "="*60)
    print(" EKSTRAKSI DATA CITRA SENTINEL-2 SELESAI DENGAN SUKSES!")
    print(f" 1. Dataset Ringkasan Sampel Poligon: {poly_csv} ({len(df_poly)} baris)")
    print(f" 2. Dataset Spektral Pixel-Level     : {pixels_csv} ({len(df_pixels)} baris)")
    print("="*60)
    print("\nDistribusi Sampel per Kelas:")
    print(df_poly['class_name'].value_counts())
    print("\nContoh 5 baris teratas:")
    print(df_poly[['sample_id', 'class_name', 'B04', 'B8A', 'B11', 'NDVI', 'Mangrove_Index_8A_11_4']].head())

if __name__ == '__main__':
    process_all_samples()
