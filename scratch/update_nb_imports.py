# -*- coding: utf-8 -*-
import json

path = 'src/Tugas_UTS/6-deployment/6-deployment.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('from lulc_engine' in line for line in cell.get('source', [])):
        cell['source'] = [
            "import os\n",
            "import sys\n",
            "import time\n",
            "import json\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import rasterio\n",
            "import joblib\n",
            "import matplotlib\n",
            "matplotlib.use('Agg')\n",
            "import matplotlib.pyplot as plt\n",
            "from matplotlib.colors import ListedColormap, BoundaryNorm\n",
            "import seaborn as sns\n",
            "\n",
            "# Pastikan modul deployment engine dapat diimpor secara lokal\n",
            "BASE_DIR = os.path.abspath('')\n",
            "UTS_DIR = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == '6-deployment' else os.path.join(BASE_DIR, 'src', 'Tugas_UTS')\n",
            "if BASE_DIR not in sys.path:\n",
            "    sys.path.insert(0, BASE_DIR)\n",
            "\n",
            "from lulc_engine import (\n",
            "    FEATURE_COLS, BAND_NAMES, CLASS_CONFIG, VALID_CLASSES,\n",
            "    REGIONS_AOI, compute_spectral_indices, extract_13_features, validate_model,\n",
            "    check_raster_coverage, read_sentinel2_raster,\n",
            "    run_full_raster_inference, save_inference_geotiffs\n",
            ")\n",
            "\n",
            "print('[OK] Modul geospasial dan dependensi tahap Deployment berhasil dimuat.')\n"
        ]
        break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated 6-deployment.ipynb with pandas successfully!')
