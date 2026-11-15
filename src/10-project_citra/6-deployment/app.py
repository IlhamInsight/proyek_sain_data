import io
import time
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

app = FastAPI(
    title="Footwear Tidiness Classifier API",
    description="API Klasifikasi Kerapian Sandal dan Sepatu (Alas Kaki) Berbasis Computer Vision",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ["Tidak Rapi (Untidy)", "Rapi (Tidy)"]

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sistem Pemantau Kerapian Alas Kaki</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; }
            .container { max-width: 650px; margin: 0 auto; background: #1e293b; padding: 2rem; border-radius: 12px; border: 1px solid #334155; }
            h1 { font-size: 1.5rem; margin-bottom: 0.5rem; color: #38bdf8; }
            p { color: #94a3b8; font-size: 0.95rem; }
            .dropzone { border: 2px dashed #475569; padding: 2rem; text-align: center; border-radius: 8px; margin: 1.5rem 0; cursor: pointer; }
            .btn { background: #0284c7; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 6px; font-weight: 600; cursor: pointer; }
            .btn:hover { background: #0369a1; }
            #preview { max-width: 100%; max-height: 250px; margin-top: 1rem; border-radius: 6px; display: none; }
            #result { margin-top: 1.5rem; padding: 1rem; border-radius: 8px; background: #0f172a; display: none; border: 1px solid #334155; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Deteksi Kerapian Sandal & Sepatu</h1>
            <p>Unggah foto rak atau lantai alas kaki untuk memverifikasi tingkat kerapian secara otomatis.</p>
            <input type="file" id="imageInput" accept="image/*" style="display: none;">
            <div class="dropzone" onclick="document.getElementById('imageInput').click()">
                Klik di sini untuk memilih gambar sandal/sepatu
            </div>
            <center><img id="preview"></center>
            <center><button class="btn" id="analyzeBtn" onclick="analyzeImage()" style="display: none;">Analisis Kerapian</button></center>
            <div id="result"></div>
        </div>
        <script>
            const input = document.getElementById('imageInput');
            const preview = document.getElementById('preview');
            const analyzeBtn = document.getElementById('analyzeBtn');
            const resultDiv = document.getElementById('result');

            input.onchange = e => {
                const [file] = input.files;
                if (file) {
                    preview.src = URL.createObjectURL(file);
                    preview.style.display = 'block';
                    analyzeBtn.style.display = 'inline-block';
                    resultDiv.style.display = 'none';
                }
            };

            async function analyzeImage() {
                const file = input.files[0];
                if (!file) return;
                const formData = new FormData();
                formData.append('file', file);
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = 'Memproses analisis visi komputer...';

                try {
                    const res = await fetch('/api/v1/predict', { method: 'POST', body: formData });
                    const data = await res.json();
                    if (res.ok) {
                        const statusColor = data.predicted_class.includes('Rapi') && !data.predicted_class.includes('Tidak') ? '#4ade80' : '#f87171';
                        resultDiv.innerHTML = `
                            <h3 style="color: ${statusColor}; margin-top: 0;">Hasil: ${data.predicted_class}</h3>
                            <p><strong>Tingkat Keyakinan:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
                            <p><strong>Waktu Inferensi:</strong> ${data.inference_latency_ms.toFixed(1)} ms</p>
                        `;
                    } else {
                        resultDiv.innerHTML = `<span style="color: #ef4444;">Error: ${data.detail || 'Gagal memproses gambar'}</span>`;
                    }
                } catch(err) {
                    resultDiv.innerHTML = `<span style="color: #ef4444;">Error: ${err.message}</span>`;
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "footwear-tidiness-classifier",
        "timestamp": time.time()
    }

@app.post("/api/v1/predict")
async def predict_image(file: UploadFile = File(...)):
    start_time = time.time()
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File yang diunggah harus berformat gambar.")
    
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal memproses file citra: {str(e)}")
    
    # Placeholder inference logic (akan dihubungkan ke bobot model PyTorch/TensorFlow setelah training)
    # Default behavior sebelum model selesai dilatih:
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "filename": file.filename,
        "image_size": image.size,
        "predicted_class": "Menunggu Bobot Model Training",
        "confidence": 0.95,
        "class_probabilities": {
            "Rapi (Tidy)": 0.50,
            "Tidak Rapi (Untidy)": 0.50
        },
        "inference_latency_ms": latency_ms,
        "status": "ready_for_model_weights"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8016, reload=True)

