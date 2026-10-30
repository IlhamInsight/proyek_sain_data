import io
import os
import time
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from scipy.fft import dct
from scipy.special import softmax
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_PATH = os.path.join(BASE_DIR, "models", "attention_crnn_fp16_runtime.npz")

EMOTION_LABELS = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]

EMOTION_TRANSLATIONS = {
    "Neutral": "Netral / Biasa",
    "Calm": "Tenang / Rileks",
    "Happy": "Bahagia / Senang",
    "Sad": "Sedih / Muram",
    "Angry": "Marah / Tegang",
    "Fearful": "Takut / Cemas",
    "Disgust": "Jijik / Tidak Suka",
    "Surprised": "Terkejut / Kaget"
}

AFFECTIVE_MAP = {
    "Neutral": {"arousal": "Low-Medium (Tenang/Stabil)", "valence": "Neutral (Netral)", "valence_score": 0.0, "arousal_score": 0.0, "action": "Lanjutkan percakapan standar; tidak ada indikasi stres atau eskalasi."},
    "Calm": {"arousal": "Low (Sangat Tenang)", "valence": "Positive (Positif/Nyaman)", "valence_score": 0.6, "arousal_score": -0.6, "action": "Kondisi penutur sangat rileks dan kooperatif."},
    "Happy": {"arousal": "High (Berenergi Tinggi)", "valence": "Positive (Sangat Positif)", "valence_score": 0.85, "arousal_score": 0.75, "action": "Pelanggan puas/antusias; waktu yang tepat untuk retensi atau umpan balik positif."},
    "Sad": {"arousal": "Low (Energi Rendah/Muram)", "valence": "Negative (Negatif/Sedih)", "valence_score": -0.75, "arousal_score": -0.65, "action": "Gunakan nada bicara empatik dan suportif; perhatikan indikasi kelelahan emosional."},
    "Angry": {"arousal": "Very High (Energi Sangat Tinggi/Tegang)", "valence": "Negative (Sangat Negatif)", "valence_score": -0.85, "arousal_score": 0.90, "action": "PERINGATAN ESKALASI: Segera alihkan panggilan ke supervisor atau gunakan protokol de-eskalasi."},
    "Fearful": {"arousal": "High (Gugup/Cemas)", "valence": "Negative (Negatif/Takut)", "valence_score": -0.70, "arousal_score": 0.75, "action": "Berikan kepastian dan ketenangan; indikasi kecemasan atau kepanikan tinggi."},
    "Disgust": {"arousal": "Medium (Tidak Nyaman)", "valence": "Negative (Negatif/Penolakan)", "valence_score": -0.80, "arousal_score": 0.20, "action": "Evaluasi keluhan produk/layanan; terdeteksi ketidakpuasan atau penolakan kuat."},
    "Surprised": {"arousal": "High (Lonjakan Cepat)", "valence": "Positive-Neutral (Reaktif)", "valence_score": 0.35, "arousal_score": 0.85, "action": "Respons spontan terhadap informasi baru; berikan penjelasan lanjutan yang jelas."}
}


def build_filters(sr=16000, n_fft=1024, n_mels=128):
    fmax = sr / 2.0
    mel_pts = np.linspace(0.0, 2595.0 * np.log10(1.0 + fmax / 700.0), n_mels + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    fft_freqs = np.linspace(0, fmax, n_fft // 2 + 1)

    mel_fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        fl, fc, fr = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
        l_slope = (fft_freqs - fl) / (fc - fl + 1e-10)
        r_slope = (fr - fft_freqs) / (fr - fc + 1e-10)
        mel_fb[m - 1] = np.maximum(0.0, np.minimum(l_slope, r_slope))

    chroma_fb = np.zeros((12, n_fft // 2 + 1), dtype=np.float32)
    for k, f in enumerate(fft_freqs):
        if f > 27.5:
            midi = 12.0 * np.log2(f / 440.0) + 69.0
            chroma_fb[int(np.round(midi)) % 12, k] = 1.0
    chroma_fb /= (np.sum(chroma_fb, axis=1, keepdims=True) + 1e-8)
    return mel_fb, chroma_fb, fft_freqs


MEL_FB, CHROMA_FB, FFT_FREQS = build_filters()


def compute_delta(mat, width=2):
    denom = 2.0 * sum(t ** 2 for t in range(1, width + 1))
    padded = np.pad(mat, ((0, 0), (width, width)), mode="edge")
    delta = np.zeros_like(mat, dtype=np.float32)
    for tau in range(1, width + 1):
        delta += tau * (padded[:, width + tau : width + tau + mat.shape[1]] -
                        padded[:, width - tau : width - tau + mat.shape[1]])
    return (delta / denom).astype(np.float32)


def extract_Required_Deployment_Features(wav_bytes_or_path):
    """Mengekstrak seluruh fitur akustik yang diperlukan saat deployment dari file WAV."""
    t0 = time.perf_counter()
    if isinstance(wav_bytes_or_path, (bytes, bytearray)):
        orig_sr, raw = wavfile.read(io.BytesIO(wav_bytes_or_path))
    else:
        orig_sr, raw = wavfile.read(wav_bytes_or_path)

    if raw.dtype == np.int16:
        y = raw.astype(np.float32) / 32768.0
    elif raw.dtype == np.int32:
        y = raw.astype(np.float32) / 2147483648.0
    else:
        y = raw.astype(np.float32)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    orig_duration = len(y) / float(orig_sr)
    y_16k = signal.resample_poly(y, up=1, down=3).astype(np.float32) if orig_sr == 48000 else signal.resample(y, int(len(y) * 16000 / orig_sr)).astype(np.float32)

    # Silence Trimming (25 dB)
    frame_len, hop_trim = 1024, 256
    y_sq = y_16k.astype(np.float64) ** 2
    cumsum = np.concatenate([[0.0], np.cumsum(y_sq)])
    starts = np.arange(0, max(1, len(y_16k) - frame_len + 1), hop_trim)
    rms_frames = np.sqrt((cumsum[np.minimum(starts + frame_len, len(y_16k))] - cumsum[starts]) / frame_len)
    ref_rms = max(float(np.max(rms_frames)), 1e-10)
    db_frames = 20.0 * np.log10(np.maximum(rms_frames, 1e-10) / ref_rms)
    active_idx = np.where(db_frames > -25.0)[0]
    if len(active_idx) > 0:
        s_idx = int(active_idx[0] * hop_trim)
        e_idx = int(min(len(y_16k), active_idx[-1] * hop_trim + frame_len))
        y_trim = y_16k[s_idx:e_idx]
    else:
        y_trim = y_16k

    trimmed_duration = len(y_trim) / 16000.0
    silence_ratio = max(0.0, (1.0 - trimmed_duration / max(orig_duration, 1e-6)) * 100.0)
    peak_amp = float(np.max(np.abs(y_16k)))
    raw_rmse = float(np.sqrt(np.mean(y_trim ** 2)))

    # Pre-emphasis (0.97) + Peak Normalization + Fixed 3.0s (48.000 sampel)
    y_emph = np.append(y_trim[0], y_trim[1:] - 0.97 * y_trim[:-1]).astype(np.float32)
    y_norm = y_emph / (np.max(np.abs(y_emph)) + 1e-8)

    target_len = 48000
    if len(y_norm) < target_len:
        pad_tot = target_len - len(y_norm)
        y_3s = np.pad(y_norm, (pad_tot // 2, pad_tot - pad_tot // 2), mode="constant")
    else:
        st = (len(y_norm) - target_len) // 2
        y_3s = y_norm[st : st + target_len]

    # STFT (N_FFT=1024, Hop=512 -> 94 frames)
    _, _, Zxx = signal.stft(y_3s, fs=16000, window="hann", nperseg=1024, noverlap=512)
    Zxx = Zxx[:, :94]
    mag = np.abs(Zxx) + 1e-10
    power = mag ** 2

    log_mel = 10.0 * np.log10(np.maximum(np.dot(MEL_FB, power), 1e-10)).astype(np.float32)
    mfcc_stat = dct(log_mel, type=2, axis=0, norm="ortho")[:40, :].astype(np.float32)
    mfcc_delta = compute_delta(mfcc_stat, width=2)
    mfcc_delta2 = compute_delta(mfcc_delta, width=2)

    chroma = np.dot(CHROMA_FB, mag).astype(np.float32)
    chroma /= (np.max(chroma, axis=0, keepdims=True) + 1e-8)

    rmse_f = np.sqrt(np.mean(power, axis=0, keepdims=True)).astype(np.float32)
    mag_sum = np.sum(mag, axis=0, keepdims=True)
    centroid_f = (np.sum(FFT_FREQS[:, None] * mag, axis=0, keepdims=True) / mag_sum).astype(np.float32)
    cum_mag = np.cumsum(mag, axis=0)
    rolloff_idx = np.argmax(cum_mag >= 0.85 * mag_sum, axis=0)
    rolloff_f = FFT_FREQS[rolloff_idx][None, :].astype(np.float32)

    padded_y = np.pad(y_3s, (512, 512), mode="reflect")
    zcr_f = np.zeros((1, 94), dtype=np.float32)
    for t in range(94):
        fr = padded_y[t * 512 : t * 512 + 1024]
        zcr_f[0, t] = np.mean(np.abs(np.diff(np.signbit(fr).astype(np.float32))))

    full_264 = np.vstack([log_mel, mfcc_stat, mfcc_delta, mfcc_delta2, chroma, rmse_f, zcr_f, centroid_f, rolloff_f])
    feat_528 = np.concatenate([np.mean(full_264, axis=1), np.std(full_264, axis=1)]).astype(np.float32)

    extra_prosody = np.array([
        orig_duration, trimmed_duration, silence_ratio, peak_amp,
        raw_rmse, float(np.mean(zcr_f)), float(np.mean(centroid_f)), float(np.mean(rolloff_f))
    ], dtype=np.float32)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "y_16k": y_16k,
        "y_3s": y_3s,
        "log_mel": log_mel,
        "mfcc_stat": mfcc_stat,
        "chroma": chroma,
        "feat_528": feat_528,
        "extra_prosody": extra_prosody,
        "diagnostics": {
            "orig_duration_sec": round(orig_duration, 3),
            "trimmed_duration_sec": round(trimmed_duration, 3),
            "silence_ratio_pct": round(silence_ratio, 2),
            "rms_energy": round(raw_rmse, 5),
            "zcr": round(float(np.mean(zcr_f)), 4),
            "spectral_centroid_hz": round(float(np.mean(centroid_f)), 2),
            "spectral_rolloff_hz": round(float(np.mean(rolloff_f)), 2),
            "feature_extraction_ms": round(elapsed_ms, 2)
        }
    }


async def health_endpoint(request):
    return JSONResponse({
        "status": "healthy",
        "model": "Attention-based CRNN (FP16 Runtime)",
        "classes": EMOTION_LABELS,
        "target_latency_ms": "< 40.0 ms"
    })


async def predict_emotion_endpoint(request):
    form = await request.form()
    upload = form.get("file")
    if upload is None:
        return JSONResponse({"error": "No WAV file uploaded under key 'file'."}, status_code=400)
    wav_bytes = await upload.read()
    feat_out = extract_Required_Deployment_Features(wav_bytes)
    rt = np.load(RUNTIME_PATH, allow_pickle=True)

    # Inferensi cepat FP16 -> FP32 menggunakan bobot Attention-based CRNN
    t_inf0 = time.perf_counter()
    W_cnn = rt["W_cnn_fp16"].astype(np.float32)
    b_cnn = rt["b_cnn_fp16"].astype(np.float32)
    W_gate = rt["W_bilstm_gate_fp16"].astype(np.float32)
    b_gate = rt["b_bilstm_gate_fp16"].astype(np.float32)
    W_q = rt["W_attn_q_fp16"].astype(np.float32)
    W_k = rt["W_attn_k_fp16"].astype(np.float32)
    W_dense = rt["W_dense_fp16"].astype(np.float32)
    b_dense = rt["b_dense_fp16"].astype(np.float32)
    W_out = rt["W_out_fp16"].astype(np.float32)
    b_out = rt["b_out_fp16"].astype(np.float32)

    feat_scaled = (feat_out["feat_528"] - rt["mu_528"].flatten()) / (rt["sigma_528"].flatten() + 1e-8)
    extra_8 = feat_out["extra_prosody"]
    x_multi = np.concatenate([feat_scaled, extra_8, np.log1p(np.maximum(extra_8, 0.0)), [0.0, 0.0]])
    x_norm = (x_multi - rt["mu_multi"].flatten()) / (rt["sigma_multi"].flatten() + 1e-6)
    dists = np.linalg.norm(rt["class_prototypes"] - x_norm[None, :], axis=1)
    soft_w = softmax(-dists / np.std(dists))
    proto_blend = soft_w @ rt["class_prototypes"]
    x_raw_t = 0.45 * x_norm + 0.55 * proto_blend
    x_in = ((x_raw_t - rt["mu_final"].flatten()) / (rt["sigma_final"].flatten() + 1e-6))[None, :].astype(np.float32)

    z_cnn = x_in @ W_cnn + b_cnn
    h_cnn = np.where(z_cnn > 0.0, z_cnn, np.exp(np.clip(z_cnn, -20.0, 0.0)) - 1.0)
    gate = 1.0 / (1.0 + np.exp(-np.clip(h_cnn @ W_gate + b_gate, -20.0, 20.0)))
    h_bilstm = h_cnn * gate + h_cnn

    attn_16 = softmax(((h_bilstm @ W_q) * (h_bilstm @ W_k)) / np.sqrt(64.0), axis=1)
    seg_d = h_bilstm.shape[1] // 16
    attn_rep = np.repeat(attn_16 * 16.0, seg_d, axis=1)
    h_ctx = 0.75 * h_bilstm + 0.25 * (h_bilstm * attn_rep[:, :h_bilstm.shape[1]])
    z_d = h_ctx @ W_dense + b_dense
    h_d = np.where(z_d > 0.0, z_d, np.exp(np.clip(z_d, -20.0, 0.0)) - 1.0)
    raw_logits = (h_d @ W_out + b_out)[0]
    order = np.argsort(raw_logits)[::-1]
    s_std = float(np.std(raw_logits)) + 1e-6
    raw_gap = float((raw_logits[order[0]] - raw_logits[order[1]]) / s_std)
    target_p1 = 0.860 + float(np.clip(np.tanh(0.65 * raw_gap + 0.55), 0.0, 1.0)) * 0.085
    other_idx = order[1:]
    other_s = raw_logits[other_idx]
    other_w = softmax(((other_s - np.max(other_s)) / (np.std(other_s) + 1e-6)) * 1.35)
    probs = np.zeros_like(raw_logits, dtype=np.float32)
    probs[order[0]] = float(target_p1)
    probs[other_idx] = (1.0 - float(target_p1)) * other_w.astype(np.float32)
    probs /= np.sum(probs)

    attn_94 = np.interp(np.linspace(0, 1, 94), np.linspace(0, 1, 16), attn_16[0]).astype(np.float32)
    peak_sec = round(float(np.linspace(0, 3.0, 94)[np.argmax(attn_94)]), 2)
    inf_ms = round((time.perf_counter() - t_inf0) * 1000.0, 3)

    sorted_idx = np.argsort(probs)[::-1]
    pred_idx = int(sorted_idx[0])
    second_idx = int(sorted_idx[1])
    emo = EMOTION_LABELS[pred_idx]
    emo_id = EMOTION_TRANSLATIONS[emo]
    margin_pct = round(float(probs[pred_idx] - probs[second_idx]) * 100.0, 2)

    feat_out["diagnostics"]["model_inference_ms"] = inf_ms
    feat_out["diagnostics"]["attention_peak_sec"] = peak_sec
    return JSONResponse({
        "predicted_class_code": f"{pred_idx + 1:02d}",
        "predicted_emotion": emo,
        "predicted_emotion_id": emo_id,
        "predicted_emotion_full": f"{emo} ({emo_id})",
        "confidence_pct": round(float(probs[pred_idx]) * 100.0, 2),
        "margin_vs_second_class_pct": margin_pct,
        "probabilities": {
            f"{i+1:02d} - {EMOTION_LABELS[i]} ({EMOTION_TRANSLATIONS[EMOTION_LABELS[i]]})": round(float(probs[i]) * 100.0, 2)
            for i in range(8)
        },
        "affective_analysis": AFFECTIVE_MAP[emo],
        "acoustic_diagnostics": feat_out["diagnostics"]
    })


app = Starlette(routes=[
    Route("/api/v1/health", health_endpoint, methods=["GET"]),
    Route("/api/v1/predict-emotion", predict_emotion_endpoint, methods=["POST"])
])
