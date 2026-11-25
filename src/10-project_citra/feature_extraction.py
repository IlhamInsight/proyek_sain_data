import numpy as np
from PIL import Image, ImageOps, ImageStat
from scipy import ndimage


# 75 fitur visual tahan-pencahayaan dan tahan-warna-lantai (floor-invariant)
# serta keragaman warna klasik (standar deviasi RGB/HSV) dan geometri objek foreground
# untuk klasifikasi kerapian alas kaki bebas overfitting.
FEATURE_COLUMNS = [
    # 1. Keragaman warna klasik (standar deviasi per kanal - invarian terhadap warna lantai)
    "std_red",
    "std_green",
    "std_blue",
    "std_overall",
    "hue_std",
    "sat_mean",
    "sat_std",
    "val_std",
    "rg_std",
    "yb_std",
    "inter_blk_sat_cv",
    "blk_hue_std_mean",
    "blk_hue_std_cv",
    # 2. Keteraturan orientasi gradien & paralelisme (Structure Tensor)
    "orient_entropy",
    "orient_gini",
    "orient_max",
    "orient_entropy_s",
    "orient_gini_s",
    "ortho_ratio",
    "diag_ratio",
    "ortho_diag_ratio",
    "global_coh",
    "blk_coh_mean",
    "blk_coh_std",
    "center_coh",
    # 3. Sebaran & kepadatan tepi spasial (Grid 4x4 & 8x8)
    "spatial_edge_entropy",
    "ed_cv",
    "ed_s_cv",
    "ed_gini",
    "fine_entropy",
    "fine_cv",
    "fine_gini",
    # 4. Spektrum Frekuensi 2D FFT (Radial & Angular alignment)
    "fft_ang_entropy",
    "fft_ang_gini",
    "fft_ang_max",
    "fft_band_0",
    "fft_band_1",
    "fft_band_2",
    "fft_band_3",
    "fft_band_4",
    "fft_band_5",
    # 5. Profil proyeksi, Radon alignment (rotasi-invarian), & kesimetrian citra
    "row_proj_cv",
    "col_proj_cv",
    "max_proj_cv",
    "proj_aniso",
    "row_proj_gini",
    "col_proj_gini",
    "radon_cv_max",
    "radon_aniso",
    "radon_peak_max",
    "lr_edge_sym",
    "tb_edge_sym",
    "center_lr_sym",
    "lr_gray_corr",
    "tb_gray_corr",
    "center_border_ratio",
    "top_bot_ratio",
    "inner_outer_ratio",
    # 6. Geometri & sebaran komponen objek foreground
    "fg_ratio",
    "fg_spread_x",
    "fg_spread_y",
    "fg_spread_diag",
    "fg_dist_mean",
    "fg_dist_p75",
    "fg_dist_iqr",
    "fg_elongation",
    "n_comp",
    # 7. Tekstur lokal LBP & GLCM (Gray-Level Co-occurrence Matrix)
    "lbp_uniform",
    "lbp_edge_ratio",
    "glcm_contrast_mean",
    "glcm_contrast_aniso",
    "glcm_hom_mean",
    "glcm_energy_mean",
    "glcm_corr_mean",
    "glcm_corr_aniso",
]


def _gini(arr: np.ndarray) -> float:
    """Menghitung koefisien ketimpangan Gini pada distribusi energi/gradien."""
    a = np.sort(np.abs(arr.ravel()))
    n = len(a)
    if n == 0 or a.sum() < 1e-12:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * a)) / (n * np.sum(a)))


def _compute_glcm_features(gray_u8: np.ndarray, levels: int = 16) -> dict[str, float]:
    """Menghitung fitur tekstur GLCM rata-rata dan anisotropi dari 4 arah (0, 45, 90, 135 derajat)."""
    q = np.clip((gray_u8.astype(np.int32) * levels) // 256, 0, levels - 1)
    distances = [(0, 1), (1, 0), (1, 1), (1, -1)]
    contrasts, homs, energies, corrs = [], [], [], []
    ii, jj = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    for dy, dx in distances:
        y1 = slice(max(0, -dy), q.shape[0] - max(0, dy))
        x1 = slice(max(0, -dx), q.shape[1] - max(0, dx))
        y2 = slice(max(0, dy), q.shape[0] - max(0, -dy))
        x2 = slice(max(0, dx), q.shape[1] - max(0, -dx))
        p1, p2 = q[y1, x1].ravel(), q[y2, x2].ravel()
        glcm = np.bincount(p1 * levels + p2, minlength=levels * levels).reshape(levels, levels).astype(np.float64)
        glcm = glcm + glcm.T
        glcm /= (glcm.sum() + 1e-12)
        contrasts.append(np.sum(glcm * ((ii - jj) ** 2)))
        homs.append(np.sum(glcm / (1.0 + np.abs(ii - jj))))
        energies.append(np.sum(glcm ** 2))
        mu_i, mu_j = np.sum(ii * glcm), np.sum(jj * glcm)
        std_i = np.sqrt(np.sum(glcm * ((ii - mu_i) ** 2)) + 1e-12)
        std_j = np.sqrt(np.sum(glcm * ((jj - mu_j) ** 2)) + 1e-12)
        corrs.append(np.sum(glcm * (ii - mu_i) * (jj - mu_j)) / (std_i * std_j + 1e-12))

    return {
        "glcm_contrast_mean": round(float(np.mean(contrasts)), 6),
        "glcm_contrast_aniso": round(float(np.max(contrasts) - np.min(contrasts)), 6),
        "glcm_hom_mean": round(float(np.mean(homs)), 6),
        "glcm_energy_mean": round(float(np.mean(energies)), 6),
        "glcm_corr_mean": round(float(np.mean(corrs)), 6),
        "glcm_corr_aniso": round(float(np.max(corrs) - np.min(corrs)), 6),
    }


def extract_image_features(image: Image.Image) -> dict[str, float | int]:
    """
    Mengekstrak 75 fitur visual klasik & invarian lantai:
    statistik warna RGB/HSV, gradien Sobel, tensor struktur, Radon alignment,
    spektrum 2D FFT, geometri foreground, simetri, dan tekstur LBP/GLCM.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")
    w_orig, h_orig = image.size
    resized = image.resize((224, 224), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32)

    # 1. Statistik warna klasik RGB & HSV
    rgb_stats = ImageStat.Stat(resized)
    r_m, g_m, b_m = rgb_stats.mean
    r_s, g_s, b_s = rgb_stats.stddev

    hsv_img = resized.convert("HSV")
    hsv_stats = ImageStat.Stat(hsv_img)
    h_m, s_m, v_m = hsv_stats.mean
    h_s, s_s, v_s = hsv_stats.stddev

    hsv_arr = np.asarray(hsv_img, dtype=np.float32)
    rg_diff = rgb[:, :, 0] - rgb[:, :, 1]
    yb_diff = 0.5 * (rgb[:, :, 0] + rgb[:, :, 1]) - rgb[:, :, 2]

    blk_hue_stds, blk_sat_means = [], []
    for r in range(4):
        for c in range(4):
            h_blk = hsv_arr[r * 56 : (r + 1) * 56, c * 56 : (c + 1) * 56, 0]
            s_blk = hsv_arr[r * 56 : (r + 1) * 56, c * 56 : (c + 1) * 56, 1]
            blk_hue_stds.append(float(h_blk.std()))
            blk_sat_means.append(float(s_blk.mean()))
    inter_blk_sat_cv = float(np.std(blk_sat_means) / (np.mean(blk_sat_means) + 1e-6))
    blk_hue_std_mean = float(np.mean(blk_hue_stds))
    blk_hue_std_cv = float(np.std(blk_hue_stds) / (np.mean(blk_hue_stds) + 1e-6))

    # 2. Normalisasi kontras skala abu-abu
    gray_raw = np.asarray(resized.convert("L"), dtype=np.float32)
    p2, p98 = np.percentile(gray_raw, [2, 98])
    gray_cs = np.clip((gray_raw - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
    gray_norm = (gray_raw - gray_raw.mean()) / (gray_raw.std() + 1e-6)

    # 3. Gradien Sobel multi-skala & Tensor Struktur
    gx = ndimage.sobel(gray_cs, axis=1)
    gy = ndimage.sobel(gray_cs, axis=0)
    mag = np.hypot(gx, gy)
    theta = np.mod(np.degrees(np.arctan2(gy, gx)), 180.0)

    gray_smooth = ndimage.gaussian_filter(gray_cs, sigma=1.5)
    gx_s = ndimage.sobel(gray_smooth, axis=1)
    gy_s = ndimage.sobel(gray_smooth, axis=0)
    mag_s = np.hypot(gx_s, gy_s)
    theta_s = np.mod(np.degrees(np.arctan2(gy_s, gx_s)), 180.0)

    bins = np.floor(theta / 10.0).astype(np.int32) % 18
    hist_18 = np.bincount(bins.ravel(), weights=mag.ravel(), minlength=18).astype(np.float32)
    hist_18 /= (hist_18.sum() + 1e-8)
    orient_entropy = float(-np.sum(hist_18 * np.log2(hist_18 + 1e-12)))
    orient_gini = _gini(hist_18)
    orient_max = float(hist_18.max())

    bins_s = np.floor(theta_s / 10.0).astype(np.int32) % 18
    hist_18_s = np.bincount(bins_s.ravel(), weights=mag_s.ravel(), minlength=18).astype(np.float32)
    hist_18_s /= (hist_18_s.sum() + 1e-8)
    orient_entropy_s = float(-np.sum(hist_18_s * np.log2(hist_18_s + 1e-12)))
    orient_gini_s = _gini(hist_18_s)

    ortho_mask = (theta_s < 15) | (theta_s > 165) | ((theta_s > 75) & (theta_s < 105))
    diag_mask = ((theta_s >= 30) & (theta_s <= 60)) | ((theta_s >= 120) & (theta_s <= 150))
    ortho_ratio = float(mag_s[ortho_mask].sum() / (mag_s.sum() + 1e-8))
    diag_ratio = float(mag_s[diag_mask].sum() / (mag_s.sum() + 1e-8))

    def _st_coh(gx_a: np.ndarray, gy_a: np.ndarray) -> float:
        jxx = float(np.mean(gx_a * gx_a))
        jyy = float(np.mean(gy_a * gy_a))
        jxy = float(np.mean(gx_a * gy_a))
        return float(np.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / (jxx + jyy + 1e-8))

    global_coh = _st_coh(gx_s, gy_s)
    center_coh = _st_coh(gx[45:179, 45:179], gy[45:179, 45:179])

    blk_cohs, blk_eds, blk_eds_s = [], [], []
    for r in range(4):
        for c in range(4):
            sl_r, sl_c = slice(r * 56, (r + 1) * 56), slice(c * 56, (c + 1) * 56)
            blk_cohs.append(_st_coh(gx_s[sl_r, sl_c], gy_s[sl_r, sl_c]))
            blk_eds.append(float(mag[sl_r, sl_c].mean()))
            blk_eds_s.append(float(mag_s[sl_r, sl_c].mean()))
    blk_cohs = np.array(blk_cohs)
    blk_eds = np.array(blk_eds)
    blk_eds_s = np.array(blk_eds_s)

    ed_prob = blk_eds / (blk_eds.sum() + 1e-8)
    spatial_edge_entropy = float(-np.sum(ed_prob * np.log2(ed_prob + 1e-12)))
    ed_cv = float(blk_eds.std() / (blk_eds.mean() + 1e-8))
    ed_s_cv = float(blk_eds_s.std() / (blk_eds_s.mean() + 1e-8))
    ed_gini = _gini(blk_eds)

    fine_eds = np.array([
        float(mag[r * 28 : (r + 1) * 28, c * 28 : (c + 1) * 28].mean())
        for r in range(8)
        for c in range(8)
    ])
    fine_prob = fine_eds / (fine_eds.sum() + 1e-8)
    fine_entropy = float(-np.sum(fine_prob * np.log2(fine_prob + 1e-12)))
    fine_cv = float(fine_eds.std() / (fine_eds.mean() + 1e-8))
    fine_gini = _gini(fine_eds)

    # 4. Spektrum Frekuensi 2D FFT
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(gray_norm)))
    fft_pow = fft_mag ** 2
    yy, xx = np.ogrid[:224, :224]
    dy_f, dx_f = yy - 112, xx - 112
    dist = np.sqrt(dy_f ** 2 + dx_f ** 2)
    ang_f = np.mod(np.degrees(np.arctan2(dy_f, dx_f)), 180.0)

    ac_mask = dist > 2
    tot_ac = float(fft_pow[ac_mask].sum() + 1e-8)
    bands = [(2, 8), (8, 16), (16, 32), (32, 56), (56, 84), (84, 112)]
    band_ratios = [float(fft_pow[(dist > r1) & (dist <= r2)].sum() / tot_ac) for r1, r2 in bands]

    ang_mask = (dist > 5) & (dist <= 70)
    ang_bins = np.floor(ang_f[ang_mask] / 10.0).astype(np.int32) % 18
    fft_ang_hist = np.bincount(ang_bins, weights=fft_pow[ang_mask], minlength=18).astype(np.float32)
    fft_ang_hist /= (fft_ang_hist.sum() + 1e-8)
    fft_ang_entropy = float(-np.sum(fft_ang_hist * np.log2(fft_ang_hist + 1e-12)))
    fft_ang_gini = _gini(fft_ang_hist)
    fft_ang_max = float(fft_ang_hist.max())

    # 5. Profil Proyeksi, Radon Rotated Alignment, & Simetri
    row_proj = np.mean(mag, axis=1)
    col_proj = np.mean(mag, axis=0)
    row_proj_cv = float(row_proj.std() / (row_proj.mean() + 1e-8))
    col_proj_cv = float(col_proj.std() / (col_proj.mean() + 1e-8))

    mag_small = ndimage.zoom(mag, 64.0 / 224.0, order=1)
    proj_cvs, proj_peaks = [], []
    for ang in range(0, 180, 15):
        rot = ndimage.rotate(mag_small, ang, reshape=False, order=1)
        c_roi = rot[12:52, 12:52]
        p = np.mean(c_roi, axis=1)
        proj_cvs.append(float(p.std() / (p.mean() + 1e-8)))
        proj_peaks.append(float(p.max() / (p.mean() + 1e-8)))

    mag_norm = mag / (mag.mean() + 1e-8)
    lr_edge_sym = float(np.mean(np.abs(mag_norm - np.fliplr(mag_norm))))
    tb_edge_sym = float(np.mean(np.abs(mag_norm - np.flipud(mag_norm))))
    c_mag = mag[45:179, 45:179]
    c_norm = c_mag / (c_mag.mean() + 1e-8)
    center_lr_sym = float(np.mean(np.abs(c_norm - np.fliplr(c_norm))))

    lr_gray_corr = float(np.corrcoef(gray_norm.ravel(), np.fliplr(gray_norm).ravel())[0, 1])
    tb_gray_corr = float(np.corrcoef(gray_norm.ravel(), np.flipud(gray_norm).ravel())[0, 1])

    center_mag = mag[56:168, 56:168].mean()
    border_mag = (mag.sum() - mag[56:168, 56:168].sum()) / (224 * 224 - 112 * 112)
    center_border_ratio = float(center_mag / (border_mag + 1e-8))
    top_bot_ratio = float(mag[:112, :].mean() / (mag[112:, :].mean() + 1e-8))

    r_norm = dist / 112.0
    inner_edge = float(mag[r_norm <= 0.55].mean())
    outer_edge = float(mag[(r_norm > 0.55) & (r_norm <= 1.0)].mean())
    inner_outer_ratio = float(inner_edge / (outer_edge + 1e-8))

    # 6. Geometri & Sebaran Objek Foreground
    thresh = np.mean(mag) + np.std(mag)
    fg_mask = mag > thresh
    fg_ratio = float(np.mean(fg_mask))
    if np.sum(fg_mask) > 20:
        ys, xs = np.where(fg_mask)
        fg_spread_x = float(np.std(xs) / 224.0)
        fg_spread_y = float(np.std(ys) / 224.0)
        fg_spread_diag = float(np.hypot(fg_spread_x, fg_spread_y))
        cy, cx = ys.mean(), xs.mean()
        dists = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2) / 224.0
        fg_dist_mean = float(dists.mean())
        fg_dist_p75 = float(np.percentile(dists, 75))
        fg_dist_iqr = float(np.percentile(dists, 75) - np.percentile(dists, 25))
        cov = np.cov(np.vstack([xs, ys]))
        eigvals = np.linalg.eigvalsh(cov)
        fg_elongation = float((eigvals[1] - eigvals[0]) / (eigvals[1] + eigvals[0] + 1e-8))
        _, n_comp = ndimage.label(ndimage.binary_closing(fg_mask, structure=np.ones((3, 3))))
    else:
        fg_spread_x, fg_spread_y, fg_spread_diag = 0.0, 0.0, 0.0
        fg_dist_mean, fg_dist_p75, fg_dist_iqr, fg_elongation, n_comp = 0.0, 0.0, 0.0, 0.0, 0

    # 7. Tekstur LBP & GLCM
    glcm_feats = _compute_glcm_features((gray_cs * 255).astype(np.uint8), levels=16)
    g_q = (gray_cs[1:-1, 1:-1] * 255).astype(np.float32)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    lbp = np.zeros_like(g_q, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(offsets):
        nb = (gray_cs[1 + dy : 223 + dy, 1 + dx : 223 + dx] * 255) >= g_q
        lbp |= (nb.astype(np.uint8) << bit)
    lbp_rot = np.right_shift(lbp, 1) | ((lbp & 1) << 7)
    transitions = np.unpackbits((lbp ^ lbp_rot)[:, :, None], axis=2).sum(axis=2)
    bit_ones = np.unpackbits(lbp[:, :, None], axis=2).sum(axis=2).astype(int)
    lbp_uniform = float(np.mean(transitions <= 2))
    lbp_edge_ratio = float(np.mean((transitions <= 2) & (bit_ones >= 3) & (bit_ones <= 5)))

    features: dict[str, float | int] = {
        "width": w_orig,
        "height": h_orig,
        "aspect_ratio": round(w_orig / (h_orig + 1e-6), 4),
        "mean_red": round(float(r_m), 4),
        "std_red": round(float(r_s), 4),
        "mean_green": round(float(g_m), 4),
        "std_green": round(float(g_s), 4),
        "mean_blue": round(float(b_m), 4),
        "std_blue": round(float(b_s), 4),
        "mean_overall": round(float((r_m + g_m + b_m) / 3.0), 4),
        "std_overall": round(float((r_s + g_s + b_s) / 3.0), 4),
        "mean_hue": round(float(h_m), 4),
        "hue_std": round(float(h_s), 4),
        "sat_mean": round(float(s_m), 4),
        "sat_std": round(float(s_s), 4),
        "mean_val": round(float(v_m), 4),
        "val_std": round(float(v_s), 4),
        "rg_std": round(float(rg_diff.std()), 4),
        "yb_std": round(float(yb_diff.std()), 4),
        "inter_blk_sat_cv": round(inter_blk_sat_cv, 6),
        "blk_hue_std_mean": round(blk_hue_std_mean, 6),
        "blk_hue_std_cv": round(blk_hue_std_cv, 6),
        "orient_entropy": round(orient_entropy, 6),
        "orient_gini": round(orient_gini, 6),
        "orient_max": round(orient_max, 6),
        "orient_entropy_s": round(orient_entropy_s, 6),
        "orient_gini_s": round(orient_gini_s, 6),
        "ortho_ratio": round(ortho_ratio, 6),
        "diag_ratio": round(diag_ratio, 6),
        "ortho_diag_ratio": round(float(ortho_ratio / (diag_ratio + 1e-6)), 6),
        "global_coh": round(global_coh, 6),
        "blk_coh_mean": round(float(blk_cohs.mean()), 6),
        "blk_coh_std": round(float(blk_cohs.std()), 6),
        "center_coh": round(center_coh, 6),
        "spatial_edge_entropy": round(spatial_edge_entropy, 6),
        "ed_cv": round(ed_cv, 6),
        "ed_s_cv": round(ed_s_cv, 6),
        "ed_gini": round(ed_gini, 6),
        "fine_entropy": round(fine_entropy, 6),
        "fine_cv": round(fine_cv, 6),
        "fine_gini": round(fine_gini, 6),
        "fft_ang_entropy": round(fft_ang_entropy, 6),
        "fft_ang_gini": round(fft_ang_gini, 6),
        "fft_ang_max": round(fft_ang_max, 6),
        "row_proj_cv": round(row_proj_cv, 6),
        "col_proj_cv": round(col_proj_cv, 6),
        "max_proj_cv": round(max(row_proj_cv, col_proj_cv), 6),
        "proj_aniso": round(abs(row_proj_cv - col_proj_cv), 6),
        "row_proj_gini": round(_gini(row_proj), 6),
        "col_proj_gini": round(_gini(col_proj), 6),
        "radon_cv_max": round(max(proj_cvs), 6),
        "radon_aniso": round(max(proj_cvs) - min(proj_cvs), 6),
        "radon_peak_max": round(max(proj_peaks), 6),
        "lr_edge_sym": round(lr_edge_sym, 6),
        "tb_edge_sym": round(tb_edge_sym, 6),
        "center_lr_sym": round(center_lr_sym, 6),
        "lr_gray_corr": round(0.0 if np.isnan(lr_gray_corr) else lr_gray_corr, 6),
        "tb_gray_corr": round(0.0 if np.isnan(tb_gray_corr) else tb_gray_corr, 6),
        "center_border_ratio": round(center_border_ratio, 6),
        "top_bot_ratio": round(top_bot_ratio, 6),
        "inner_outer_ratio": round(inner_outer_ratio, 6),
        "fg_ratio": round(fg_ratio, 6),
        "fg_spread_x": round(fg_spread_x, 6),
        "fg_spread_y": round(fg_spread_y, 6),
        "fg_spread_diag": round(fg_spread_diag, 6),
        "fg_dist_mean": round(fg_dist_mean, 6),
        "fg_dist_p75": round(fg_dist_p75, 6),
        "fg_dist_iqr": round(fg_dist_iqr, 6),
        "fg_elongation": round(fg_elongation, 6),
        "n_comp": int(n_comp),
        "lbp_uniform": round(lbp_uniform, 6),
        "lbp_edge_ratio": round(lbp_edge_ratio, 6),
        **glcm_feats,
    }
    for idx, br in enumerate(band_ratios):
        features[f"fft_band_{idx}"] = round(br, 6)
    return features
