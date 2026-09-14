import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROJECT_ROOT = os.path.abspath(os.getcwd())
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
file_path = os.path.join(DATA_DIR, 'data_kualitas_udara_mendenrejo.csv')

df = pd.read_csv(file_path)
co = df['CO'].interpolate(method='linear').bfill().ffill()
df['CO_imputed'] = co
df['DATE'] = pd.to_datetime(df['DATE_TIME'])
df = df.sort_values(by='DATE').reset_index(drop=True)

mean_val = float(co.mean())
std_val = float(co.std())
upper_bound = mean_val + 1.5 * std_val
lower_bound = mean_val - 1.5 * std_val

df['is_upper_outlier'] = df['CO_imputed'] > upper_bound
df['is_lower_outlier'] = df['CO_imputed'] < lower_bound

upper_pts = df[df['is_upper_outlier']].sort_values(by='DATE').reset_index(drop=True)
lower_pts = df[df['is_lower_outlier']].sort_values(by='DATE').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(20, 10), dpi=140)

# Main line
ax.plot(df['DATE'], df['CO_imputed'], color='#3B82F6', linewidth=1.2, alpha=0.7, label='Konsentrasi CO Harian (ug/m3)')

# Threshold lines
ax.axhline(upper_bound, color='#DC2626', linestyle='--', linewidth=1.6, label=f'Batas Ambang Atas ({upper_bound:.2f} ug/m3)')
ax.axhline(lower_bound, color='#7C3AED', linestyle='--', linewidth=1.6, label=f'Batas Ambang Bawah ({lower_bound:.2f} ug/m3)')
ax.axhline(mean_val, color='#059669', linestyle=':', linewidth=1.4, label=f'Rata-rata / Mean ({mean_val:.2f} ug/m3)')

# Scatter points
ax.scatter(upper_pts['DATE'], upper_pts['CO_imputed'], color='#EF4444', s=65, zorder=5, edgecolors='#7F1D1D', linewidths=1.2,
           label=f'Titik Outlier Atas (Semua {len(upper_pts)} kejadian beranotasi nilai)')
ax.scatter(lower_pts['DATE'], lower_pts['CO_imputed'], color='#8B5CF6', s=65, zorder=5, edgecolors='#4C1D95', linewidths=1.2,
           label=f'Titik Outlier Bawah (Semua {len(lower_pts)} kejadian beranotasi nilai)')

# Annotate ALL 22 Upper Outliers with alternating offsets
offsets_upper = [16, 38, 60]
for i, row in upper_pts.iterrows():
    offset_y = offsets_upper[i % 3]
    tgl = row['DATE'].strftime('%d/%m')
    val = f"{row['CO_imputed']:.1f}"
    ax.annotate(f"{tgl}\n{val}",
                xy=(row['DATE'], row['CO_imputed']),
                xytext=(0, offset_y), textcoords='offset points',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#991B1B',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#FEE2E2', edgecolor='#EF4444', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='#DC2626', lw=0.9))

# Annotate ALL 27 Lower Outliers with alternating offsets
offsets_lower = [-20, -42, -64]
for i, row in lower_pts.iterrows():
    offset_y = offsets_lower[i % 3]
    tgl = row['DATE'].strftime('%d/%m')
    val = f"{row['CO_imputed']:.1f}"
    ax.annotate(f"{tgl}\n{val}",
                xy=(row['DATE'], row['CO_imputed']),
                xytext=(0, offset_y), textcoords='offset points',
                ha='center', va='top', fontsize=7.5, fontweight='bold', color='#4C1D95',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#EDE9FE', edgecolor='#8B5CF6', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='#7C3AED', lw=0.9))

ax.set_title('Grafik Identifikasi Outlier Deret Waktu CO Harian (Lengkap dengan Tanggal & Nilai di Seluruh Titik Outlier)', fontsize=14, fontweight='bold', pad=18)
ax.set_xlabel('Tanggal Pengamatan (1 September 2025 s/d 1 September 2026)', fontsize=11, fontweight='semibold')
ax.set_ylabel('Konsentrasi CO (ug/m3)', fontsize=11, fontweight='semibold')
ax.set_ylim(-180, 2300)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=25)
ax.grid(True, linestyle='--', alpha=0.45)
ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=9.5)

plt.tight_layout()
plt.savefig('scratch_test_all_annotated.png')
plt.close()
print("Saved scratch_test_all_annotated.png successfully!")
