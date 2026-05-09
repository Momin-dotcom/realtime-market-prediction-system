import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import os

# ── Load the real timeseries CSV ──────────────────────────────────────────────
df = pd.read_csv('data/processed/timeseries_dataset.csv')
print(f"Loaded timeseries_dataset.csv: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ── Select only the scaled feature columns + target ───────────────────────────
# These are already normalized by Member 1 — use them directly
feature_cols = [
    'open_scaled', 'high_scaled', 'low_scaled', 'close_scaled',
    'volume_scaled', 'price_change_pct_scaled',
    'MA_7_scaled', 'MA_21_scaled', 'RSI_scaled',
    'MACD_scaled', 'MACD_signal_scaled',
    'BB_mid_scaled', 'BB_upper_scaled', 'BB_lower_scaled',
    'sentiment_ratio_scaled'
]

# Target column — direction: UP=1, DOWN=0
target_col = 'direction'

# ── Verify columns exist ───────────────────────────────────────────────────────
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    print(f"WARNING: Missing columns: {missing}")
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"Using {len(feature_cols)} available feature columns")

print(f"\nUsing {len(feature_cols)} features: {feature_cols}")

# ── Convert direction to binary labels ────────────────────────────────────────
df[target_col] = df[target_col].str.strip().str.upper()
print(f"\nDirection value counts:\n{df[target_col].value_counts()}")

df['label'] = (df[target_col] == 'UP').astype(int)
print(f"Label distribution: {df['label'].value_counts().to_dict()}")

# ── Drop rows with any NaN in feature columns ─────────────────────────────────
df = df.dropna(subset=feature_cols + ['label'])
print(f"\nAfter dropping NaN rows: {df.shape}")

# ── Build sliding window sequences ────────────────────────────────────────────
WINDOW_SIZE = 10   # use last 10 timesteps to predict next direction

features = df[feature_cols].values
labels   = df['label'].values

X, y = [], []
for i in range(len(features) - WINDOW_SIZE):
    X.append(features[i : i + WINDOW_SIZE])
    y.append(labels[i + WINDOW_SIZE])

X = np.array(X, dtype=np.float64)
y = np.array(y, dtype=np.int64)

print(f"\nSequences built:")
print(f"  X shape: {X.shape}  →  (samples, timesteps, features)")
print(f"  y shape: {y.shape}")
print(f"  Label balance: {np.sum(y==0)} DOWN, {np.sum(y==1)} UP")

# ── Train / Val / Test split  (70% / 15% / 15%) ───────────────────────────────
# shuffle=False because this is time-series — order matters
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, shuffle=False
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, shuffle=False   # 0.176 of 85% ≈ 15% overall
)

print(f"\nSplit sizes:")
print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"  X_val:   {X_val.shape}    y_val:   {y_val.shape}")
print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")

# ── Save the new .npy files ────────────────────────────────────────────────────
out_dir = 'data/processed'
os.makedirs(out_dir, exist_ok=True)

np.save(f'{out_dir}/X_train.npy', X_train)
np.save(f'{out_dir}/X_val.npy',   X_val)
np.save(f'{out_dir}/X_test.npy',  X_test)
np.save(f'{out_dir}/y_train.npy', y_train)
np.save(f'{out_dir}/y_val.npy',   y_val)
np.save(f'{out_dir}/y_test.npy',  y_test)

print(f"\nAll 6 .npy files saved to {out_dir}/")

# ── Save feature column list for reference ────────────────────────────────────
with open(f'{out_dir}/feature_cols.txt', 'w') as f:
    f.write('\n'.join(feature_cols))
print(f"Feature column list saved to {out_dir}/feature_cols.txt")
print(f"\nNumber of features per timestep: {len(feature_cols)}")
print("Done.")