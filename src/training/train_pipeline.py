import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

from src.training.train_wgan import train_wgan
from src.training.train_transformer import train_transformer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # === Step 1: Load Dataset ===
    print("[INFO] Loading and preparing dataset...")
    df = pd.read_csv("data/processed/cicids_clean.csv")
    df.columns = df.columns.str.strip()

    selected_features = [
        'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets',
        'Total_Length_of_Fwd_Packets', 'Total_Length_of_Bwd_Packets',
        'Fwd_Packet_Length_Max', 'Fwd_Packet_Length_Min', 'Fwd_Packet_Length_Mean'
    ]

    # Check column presence
    missing = [col for col in selected_features + ['Label'] if col not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] Missing required columns: {missing}")

    benign_df = df[df['Label'] == 'BENIGN'][selected_features].fillna(0)
    print(f"[INFO] Benign sample shape for WGAN: {benign_df.shape}")

    # === Step 2: Scale Benign Data ===
    print("[INFO] Scaling benign data...")
    scaler = MinMaxScaler()
    benign_scaled = scaler.fit_transform(benign_df.values.astype(np.float32))

    os.makedirs("models/wgan", exist_ok=True)
    joblib.dump(scaler, "models/wgan/scaler.pkl")

    # === Step 3: Train WGAN ===
    print("[INFO] Training WGAN to generate synthetic benign traffic...")
    real_tensor = torch.tensor(benign_scaled, dtype=torch.float32)
    synthetic_tensor = train_wgan(real_tensor, device=device)
    synthetic_np = synthetic_tensor.cpu().numpy()

    # === Step 4: Reverse Scale & Save Synthetic Data ===
    print("[INFO] Reversing scaling and saving synthetic data...")
    synthetic_original = scaler.inverse_transform(synthetic_np)
    synthetic_df = pd.DataFrame(synthetic_original, columns=selected_features)

    os.makedirs("data/synthetic", exist_ok=True)
    synthetic_df.to_csv("data/synthetic/synthetic_data.csv", index=False)
    print("[INFO] Saved synthetic samples to data/synthetic/synthetic_data.csv")

    # === Step 5: Prepare Real Data + Labels ===
    print("[INFO] Preparing data for Transformer training...")
    label_map = {'BENIGN': 0}
    df['label_num'] = df['Label'].map(label_map).fillna(1).astype(int)

    real_features = df[selected_features].fillna(0).values.astype(np.float32)
    real_labels = df['label_num'].values

    synthetic_features = synthetic_df[selected_features].values.astype(np.float32)
    synthetic_labels = np.zeros(len(synthetic_features), dtype=int)

    X_train = np.vstack([real_features, synthetic_features])
    y_train = np.hstack([real_labels, synthetic_labels])

    print(f"[INFO] Final training set: {X_train.shape}, Labels: {y_train.shape}")

    # === Step 6: Train Transformer ===
    print("[INFO] Training Transformer classifier...")
    train_transformer(X_train, torch.tensor(y_train, dtype=torch.long), device=device)

    print("[DONE] Pipeline complete.")

if __name__ == "__main__":
    main()
