import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import resample
import numpy as np
import pandas as pd
from collections import Counter
import joblib
from src.models.transformer import TransformerClassifier


def check_data_validity(X, y):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)
    if y.ndim == 2 and y.shape[1] > 1:
        print("Detected one-hot encoded labels. Converting to class indices.")
        y = np.argmax(y, axis=1)
    return X, y


def balance_data(X, y):
    X0, y0 = X[y == 0], y[y == 0]
    X1, y1 = X[y == 1], y[y == 1]
    if len(y0) > len(y1):
        X0, y0 = resample(X0, y0, replace=False, n_samples=len(y1), random_state=42)
    elif len(y1) > len(y0):
        X1, y1 = resample(X1, y1, replace=False, n_samples=len(y0), random_state=42)
    X_bal = np.vstack([X0, X1])
    y_bal = np.hstack([y0, y1])
    idx = np.random.permutation(len(y_bal))
    return X_bal[idx], y_bal[idx]


def train_transformer(
    X_real, y_real,
    synthetic_data=None, synthetic_labels=None,
    device=torch.device("cpu"),
    epochs=20, batch_size=128, lr=1e-3,
    use_amp=True, val_split=0.2,
    grad_clip=1.0, early_stopping_patience=5
):
    X_real, y_real = check_data_validity(X_real, y_real)

    if synthetic_data is not None and synthetic_labels is not None:
        synthetic_data, synthetic_labels = check_data_validity(synthetic_data, synthetic_labels)
        X_combined = np.vstack([X_real, synthetic_data])
        y_combined = np.hstack([y_real, synthetic_labels])
    else:
        X_combined, y_combined = X_real, y_real

    X_bal, y_bal = balance_data(X_combined, y_combined)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_bal)
    os.makedirs("models/transformer", exist_ok=True)
    joblib.dump(scaler, "models/transformer/scaler.save")

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_bal, dtype=torch.long)  # <== torch.long for CrossEntropyLoss

    dataset = TensorDataset(X_tensor, y_tensor)
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = TransformerClassifier(input_dim=X_tensor.shape[1], num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler_amp = torch.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    best_val_acc = 0
    no_improve = 0

    print("Starting Transformer training...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_preds, train_labels = 0, [], []

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=use_amp and device.type == "cuda"):
                outputs = model(X_batch)  # [batch_size, 2]
                loss = criterion(outputs, y_batch)  # [batch_size]

            if torch.isnan(loss):
                print("NaN loss detected — skipping batch.")
                continue

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            train_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_preds.append(preds.cpu())
            train_labels.append(y_batch.cpu())

        train_loss /= train_size
        train_preds = torch.cat(train_preds)
        train_labels = torch.cat(train_labels)
        train_acc = accuracy_score(train_labels, train_preds)
        train_prec = precision_score(train_labels, train_preds, zero_division=0)
        train_rec = recall_score(train_labels, train_preds, zero_division=0)
        train_f1 = f1_score(train_labels, train_preds, zero_division=0)

        model.eval()
        val_loss, val_preds, val_labels = 0, [], []

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                out = model(X_val)
                loss = criterion(out, y_val)
                val_loss += loss.item() * X_val.size(0)
                preds = torch.argmax(out, dim=1)
                val_preds.append(preds.cpu())
                val_labels.append(y_val.cpu())

        val_loss /= val_size
        val_preds = torch.cat(val_preds)
        val_labels = torch.cat(val_labels)
        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds, zero_division=0)
        val_rec = recall_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        conf = confusion_matrix(val_labels, val_preds)

        print(f"\nEpoch {epoch} Summary:")
        print(f"Train: Loss={train_loss:.4f} Acc={train_acc:.4f} Prec={train_prec:.4f} Rec={train_rec:.4f} F1={train_f1:.4f}")
        print(f"Val:   Loss={val_loss:.4f} Acc={val_acc:.4f} Prec={val_prec:.4f} Rec={val_rec:.4f} F1={val_f1:.4f}")
        print(f"Confusion Matrix:\n{conf}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), "models/transformer/transformer_best.pth")
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print("Training complete.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv("data/processed/cicids_clean.csv")
    df.columns = df.columns.str.strip()

    features = [
        'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets',
        'Total_Length_of_Fwd_Packets', 'Total_Length_of_Bwd_Packets',
        'Fwd_Packet_Length_Max', 'Fwd_Packet_Length_Min', 'Fwd_Packet_Length_Mean'
    ]

    X_real = df[features].fillna(0).values
    y_real = (df['Label'] != 'BENIGN').astype(int).values

    synthetic_data = synthetic_labels = None
    if os.path.exists("data/synthetic/synthetic_data.csv"):
        synthetic_df = pd.read_csv("data/synthetic/synthetic_data.csv")
        synthetic_df.columns = synthetic_df.columns.str.strip()
        synthetic_data = synthetic_df[features].fillna(0).values
        synthetic_labels = np.ones(len(synthetic_data), dtype=int)
        print(f"Loaded {len(synthetic_data)} synthetic samples.")
    else:
        print("No synthetic data found. Training on real data only.")

    train_transformer(
        X_real, y_real,
        synthetic_data, synthetic_labels,
        device=device
    )
