import torch
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# -------------------- Scaler Utilities --------------------

def save_scaler(scaler: StandardScaler, path: str):
    """Save fitted scaler to disk."""
    joblib.dump(scaler, path)

def load_scaler(path: str) -> StandardScaler:
    """Load saved scaler from disk."""
    return joblib.load(path)

# -------------------- Model Utilities --------------------

def load_model(model_path: str):
    """Load a PyTorch model with CUDA fallback."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.load(model_path, map_location=device)

# -------------------- Evaluation Utilities --------------------

def compute_metrics(y_true, y_pred):
    """Compute classification evaluation metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

def print_metrics(metrics):
    """Print evaluation metrics in readable format."""
    print("Model Performance:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k.capitalize()}: {v:.4f}")
    print("Confusion Matrix:\n", metrics["confusion_matrix"])
