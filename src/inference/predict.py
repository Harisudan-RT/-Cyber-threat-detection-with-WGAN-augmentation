import torch
import pandas as pd
import joblib
from src.models.transformer import TransformerClassifier
from src.utils.helpers import compute_metrics, print_metrics

# Use exact CSV feature names here
FEATURES = [
    'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets',
    'Flow_Bytes/s', 'Fwd_Packet_Length_Max', 'Bwd_Packet_Length_Max',
    'Flow_IAT_Mean', 'Flow_IAT_Std'
]

# Map class indices to labels
CLASS_NAMES = {0: 'Benign', 1: 'Malicious'}

def predict_from_csv(csv_path, model_path, scaler_path):
    """Predict labels from a CSV using trained Transformer model."""

    # Load pre-trained model
    model = TransformerClassifier(input_dim=len(FEATURES), model_dim=64, num_classes=1)  # Use num_classes=1 for binary output
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Load CSV data
    df = pd.read_csv(csv_path)

    # Check if all features are present
    if not all(f in df.columns for f in FEATURES):
        missing = [f for f in FEATURES if f not in df.columns]
        raise ValueError(f"Missing required features in CSV: {missing}")

    # Extract features and scale
    X = df[FEATURES].fillna(0).values
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        outputs = model(X_tensor)
        if outputs.shape[1] == 1:
            pred_indices = (torch.sigmoid(outputs).squeeze() > 0.5).int().numpy()
        else:
            pred_indices = torch.argmax(outputs, dim=1).numpy()
        pred_labels = [CLASS_NAMES[i] for i in pred_indices]

    # Add predictions to DataFrame
    df['Prediction'] = pred_labels

    # If true labels exist, compute metrics
    label_col = next((col for col in df.columns if col.lower() == 'label'), None)
    if label_col:
        y_true = df[label_col].map(lambda x: 'Benign' if str(x).strip().upper() == 'BENIGN' else 'Malicious')
        metrics = compute_metrics(y_true, pred_labels)
        print_metrics(metrics)

    return df

if __name__ == "__main__":
    csv_path = "data/processed/cicids_test.csv"        # Path to input CSV
    model_path = "models/transformer/transformer_best.pth"  # Path to saved model
    scaler_path = "models/transformer/scaler.save"          # Path to saved scaler

    results_df = predict_from_csv(csv_path, model_path, scaler_path)
    print(results_df.head())
