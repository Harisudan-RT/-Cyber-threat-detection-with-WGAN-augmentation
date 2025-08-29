import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_linear(x).unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        return self.classifier(x)

def evaluate_model(model, X, y, device):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        outputs = model(X_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    conf_mat = confusion_matrix(y, preds)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall: {rec:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_mat}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    selected_features = [
        'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets',
        'Total_Length_of_Fwd_Packets', 'Total_Length_of_Bwd_Packets',
        'Fwd_Packet_Length_Max', 'Fwd_Packet_Length_Min', 'Fwd_Packet_Length_Mean'
    ]

    # Load your test dataset CSV here
    df_test = pd.read_csv("data/processed/cicids_test.csv")
    df_test.columns = df_test.columns.str.strip()

    X_test = df_test[selected_features].fillna(0).values
    y_test = (df_test['Label'] != 'BENIGN').astype(int).values

    scaler = MinMaxScaler()
    X_test_scaled = scaler.fit_transform(X_test)  # Optional: save & load scaler from training instead for consistency

    model = TransformerClassifier(input_dim=len(selected_features)).to(device)
    model.load_state_dict(torch.load("models/transformer/transformer_best.pth", map_location=device))

    evaluate_model(model, X_test_scaled, y_test, device)
