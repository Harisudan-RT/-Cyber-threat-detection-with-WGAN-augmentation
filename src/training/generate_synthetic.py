import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.models.wgan import Generator  # Adjust import if needed

def load_scaler(scaler_path):
    import joblib
    return joblib.load(scaler_path)

def generate_synthetic_samples(
    generator,
    latent_dim,
    num_samples,
    device,
    scaler=None,
):
    generator.eval()
    noise = torch.randn(num_samples, latent_dim, device=device)

    with torch.no_grad():
        fake_data = generator(noise).cpu().numpy()

    # If output is sigmoid, data in [0,1], inverse transform to original scale
    if scaler is not None:
        fake_data = scaler.inverse_transform(fake_data)

    return fake_data

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating synthetic data on device: {device}")

    # Parameters
    latent_dim = 32
    num_synthetic_samples = 10000  # Adjust how many you want to generate

    # Paths (adjust if needed)
    model_path = "models/wgan/generator.pth"
    scaler_path = "models/wgan/scaler.pkl"
    synthetic_data_path = "data/synthetic/synthetic_data.csv"

    # Define feature columns exactly as in your original data
    feature_columns = [
        'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets',
        'Total_Length_of_Fwd_Packets', 'Total_Length_of_Bwd_Packets',
        'Fwd_Packet_Length_Max', 'Fwd_Packet_Length_Min', 'Fwd_Packet_Length_Mean'
    ]

    # Load the trained generator
    generator = Generator(latent_dim=latent_dim, output_dim=len(feature_columns)).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    # Load scaler for inverse transforming synthetic data
    scaler = load_scaler(scaler_path)

    # Generate synthetic samples
    synthetic_samples = generate_synthetic_samples(
        generator,
        latent_dim=latent_dim,
        num_samples=num_synthetic_samples,
        device=device,
        scaler=scaler
    )

    # Create DataFrame and save
    df_synthetic = pd.DataFrame(synthetic_samples, columns=feature_columns)
    os.makedirs(os.path.dirname(synthetic_data_path), exist_ok=True)
    df_synthetic.to_csv(synthetic_data_path, index=False)

    print(f"Saved {num_synthetic_samples} synthetic samples to {synthetic_data_path}")
