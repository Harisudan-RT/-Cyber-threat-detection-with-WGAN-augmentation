import torch
import pandas as pd
from src.models.wgan import Generator

# Consistent features (same used during training)
FEATURES = [
    'flow_duration', 'total_fwd_packets', 'total_backward_packets',
    'flow_bytes_s', 'fwd_packet_length_max', 'bwd_packet_length_max',
    'flow_iat_mean', 'flow_iat_std'
]

def generate_synthetic_samples(generator_path, num_samples=1000, noise_dim=100, output_file="data/synthetic/synthetic_data.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load generator model
    generator = Generator(input_dim=noise_dim, output_dim=len(FEATURES)).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()

    # Generate synthetic noise
    z = torch.randn(num_samples, noise_dim).to(device)
    with torch.no_grad():
        synthetic_data = generator(z).cpu().numpy()

    # Save to CSV
    df = pd.DataFrame(synthetic_data, columns=FEATURES)
    df['label'] = 0  # Mark as benign
    df.to_csv(output_file, index=False)

    print(f"✅ Generated {num_samples} synthetic samples at {output_file}")
