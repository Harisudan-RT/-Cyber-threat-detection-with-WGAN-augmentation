import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import gc

# Ensure CUDA context is initialized early
if torch.cuda.is_available():
    _ = torch.cuda.current_device()

# Generator network with Tanh output [-1, 1]
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, x):
        return self.model(x)

# Critic (Discriminator)
class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

# Gradient Penalty for WGAN-GP
def gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=device).expand_as(real)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)
    mixed_scores = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grad_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()

# WGAN training function
def train_wgan(real_data, device='cpu', epochs=100, batch_size=256, noise_dim=100, lr=1e-4, subsample_size=300000, n_critic=3):
    if not isinstance(real_data, torch.Tensor):
        real_data = torch.tensor(real_data, dtype=torch.float32)
    real_data = real_data.to(device)

    print(f"Input data min: {real_data.min().item():.6f}, max: {real_data.max().item():.6f}")
    feature_dim = real_data.shape[1]

    generator = Generator(noise_dim, feature_dim).to(device)
    critic = Critic(feature_dim).to(device)

    opt_gen = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_critic = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))
    lambda_gp = 10

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    dataset = TensorDataset(real_data)

    print(f"Starting WGAN training on device: {device}")
    for epoch in range(1, epochs + 1):
        indices = np.random.choice(len(real_data), subsample_size, replace=False)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(indices),
            drop_last=True,
            num_workers=0
        )

        for real_batch, in dataloader:
            real_batch = real_batch.to(device)

            # Train Critic
            for _ in range(n_critic):
                noise = torch.randn(batch_size, noise_dim, device=device)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    fake_batch = generator(noise).detach()
                    critic_real = critic(real_batch)
                    critic_fake = critic(fake_batch)
                    gp = gradient_penalty(critic, real_batch, fake_batch, device)
                    loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp

                opt_critic.zero_grad()
                scaler.scale(loss_critic).backward()
                scaler.step(opt_critic)
                scaler.update()

            # Train Generator
            noise = torch.randn(batch_size, noise_dim, device=device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                fake_batch = generator(noise)
                loss_gen = -torch.mean(critic(fake_batch))

            opt_gen.zero_grad()
            scaler.scale(loss_gen).backward()
            scaler.step(opt_gen)
            scaler.update()

        torch.cuda.empty_cache()
        gc.collect()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Critic Loss: {loss_critic.item():.6f} | Generator Loss: {loss_gen.item():.6f}")
            if device.type == 'cuda':
                print(f"  CUDA memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
                print(f"  CUDA memory reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

    print("✅ WGAN training complete.")

    with torch.no_grad():
        final_noise = torch.randn(10000, noise_dim, device=device)
        synthetic_samples = generator(final_noise)

    # Convert to numpy if it's still a tensor
    if isinstance(synthetic_samples, torch.Tensor):
        synthetic_samples = synthetic_samples.cpu().numpy()

    os.makedirs("models/wgan", exist_ok=True)
    torch.save(generator.state_dict(), "models/wgan/generator.pth")

    return synthetic_samples


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv("data/processed/cicids_clean.csv")
    df.columns = df.columns.str.strip()

    selected_features = [
        'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets',
        'Flow_Bytes/s', 'Fwd_Packet_Length_Max', 'Bwd_Packet_Length_Max',
        'Flow_IAT_Mean', 'Flow_IAT_Std'
    ]

    benign_df = df[df['Label'] == 'BENIGN'][selected_features].fillna(0)

    # Scale features into [-1, 1] to match Tanh output
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_benign = scaler.fit_transform(benign_df)

    synthetic_data = train_wgan(
        real_data=scaled_benign,
        device=device,
        epochs=100,
        batch_size=256,
        noise_dim=100,
        lr=1e-4,
        subsample_size=300000,
        n_critic=3
    )

    # Inverse scale synthetic data to original feature ranges
    synthetic_original = scaler.inverse_transform(synthetic_data)

    # Save synthetic data CSV for inspection or further use
    synthetic_df = pd.DataFrame(synthetic_original, columns=selected_features)
    synthetic_df.to_csv("synthetic_wgan_data.csv", index=False)
    print("Synthetic data shape:", synthetic_df.shape)
    print("Saved synthetic data in original scale to synthetic_wgan_data.csv")
