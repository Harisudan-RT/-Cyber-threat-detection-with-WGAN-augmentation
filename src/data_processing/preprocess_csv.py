import os
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def scale_to_minus1_1(df, exclude_cols):
    """
    Scale DataFrame numeric columns to [-1, 1] using MinMaxScaler,
    excluding specified columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scale_cols = [col for col in numeric_cols if col not in exclude_cols]

    # Replace infinite values with NaN to avoid errors
    df[scale_cols] = df[scale_cols].replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaN in scaling columns
    df.dropna(subset=scale_cols, inplace=True)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    return df, scaler


def preprocess_iscx_for_wgan(data_folder, save_path="data/processed/cicids_clean.csv"):
    all_data = []

    for filename in tqdm(os.listdir(data_folder)):
        if filename.endswith(".csv"):
            path = os.path.join(data_folder, filename)
            print(f"Processing: {path}")

            df = pd.read_csv(path, low_memory=False, encoding='ISO-8859-1')

            # Skip small files
            if df.shape[0] < 100:
                print(f"Skipping small file: {filename}")
                continue

            # Clean column names
            df.columns = df.columns.str.strip().str.replace(' ', '_')

            required_cols = ['Label', 'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Required columns missing in {filename}, skipping.")
                continue

            # Columns to exclude from numeric conversion
            exclude_cols = ['Label', 'Flow_ID', 'Source_IP', 'Destination_IP', 'Timestamp']

            # Convert all columns except exclude_cols to numeric (coerce errors to NaN)
            for col in df.columns:
                if col not in exclude_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows with missing values in important columns
            df = df.dropna(subset=required_cols + ['Label'])

            # Filter out invalid or negative values in key numeric columns
            df = df[df['Flow_Duration'] >= 0]
            df = df[(df['Total_Fwd_Packets'] >= 0) & (df['Total_Backward_Packets'] >= 0)]

            # Additional logical check if columns exist
            if 'Fwd_Packet_Length_Min' in df.columns and 'Fwd_Packet_Length_Max' in df.columns:
                df = df[df['Fwd_Packet_Length_Min'] <= df['Fwd_Packet_Length_Max']]

            # Create binary label: 0 for BENIGN, 1 for attack
            df['attack'] = df['Label'].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1)

            all_data.append(df)

    if not all_data:
        raise RuntimeError("No usable CSV files found for preprocessing.")

    combined_df = pd.concat(all_data, ignore_index=True)

    # Replace infinite and drop rows with NaN before scaling to avoid errors
    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined_df.dropna(inplace=True)

    # Scale numeric features to [-1, 1] for WGAN training, excluding the 'attack' label
    combined_df, scaler = scale_to_minus1_1(combined_df, exclude_cols=['attack'])

    # Save processed data
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    combined_df.to_csv(save_path, index=False)
    print(f"Saved WGAN-ready data to {save_path}")

    return combined_df, scaler


if __name__ == "__main__":
    data_folder = "data/TrafficLabelling/"
    df, scaler = preprocess_iscx_for_wgan(data_folder)
