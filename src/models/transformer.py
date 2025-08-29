import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        x = x + self.pos_enc[:, :x.size(1), :]
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=2, num_classes=2, dropout=0.1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.output_fc = nn.Linear(model_dim, num_classes)  # num_classes=2 for binary classification

    def forward(self, x):
        # x: [batch_size, input_dim]
        x = self.input_fc(x).unsqueeze(1)  # [batch_size, 1, model_dim]
        x = self.pos_encoder(x)            # add positional encoding
        x = self.transformer_encoder(x)    # [batch_size, 1, model_dim]
        x = x.mean(dim=1)                  # Global average pooling over sequence
        x = self.dropout(x)
        out = self.output_fc(x)            # [batch_size, num_classes]
        return out
