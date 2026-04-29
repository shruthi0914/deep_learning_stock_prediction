import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------
# 1. Configuration
# -----------------------------
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SEQ_LENGTH      = 60       # look-back window: 60 days of history
FORECAST_DAYS   = 5        # predict next 5 days of returns
NUM_STOCKS      = 50       # full set of 50 stocks
TRAIN_RATIO     = 0.8

# Transformer hyperparameters
D_MODEL         = 128      # larger embedding dimension for richer sequence features
N_HEAD          = 8        # attention heads (128 / 8 = 16 dims per head)
NUM_LAYERS      = 3        # deeper encoder stack
DIM_FEEDFORWARD = 512      # FFN inner dim = 4 * d_model
DROPOUT         = 0.05

# Training
LEARNING_RATE   = 0.0003
NUM_EPOCHS      = 25
BATCH_SIZE      = 64
TARGET_COL_INDEX = 1       # 'Return' column in [Close, Return, MA_10, MA_50]


# -----------------------------
# 2. Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    """
    Injects position information into the sequence since Transformers
    have no built-in notion of order (unlike LSTMs).
    Uses sinusoidal encoding from 'Attention is All You Need'.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build a (max_len, d_model) matrix of positional values
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()          # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)   # even indices → sin
        pe[:, 1::2] = torch.cos(position * div_term)   # odd  indices → cos

        pe = pe.unsqueeze(0)                            # (1, max_len, d_model)
        self.register_buffer('pe', pe)                  # not a trainable param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# -----------------------------
# 3. Transformer Model
# -----------------------------
class StockTransformer(nn.Module):
    """
    Encoder-only Transformer for multi-step stock return forecasting.

    Input  shape: (batch, seq_len=60, input_size=4)
    Output shape: (batch, forecast_days=5)

    Architecture:
        1. Linear projection: input_size → d_model
        2. Positional Encoding
        3. TransformerEncoder (num_layers stacked encoder blocks)
           Each block = Multi-Head Self-Attention + FFN
        4. Take last timestep hidden state
        5. Linear: d_model → forecast_days
    """
    def __init__(
        self,
        input_size: int = 4,
        d_model: int = D_MODEL,
        nhead: int = N_HEAD,
        num_layers: int = NUM_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
        forecast_days: int = FORECAST_DAYS,
    ):
        super().__init__()

        # Project raw features (4) up to d_model (64)
        self.input_proj = nn.Linear(input_size, d_model)

        # Add positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # Stack of Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,       # input is (batch, seq, feature) not (seq, batch, feature)
            activation='gelu',
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final regression head: predict all 5 future returns at once
        self.fc = nn.Linear(d_model, forecast_days)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 60, 4)
        x = self.input_proj(x)       # (B, 60, 64)
        x = self.pos_encoding(x)     # (B, 60, 64)  ← position info added
        x = self.encoder(x)          # (B, 60, 128) ← all 60 days attend to each other
        x = x.mean(dim=1)            # (B, 128)     ← pool full sequence
        x = self.fc(x)               # (B, 5)       ← predict 5 future returns
        return x


# -----------------------------
# 4. Sequence creation (5-day target)
# -----------------------------
def create_sequences(data: np.ndarray, seq_length: int = 60, forecast_days: int = 5):
    """
    Creates sliding window sequences.

    Each sample uses 60 days of history to predict the next 5 days of Return.

    - X[i] = data[i : i+seq_length]          → shape (60, 4)
    - y[i] = data[i+seq_length : i+seq_length+forecast_days, TARGET_COL_INDEX]
                                              → shape (5,)

    We stop seq_length + forecast_days before the end so every sample
    has a full 5-day future target window available.
    """
    X, y = [], []

    for i in range(len(data) - seq_length - forecast_days + 1):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length : i + seq_length + forecast_days, TARGET_COL_INDEX])

    return np.array(X), np.array(y)


# -----------------------------
# 5. Load and preprocess stocks
# -----------------------------
def load_and_prepare_data(data_path: str, num_stocks: int = 50):
    files = sorted(os.listdir(data_path))[:num_stocks]
    all_dfs = []

    for file in files:
        file_path = os.path.join(data_path, file)
        try:
            df = pd.read_csv(file_path)

            required_cols = {"Date", "Close"}
            if not required_cols.issubset(df.columns):
                continue

            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")

            df["Return"] = df["Close"].pct_change()
            df["MA_10"]  = df["Close"].rolling(10).mean()
            df["MA_50"]  = df["Close"].rolling(50).mean()

            df = df.dropna().copy()
            df["Stock"] = file

            # Need at least seq_length + forecast_days rows
            if len(df) > SEQ_LENGTH + FORECAST_DAYS:
                all_dfs.append(df)

        except Exception as e:
            print(f"Skipping {file} due to error: {e}")

    if not all_dfs:
        raise ValueError("No valid stock files found.")

    return pd.concat(all_dfs, axis=0, ignore_index=True)


# -----------------------------
# 6. Build multi-stock dataset
# -----------------------------
def build_dataset(combined_df: pd.DataFrame):
    all_X, all_y = [], []

    for stock in combined_df["Stock"].unique():
        stock_df = combined_df[combined_df["Stock"] == stock].copy()
        feature_df = stock_df[["Close", "Return", "MA_10", "MA_50"]]

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(feature_df)

        X_stock, y_stock = create_sequences(scaled_data, SEQ_LENGTH, FORECAST_DAYS)

        if len(X_stock) > 0:
            all_X.append(X_stock)
            all_y.append(y_stock)

    X = np.concatenate(all_X, axis=0)   # (N, 60, 4)
    y = np.concatenate(all_y, axis=0)   # (N, 5)

    return X, y


# -----------------------------
# 7. Train function (mini-batch)
# -----------------------------
def train_model(model, X_train_t, y_train_t, criterion, optimizer, num_epochs=20, batch_size=64):
    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(X_train_t, y_train_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in loader:
            outputs = model(X_batch)                    # (batch, 5)
            loss    = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.6f}")

    return train_losses


# -----------------------------
# 8. Evaluate function
# -----------------------------
def evaluate_model(model, X_test_t, y_test_t):
    model.eval()

    with torch.no_grad():
        y_pred_t = model(X_test_t)

    y_pred = y_pred_t.cpu().numpy()   # (N, 5)
    y_true = y_test_t.cpu().numpy()   # (N, 5)

    # Compute MAE and RMSE for each future day separately
    print("\nPer-day evaluation:")
    for day in range(FORECAST_DAYS):
        mae  = mean_absolute_error(y_true[:, day], y_pred[:, day])
        rmse = np.sqrt(mean_squared_error(y_true[:, day], y_pred[:, day]))
        print(f"  Day+{day+1} → MAE: {mae:.6f}  RMSE: {rmse:.6f}")

    # Overall averaged metrics
    mae_avg  = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    rmse_avg = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    print(f"\nOverall → MAE: {mae_avg:.6f}  RMSE: {rmse_avg:.6f}")

    return y_true, y_pred, mae_avg, rmse_avg


# -----------------------------
# 9. Main pipeline
# -----------------------------
def main():
    print("Loading and preparing data...")
    combined_df = load_and_prepare_data(DATA_PATH, NUM_STOCKS)
    print("Combined dataframe shape:", combined_df.shape)

    print("Building sequences...")
    X, y = build_dataset(combined_df)
    print(f"X shape: {X.shape}")   # (N, 60, 4)
    print(f"y shape: {y.shape}")   # (N, 5)

    split_idx = int(TRAIN_RATIO * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)   # (N, 5)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32).to(device)   # (N, 5)

    model = StockTransformer(
        input_size      = X_train.shape[2],
        d_model         = D_MODEL,
        nhead           = N_HEAD,
        num_layers      = NUM_LAYERS,
        dim_feedforward = DIM_FEEDFORWARD,
        dropout         = DROPOUT,
        forecast_days   = FORECAST_DAYS,
    ).to(device)

    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print("\nTraining Transformer...")
    train_losses = train_model(
        model, X_train_t, y_train_t, criterion, optimizer,
        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE
    )

    print("\nEvaluating Transformer...")
    y_true, y_pred, mae, rmse = evaluate_model(model, X_test_t, y_test_t)

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/transformer_model.pth")
    print("\nModel saved to models/transformer_model.pth")

    # Plot training loss
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses)
    plt.title("Transformer Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig("outputs/transformer_training_loss.png")
    plt.show()

    # Plot actual vs predicted for Day+1
    plt.figure(figsize=(10, 5))
    plt.plot(y_true[:500, 0], label="Actual Return (Day+1)")
    plt.plot(y_pred[:500, 0], label="Predicted Return (Day+1)")
    plt.title("Transformer: Actual vs Predicted Return — Day+1")
    plt.xlabel("Sample")
    plt.ylabel("Scaled Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/transformer_actual_vs_predicted.png")
    plt.show()


if __name__ == "__main__":
    main()
