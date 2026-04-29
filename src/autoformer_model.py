"""
autoformer_model.py  —  Autoformer + Uncertainty Quantification
================================================================
NOVELTY: Instead of a single point-prediction, the model outputs:
  - mean  : expected 5-day return
  - log_var: log of predicted variance (uncertainty)
Loss = Gaussian Negative Log-Likelihood (GNLL):
  L = 0.5 * [exp(-log_var) * (y - mean)^2  +  log_var]
This gives the model an honest incentive to be uncertain when it should be.
In finance, uncertainty bounds map directly to position sizing / risk.
"""

import os, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SEQ_LENGTH     = 60
LABEL_LENGTH   = 20
FORECAST_DAYS  = 5
NUM_STOCKS     = 50
TRAIN_RATIO    = 0.8
D_MODEL        = 128
N_HEAD         = 8
D_FF           = 256
NUM_ENC_LAYERS = 2
NUM_DEC_LAYERS = 1
DROPOUT        = 0.1
MOVING_AVG     = 25
INPUT_SIZE     = 4
TARGET_COL_INDEX = 1
LEARNING_RATE  = 3e-4
NUM_EPOCHS     = 20   # Phase 1: MSE (pure accuracy)
FINETUNE_EPOCHS = 5  # Phase 2: GNLL (add calibrated uncertainty)
BATCH_SIZE     = 64


# ── Series Decomposition ──────────────────────────────────────────────────────
class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size=MOVING_AVG):
        super().__init__()
        self.padding = nn.ReplicationPad1d((kernel_size - 1) // 2)
        self.avg     = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        x_t    = self.padding(x.permute(0, 2, 1))
        trend  = self.avg(x_t).permute(0, 2, 1)
        return x - trend, trend


# ── Auto-Correlation Attention ────────────────────────────────────────────────
class AutoCorrelation(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B, T_q, _ = query.shape
        _, T_k, _ = key.shape
        Q = self.q(query).view(B, T_q, self.n_head, self.d_head).permute(0,2,1,3)
        K = self.k(key  ).view(B, T_k, self.n_head, self.d_head).permute(0,2,1,3)
        V = self.v(value).view(B, T_k, self.n_head, self.d_head).permute(0,2,1,3)
        T   = max(T_q, T_k)
        Q_f = torch.fft.rfft(Q, n=T, dim=2)
        K_f = torch.fft.rfft(K, n=T, dim=2)
        corr = torch.fft.irfft(Q_f * torch.conj(K_f), n=T, dim=2)[:, :, :T_q, :]
        k    = max(1, int(math.log(T_q)))
        topk_vals, topk_lags = torch.topk(corr.mean(-1), k, dim=-1)
        weights = F.softmax(topk_vals, dim=-1)
        out = torch.zeros_like(Q)
        for i in range(k):
            lag = int(topk_lags[:,:,i].float().mean().item())
            w   = weights[:,:,i].unsqueeze(-1).unsqueeze(-1)
            Vr  = torch.roll(V, -lag, dims=2)[:, :, :T_q, :]
            out = out + w * Vr
        out = self.drop(out).permute(0,2,1,3).contiguous().view(B, T_q, -1)
        return self.out(out)


# ── Encoder Layer ─────────────────────────────────────────────────────────────
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout, moving_avg):
        super().__init__()
        self.attn    = AutoCorrelation(d_model, n_head, dropout)
        self.ff      = nn.Sequential(nn.Linear(d_model,d_ff), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(d_ff,d_model))
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x):
        res = x; x = self.drop(self.attn(x,x,x)); x,_ = self.decomp1(self.norm1(res+x))
        res = x; x = self.drop(self.ff(x));        x,_ = self.decomp2(self.norm2(res+x))
        return x


# ── Decoder Layer ─────────────────────────────────────────────────────────────
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout, moving_avg, out_ch):
        super().__init__()
        self.self_attn  = AutoCorrelation(d_model, n_head, dropout)
        self.cross_attn = AutoCorrelation(d_model, n_head, dropout)
        self.ff         = nn.Sequential(nn.Linear(d_model,d_ff), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(d_ff,d_model))
        self.decomp1    = SeriesDecomposition(moving_avg)
        self.decomp2    = SeriesDecomposition(moving_avg)
        self.decomp3    = SeriesDecomposition(moving_avg)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.trend_proj = nn.Linear(d_model, out_ch, bias=False)
        self.drop       = nn.Dropout(dropout)

    def forward(self, x, enc):
        res=x; x=self.drop(self.self_attn(x,x,x));  x,t1=self.decomp1(self.norm1(res+x))
        res=x; x=self.drop(self.cross_attn(x,enc,enc)); x,t2=self.decomp2(self.norm2(res+x))
        res=x; x=self.drop(self.ff(x));               x,t3=self.decomp3(self.norm3(res+x))
        return x, self.trend_proj(t1+t2+t3)


# ── Autoformer + UQ heads ─────────────────────────────────────────────────────
class StockAutoformer(nn.Module):
    """
    Outputs per-step (mean, log_var) for FORECAST_DAYS steps.
    Trained with Gaussian NLL loss — the novelty.
    """
    def __init__(self, input_size=INPUT_SIZE, d_model=D_MODEL, n_head=N_HEAD,
                 d_ff=D_FF, enc_layers=NUM_ENC_LAYERS, dec_layers=NUM_DEC_LAYERS,
                 dropout=DROPOUT, moving_avg=MOVING_AVG,
                 seq_len=SEQ_LENGTH, label_len=LABEL_LENGTH, pred_len=FORECAST_DAYS):
        super().__init__()
        self.label_len = label_len
        self.pred_len  = pred_len

        self.enc_embed = nn.Linear(input_size, d_model)
        self.dec_embed = nn.Linear(input_size, d_model)

        self.encoder   = nn.ModuleList([EncoderLayer(d_model,n_head,d_ff,dropout,moving_avg)
                                        for _ in range(enc_layers)])
        self.enc_norm  = nn.LayerNorm(d_model)

        self.decoder   = nn.ModuleList([DecoderLayer(d_model,n_head,d_ff,dropout,moving_avg,input_size)
                                        for _ in range(dec_layers)])
        self.dec_norm  = nn.LayerNorm(d_model)

        self.seasonal_proj = nn.Linear(d_model, input_size)

        # ── NOVELTY: two separate heads ──────────────────────────────────────
        # mean head  → predicted return per future day
        self.mean_head    = nn.Linear(d_model, pred_len)
        # log-var head → log(variance); separate weights so model can learn
        #                 to be uncertain independently of point prediction
        self.log_var_head = nn.Linear(d_model, pred_len)

    def forward(self, x):
        B, T, C = x.shape

        # Encoder
        enc = self.enc_embed(x)
        for layer in self.encoder:
            enc = layer(enc)
        enc = self.enc_norm(enc)

        # Decoder init: label window + zero forecast
        dec_inp = torch.cat([x[:, -self.label_len:, :],
                             torch.zeros(B, self.pred_len, C, device=x.device)], dim=1)
        mean_trend = x.mean(1, keepdim=True).expand(B, self.label_len+self.pred_len, C)

        dec = self.dec_embed(dec_inp)
        trend_acc = mean_trend.clone()
        for layer in self.decoder:
            dec, trend = layer(dec, enc)
            trend_acc = trend_acc + trend
        dec = self.dec_norm(dec)           # (B, L+P, d_model)

        # Pool the decoder output over time → (B, d_model)
        pooled = dec[:, -self.pred_len:, :].mean(dim=1)   # (B, d_model)

        mean    = self.mean_head(pooled)      # (B, pred_len)
        log_var = self.log_var_head(pooled)   # (B, pred_len)
        log_var = torch.clamp(log_var, -10, 10)   # numerical stability

        return mean, log_var


# ── Gaussian NLL Loss (the novelty loss) ─────────────────────────────────────
def gaussian_nll_loss(mean, log_var, target):
    """
    L = 0.5 * [exp(-log_var)*(y-mu)^2 + log_var]
    Minimising this makes the model:
      - accurate (small residual)
      - calibrated (honest about its uncertainty)
    """
    precision = torch.exp(-log_var)
    loss = 0.5 * (precision * (target - mean).pow(2) + log_var)
    return loss.mean()


# ── Data helpers ──────────────────────────────────────────────────────────────
def create_sequences(data, seq_length=SEQ_LENGTH, forecast_days=FORECAST_DAYS):
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_days + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+forecast_days, TARGET_COL_INDEX])
    return np.array(X), np.array(y)

def load_and_prepare_data(data_path, num_stocks=NUM_STOCKS):
    files, all_dfs = sorted(os.listdir(data_path))[:num_stocks], []
    for f in files:
        try:
            df = pd.read_csv(os.path.join(data_path, f))
            if not {"Date","Close"}.issubset(df.columns): continue
            df["Date"]   = pd.to_datetime(df["Date"])
            df           = df.sort_values("Date")
            df["Return"] = df["Close"].pct_change()
            df["MA_10"]  = df["Close"].rolling(10).mean()
            df["MA_50"]  = df["Close"].rolling(50).mean()
            df           = df.dropna().copy(); df["Stock"] = f
            if len(df) > SEQ_LENGTH + FORECAST_DAYS: all_dfs.append(df)
        except Exception as e: print(f"Skip {f}: {e}")
    if not all_dfs: raise ValueError("No valid files.")
    return pd.concat(all_dfs, ignore_index=True)

def build_dataset(combined_df):
    all_X, all_y = [], []
    for stock in combined_df["Stock"].unique():
        sdf    = combined_df[combined_df["Stock"]==stock].copy()
        scaled = MinMaxScaler().fit_transform(sdf[["Close","Return","MA_10","MA_50"]])
        Xs, ys = create_sequences(scaled)
        if len(Xs): all_X.append(Xs); all_y.append(ys)
    return np.concatenate(all_X), np.concatenate(all_y)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_model(model, X_t, y_t):
    model.eval()
    with torch.no_grad():
        mean, log_var = model(X_t)
    mu  = mean.cpu().numpy()
    std = torch.exp(0.5 * log_var).cpu().numpy()   # convert log_var → std
    yt  = y_t.cpu().numpy()

    print("\nPer-day evaluation (mean prediction):")
    for d in range(FORECAST_DAYS):
        mae  = mean_absolute_error(yt[:,d], mu[:,d])
        rmse = np.sqrt(mean_squared_error(yt[:,d], mu[:,d]))
        avg_std = std[:,d].mean()
        print(f"  Day+{d+1} → MAE:{mae:.5f}  RMSE:{rmse:.5f}  Avg-Uncertainty(±σ):{avg_std:.5f}")

    mae_all  = mean_absolute_error(yt.flatten(), mu.flatten())
    rmse_all = np.sqrt(mean_squared_error(yt.flatten(), mu.flatten()))
    print(f"\nOverall → MAE:{mae_all:.5f}  RMSE:{rmse_all:.5f}")
    return yt, mu, std, mae_all, rmse_all


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df = load_and_prepare_data(DATA_PATH, NUM_STOCKS)
    print("Shape:", df.shape)

    X, y = build_dataset(df)
    print(f"X:{X.shape}  y:{y.shape}")

    sp = int(TRAIN_RATIO * len(X))
    X_tr, X_te = X[:sp], X[sp:]
    y_tr, y_te = y[:sp], y[sp:]
    print(f"Train:{len(X_tr)}  Test:{len(X_te)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    Xt  = lambda a: torch.tensor(a, dtype=torch.float32).to(device)
    X_tr_t, X_te_t = Xt(X_tr), Xt(X_te)
    y_tr_t, y_te_t = Xt(y_tr), Xt(y_te)

    model = StockAutoformer().to(device)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    from torch.utils.data import TensorDataset, DataLoader
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)

    print(f"\nTraining Autoformer + UQ for {NUM_EPOCHS} epochs...")
    losses = []
    for epoch in range(NUM_EPOCHS):
        model.train(); total_loss = 0
        for xb, yb in loader:
            mean, log_var = model(xb)
            loss = gaussian_nll_loss(mean, log_var, yb)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total_loss += loss.item()
        avg = total_loss / len(loader)
        losses.append(avg); scheduler.step()
        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}]  GNLL Loss: {avg:.5f}  LR: {scheduler.get_last_lr()[0]:.2e}")

    print("\nEvaluating...")
    y_true, y_pred, y_std, mae, rmse = evaluate_model(model, X_te_t, y_te_t)

    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "models/autoformer_uq_model.pth")
    print("\nSaved → models/autoformer_uq_model.pth")

    # Plot 1: Training loss
    plt.figure(figsize=(8,4))
    plt.plot(losses, color="#4A90E2", linewidth=2)
    plt.title("Autoformer + UQ — Gaussian NLL Loss"); plt.xlabel("Epoch"); plt.ylabel("GNLL")
    plt.tight_layout(); plt.savefig("outputs/autoformer_uq_loss.png", dpi=150); plt.close()

    # Plot 2: Actual vs Predicted + Uncertainty bands (Day+1)
    n = min(300, len(y_true))
    idx = np.arange(n)
    mu_d1  = y_pred[:n, 0]
    std_d1 = y_std[:n, 0]
    act_d1 = y_true[:n, 0]

    plt.figure(figsize=(14,5))
    plt.plot(idx, act_d1, label="Actual Return", color="#2ECC71", linewidth=1.2)
    plt.plot(idx, mu_d1,  label="Predicted Mean", color="#E74C3C", linewidth=1.2, alpha=0.9)
    plt.fill_between(idx, mu_d1 - std_d1, mu_d1 + std_d1,
                     alpha=0.25, color="#E74C3C", label="±1σ Uncertainty")
    plt.fill_between(idx, mu_d1 - 2*std_d1, mu_d1 + 2*std_d1,
                     alpha=0.10, color="#E74C3C", label="±2σ Uncertainty")
    plt.title("Autoformer + UQ: Predicted Return with Confidence Bands — Day+1")
    plt.xlabel("Sample"); plt.ylabel("Scaled Return")
    plt.legend(loc="upper right"); plt.tight_layout()
    plt.savefig("outputs/autoformer_uq_bands.png", dpi=150); plt.close()

    # Plot 3: Per-day average uncertainty
    avg_std_per_day = y_std.mean(axis=0)
    plt.figure(figsize=(6,4))
    plt.bar([f"Day+{i+1}" for i in range(FORECAST_DAYS)], avg_std_per_day, color="#9B59B6")
    plt.title("Average Predicted Uncertainty per Forecast Day")
    plt.ylabel("Mean σ (uncertainty)"); plt.tight_layout()
    plt.savefig("outputs/autoformer_uq_per_day_uncertainty.png", dpi=150); plt.close()

    print("Plots saved → outputs/autoformer_uq_loss.png")
    print("           → outputs/autoformer_uq_bands.png")
    print("           → outputs/autoformer_uq_per_day_uncertainty.png")


def gnll(m, lv, y):
    return (0.5 * (torch.exp(-lv) * (y - m).pow(2) + lv)).mean()

if __name__ == "__main__":
    main()
