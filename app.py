import os, sys, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Autoformer Stock Predictor", page_icon="📈", layout="wide")

DATA_PATH = "data"
MODEL_PATH_AF = "models/autoformer_uq_model.pth"
MODEL_PATH_LSTM = "models/lstm_model.pth"
MODEL_PATH_TRANS = "models/transformer_model.pth"
SEQ_LEN = 60
FORECAST = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Models ---
from src.lstm_model import LSTMModel
from src.transformer_model import StockTransformer as TransformerModel
from src.autoformer_model import StockAutoformer


# --- Load Models ---
@st.cache_resource
def load_models():
    # LSTM
    lstm_model = LSTMModel(input_size=4).to(device)
    if os.path.exists(MODEL_PATH_LSTM): lstm_model.load_state_dict(torch.load(MODEL_PATH_LSTM, map_location=device))
    lstm_model.eval()

    # Transformer
    trans_model = TransformerModel(input_size=4).to(device)
    if os.path.exists(MODEL_PATH_TRANS): trans_model.load_state_dict(torch.load(MODEL_PATH_TRANS, map_location=device))
    trans_model.eval()

    # Autoformer
    af_model = StockAutoformer(input_size=4).to(device)
    if os.path.exists(MODEL_PATH_AF): af_model.load_state_dict(torch.load(MODEL_PATH_AF, map_location=device))
    af_model.eval()

    return lstm_model, trans_model, af_model

# --- App Logic ---
@st.cache_data
def get_available_stocks():
    if not os.path.exists(DATA_PATH): return []
    return [f for f in sorted(os.listdir(DATA_PATH)) if f.endswith('.txt')]

@st.cache_data
def load_stock_data(stock_file):
    df = pd.read_csv(os.path.join(DATA_PATH, stock_file))
    if not {'Date','Close'}.issubset(df.columns): return None
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['Return'] = df['Close'].pct_change()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df = df.dropna().reset_index(drop=True)
    return df

st.title("📈 Multi-Model Live Comparison Dashboard")
st.markdown("""
Welcome to the live demo! This dashboard evaluates your exact chosen stock by running the 60-day historical window through all three models simultaneously.
* **LSTM**: Naive point predictor
* **Transformer**: Attention-based point predictor
* **Autoformer + UQ**: Probabilistic predictor with confidence bounds
""")

stocks = get_available_stocks()
if not stocks:
    st.error("No stock data found in the `data/` folder.")
    st.stop()

selected_stock = st.selectbox("Select a Stock to Predict:", stocks)

if st.button("Generate Forecasts", type="primary"):
    with st.spinner("Loading data and running all 3 models..."):
        df = load_stock_data(selected_stock)
        if df is None or len(df) < SEQ_LEN:
            st.error(f"Not enough valid data in {selected_stock} to generate a 60-day sequence.")
            st.stop()

        # Get the very last 60 days
        last_60_df = df.iloc[-SEQ_LEN:].copy()
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(last_60_df[['Close', 'Return', 'MA_10', 'MA_50']])
        
        # Prepare input tensor: shape (1, 60, 4)
        X_input = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Run inference
        lstm_model, trans_model, af_model = load_models()
        with torch.no_grad():
            mu_lstm = lstm_model(X_input)[0].cpu().numpy()
            mu_trans = trans_model(X_input)[0].cpu().numpy()
            mean_pred, log_var_pred = af_model(X_input)
            mu_af = mean_pred[0].cpu().numpy()
            std_af = torch.exp(0.5 * log_var_pred)[0].cpu().numpy()
        
        # Plotting
        st.subheader(f"Multi-Model Prediction for {selected_stock.replace('.txt', '').upper()}")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Historical plot (last 20 days for context)
        hist_idx = np.arange(-20, 0)
        hist_returns = scaled_data[-20:, 1] # Index 1 is Return
        ax.plot(hist_idx, hist_returns, marker='o', label="Historical Actual Return", color="#2ECC71", lw=2)
        
        # Future predictions plot
        future_idx = np.arange(0, FORECAST)
        
        # Connection indices
        conn_idx = [-1, 0]
        
        # 1. LSTM Plot
        ax.plot(conn_idx, [hist_returns[-1], mu_lstm[0]], color="#3498DB", lw=1.5, linestyle="--", alpha=0.6)
        ax.plot(future_idx, mu_lstm, marker='s', label="LSTM Point Prediction", color="#3498DB", lw=2)

        # 2. Transformer Plot
        ax.plot(conn_idx, [hist_returns[-1], mu_trans[0]], color="#9B59B6", lw=1.5, linestyle="--", alpha=0.6)
        ax.plot(future_idx, mu_trans, marker='^', label="Transformer Point Prediction", color="#9B59B6", lw=2)
        
        # 3. Autoformer Plot
        ax.plot(conn_idx, [hist_returns[-1], mu_af[0]], color="#E74C3C", lw=1.5, linestyle="--", alpha=0.6)
        ax.plot(future_idx, mu_af, marker='o', label="Autoformer Mean Prediction", color="#E74C3C", lw=2.5)
        
        # Autoformer Uncertainty Bands
        ax.fill_between(future_idx, mu_af - std_af, mu_af + std_af, color="#E74C3C", alpha=0.25, label="Autoformer +/- 1 Std Risk")
        ax.fill_between(future_idx, mu_af - 2*std_af, mu_af + 2*std_af, color="#E74C3C", alpha=0.10, label="Autoformer +/- 2 Std Risk")
        
        ax.axvline(x=-0.5, color="gray", linestyle="--", alpha=0.5)
        ax.text(-0.3, ax.get_ylim()[1]*0.95, "FUTURE →", color="gray", fontweight="bold")
        
        ax.set_title(f"Comparing LSTM vs Transformer vs Autoformer+UQ: {selected_stock}")
        ax.set_xlabel("Days (0 = Tomorrow)")
        ax.set_ylabel("Scaled Return")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
        
        st.pyplot(fig)
        
        # Display data table
        st.markdown("### Exact Predictions (Scaled Returns)")
        res_df = pd.DataFrame({
            "Day": [f"Day +{i+1}" for i in range(FORECAST)],
            "LSTM Prediction": [f"{m:.5f}" for m in mu_lstm],
            "Transformer Prediction": [f"{m:.5f}" for m in mu_trans],
            "Autoformer Prediction (Mean)": [f"{m:.5f}" for m in mu_af],
            "Autoformer Uncertainty (Std)": [f"{s:.5f}" for s in std_af],
        })
        st.dataframe(res_df, use_container_width=True)
        
        st.success("All models successfully inferred future data!")
