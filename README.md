# Deep Learning Stock Prediction

A deep learning project for predicting stock prices using various neural network architectures including LSTM, Transformer, and AutoFormer models.

## Project Structure

```
.
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── data/                       # Stock price data files (.txt format)
├── models/                     # Trained model weights and checkpoints
├── notebooks/                  # Jupyter notebooks and analysis scripts
│   ├── 03_autoformer_uq.ipynb # AutoFormer model with uncertainty quantification
│   └── make_notebooks.py       # Notebook generation utility
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── lstm_model.py          # LSTM-based prediction model
│   ├── transformer_model.py   # Transformer-based prediction model
│   └── autoformer_model.py    # AutoFormer-based prediction model
└── outputs/                    # Model outputs and predictions
```

## Models

### 1. LSTM Model
Traditional Long Short-Term Memory neural network for sequential time-series prediction.

### 2. Transformer Model
Modern transformer architecture adapted for stock price forecasting with attention mechanisms.

### 3. AutoFormer Model
Advanced forecasting architecture combining decomposition and autoregressive methods with uncertainty quantification.

## Requirements

Python 3.8+ with the following dependencies (see `requirements.txt`):
- PyTorch
- TensorFlow/Keras
- NumPy
- Pandas
- Scikit-learn
- Jupyter

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shruthi0914/deep_learning_stock_prediction.git
cd deep_learning_stock_prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Main Application
```bash
python app.py
```

### Training Models
Each model can be trained independently:
```bash
python src/lstm_model.py
python src/transformer_model.py
python src/autoformer_model.py
```

### Jupyter Notebooks
Access the analysis and model evaluation notebooks:
```bash
jupyter notebook notebooks/
```

## Data

Stock price data is stored in the `data/` directory as tab-separated text files with ticker symbols.

Format: `{ticker}.us.txt`
- Example: `aapl.us.txt`, `msft.us.txt`, `googl.us.txt`

## Results

Predictions and model outputs are saved in the `outputs/` directory for evaluation and visualization.

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.

## Author

Shruthi Raj - [@shruthi0914](https://github.com/shruthi0914)
