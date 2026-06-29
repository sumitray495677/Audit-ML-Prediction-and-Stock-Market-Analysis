# Nifty50 Stock Market Value Prediction (LSTM)

A deep learning project that predicts the **Open price** of the Nifty 50 index (`^NSEI`) using an LSTM (Long Short-Term Memory) neural network trained on historical price and technical-indicator data.

## 📁 Project Structure

```
.
├── Nifty50_Dataset.csv              # Historical Nifty50 OHLCV + technical indicators
├── Datapreprocessing.py             # Data loading, scaling, sequence creation (module: dp)
├── Market_value_LSTM_model.py       # LSTM model training & evaluation script
├── Dataset.py                       # Dataset utilities
├── Stock_Future_value.py            # Future value prediction script
├── Market_Value.keras / .zip        # Saved trained model (weights + config)
└── README.md
```

> **Note:** Several files in this repository (`Datapreprocessing.py`, `Dataset.py`, `Stock_Future_value.py`, `Market_value_LSTM_model.py`) are currently **empty placeholders**. The working/reference logic exists in `Market_value_LSTM_model__1_.py`. Restore or rewrite the empty files before running the full pipeline.

## 📊 Dataset

`Nifty50_Dataset.csv` contains historical daily data for the Nifty 50 index with the following columns:

| Column | Description |
|---|---|
| `Date` | Trading date |
| `Open`, `High`, `Low`, `Close` | OHLC prices |
| `Volume` | Trading volume |
| `RSI` | Relative Strength Index |
| `MACD`, `MACD_Signal` | Moving Average Convergence Divergence |
| `ATR` | Average True Range |

## 🧠 Model Architecture

The LSTM model (defined in the training script) is a regression model that predicts the next `Open` price:

```python
value_model = Sequential([
    LSTM(units=152, input_shape=(SEQ_LEN, num_features)),
    Dropout(0.2),
    Dense(1)   # Regression output
])
```

- **Optimizer:** Adagrad (`learning_rate=0.001`)
- **Loss:** Mean Squared Error
- **Metric:** Mean Absolute Error (MAE)
- **Epochs:** 52
- **Batch size:** 8

## ⚙️ Pipeline

1. **Data Preprocessing** (`Datapreprocessing.py`)
   - Loads `Nifty50_Dataset.csv`
   - Scales features (e.g., MinMax/Standard scaling) via `dp.scaler`
   - Creates sequences of length `dp.SEQ_LEN` for time-series modeling
   - Splits data into `X_value_train`, `Y_value_train`, `X_value_test`, `Y_value_test`

2. **Model Training** (`Market_value_LSTM_model.py`)
   - Builds and trains the LSTM model on the preprocessed sequences
   - Validates on the held-out test set

3. **Evaluation**
   - Predictions are inverse-transformed back to original price scale
   - Metrics computed:
     - **RMSE** (Root Mean Squared Error)
     - **MAPE** (Mean Absolute Percentage Error)
     - **R² Score**
   - Actual vs Predicted prices plotted with `matplotlib`

4. **Model Saving**
   - Trained model is saved as `Market Value.keras`
   - Provided here as `Market_Value_keras.zip`, which unpacks to:
     - `metadata.json`
     - `config.json`
     - `model.weights.h5`

5. **Future Prediction** (`Stock_Future_value.py`)
   - Intended to load the saved model and predict future stock/index values (currently empty — needs implementation)

## 🚀 Getting Started

### Requirements
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```

### Run Training
```bash
python "Market_value_LSTM_model__1_.py"
```

### Load the Saved Model
```python
from tensorflow.keras.models import load_model

model = load_model("Market Value.keras")
predictions = model.predict(X_new)
```

## 📈 Output

The script prints evaluation metrics and displays a plot comparing actual vs. predicted Open prices:

```
RMSE: <value>
MAPE: <value>%
R² Score: <value>
```

## 🔧 TODO / Known Issues

- [ ] Populate empty files: `Datapreprocessing.py`, `Dataset.py`, `Stock_Future_value.py`, `Market_value_LSTM_model.py`
- [ ] Add a `requirements.txt`
- [ ] Add documentation for `SEQ_LEN` and feature selection used in preprocessing
- [ ] Implement future-value forecasting logic in `Stock_Future_value.py`


