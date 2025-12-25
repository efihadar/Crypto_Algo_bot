# 🤖 ML System Documentation

## Overview

The ML system provides **intelligent trade optimization** by learning from historical trade outcomes. It analyzes 30+ features and predicts optimal stop-loss and take-profit levels.

---

## 🏗️ Architecture

```
ml_system/
├── __init__.py              # Package initialization
├── config.py                # ML configuration & settings
├── data_storage.py          # Trade history persistence
├── feature_engineering.py   # Feature extraction (30+ features)
├── model_training.py        # Model training & evaluation
├── predictor.py             # Predictions & optimization
└── ml_manager.py            # Central coordinator
```

---

## 🚀 Quick Start

### 1. Installation

```bash
pip install scikit-learn pandas numpy
# Optional: for better models
pip install xgboost lightgbm
```

### 2. Configuration

Add to `.env`:
```bash
ML_ENABLED=true
ML_USE_PREDICTIONS=true
ML_BLEND_WEIGHT=0.7
ML_MIN_SAMPLES=50
```

### 3. Usage

The ML system integrates automatically:

```python
from ml_system import MLManager

# Initialize (done in bot_main.py)
ml_manager = MLManager()

# Record trade outcomes (done in order_manager.py)
ml_manager.record_trade_outcome(
    signal=signal,
    df=price_df,
    entry_price=entry,
    exit_price=exit,
    pnl=pnl,
    sl_used=sl,
    tp_used=tp
)

# Get predictions (done in bot_main.py)
ml_sl, ml_tp = ml_manager.predict_stops(signal, df, entry_price)
```

---

## 📊 Features Extracted

### Price Features (6)
- Distance from recent high/low
- Position vs median/quartiles
- High-low range percentage

### Technical Indicators (10)
- EMA distances (fast, slow, long)
- EMA trend alignment
- RSI value & normalized
- ATR percentage
- MACD value, histogram, signal

### Volume Features (5)
- Volume ratio (current vs MA)
- Short-term volume trend
- Volume z-score
- Volume spike detection

### Signal Features (6)
- Signal side (BUY/SELL)
- Signal strength
- SL/TP distance percentages
- Risk/reward ratio
- Momentum flag

### Context Features (5)
- Price volatility
- 5/10/20 bar momentum
- Trend consistency

**Total: 32 features** per trade

---

## 🎯 How It Works

### 1. Data Collection
Every closed trade is stored with:
- All extracted features
- Entry/exit prices
- Actual PnL
- SL/TP used

### 2. Model Training
When sufficient data (default: 50+ trades):
- Features → DataFrame
- Labels: 0 (loss) or 1 (profit)
- Train/test split (80/20)
- Model training (Random Forest by default)
- Evaluation metrics calculated
- Model saved to disk

### 3. Prediction
For new trades:
- Extract current features
- Model predicts profit probability
- Adjust SL/TP based on confidence
- Blend with traditional strategy

### 4. Blending Strategy
```
final_sl = traditional_sl * (1 - weight) + ml_sl * weight
final_tp = traditional_tp * (1 - weight) + ml_tp * weight
```
Default weight: 0.7 (70% ML, 30% traditional)

---

## 📈 Model Performance

### Metrics Tracked
- **Accuracy**: % of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

### Example Output
```
🎓 Model trained successfully!
   Train accuracy: 78.5%
   Test accuracy: 72.3%
   Test F1: 68.9%
```

### Model Types Supported
- **Random Forest** (default) - Robust, interpretable
- **XGBoost** - High performance, gradient boosting
- **LightGBM** - Fast, efficient for large datasets

---

## 🔧 Configuration Reference

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_ENABLED` | `true` | Enable ML system |
| `ML_USE_PREDICTIONS` | `true` | Use predictions in trading |
| `ML_BLEND_WEIGHT` | `0.7` | ML vs traditional weight |
| `ML_MIN_SAMPLES` | `50` | Min trades to train |
| `ML_MAX_SAMPLES` | `1000` | Max trades to store |

### Training Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_RETRAIN_HOURS` | `24` | Hours between retraining |
| `ML_MODEL_TYPE` | `random_forest` | Model algorithm |
| `ML_TEST_SIZE` | `0.2` | Test set proportion |
| `ML_RANDOM_STATE` | `42` | Random seed |

### Quality Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_MIN_ACCURACY` | `0.55` | Min acceptable accuracy |
| `ML_CONFIDENCE_THRESHOLD` | `0.6` | Min confidence for action |

### Storage Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_DATA_DIR` | `./ml_data` | Data storage directory |
| `ML_MODELS_DIR` | `./ml_models` | Model storage directory |

---

## 🎮 Manual Operations

### Force Retraining
```python
ml_manager.force_retrain()
```

### Get Statistics
```python
stats = ml_manager.get_model_stats()
print(f"Win rate: {stats['win_rate']:.1f}%")
print(f"Samples: {stats['training_samples']}")
```

### Health Check
```python
health = ml_manager.get_system_health()
print(f"Status: {health['health']}")
print(f"Score: {health['score']}/100")
```

### Export Report
```python
report = ml_manager.export_model_report()
print(report)
```

---

## 📝 Data Files

### Trade History
Location: `ml_data/trades_history.json`

Format:
```json
[
  {
    "symbol": "BTCUSDT",
    "side": "BUY",
    "entry_price": 50000.0,
    "exit_price": 51000.0,
    "pnl": 95.5,
    "outcome": "profit",
    "features": { ... },
    "timestamp": "2025-01-15T10:30:00"
  }
]
```

### Model Files
Location: `ml_models/model_*.pkl`

Each model includes:
- `model_*.pkl` - Trained model
- `model_*_metadata.json` - Training info & metrics

---

## 🐛 Troubleshooting

### "No model loaded"
**Cause**: Not enough training data yet
**Solution**: Wait for 50+ completed trades

### "ML prediction unavailable"
**Cause**: Feature extraction failed
**Solution**: Check price data quality

### Low Model Accuracy
**Cause**: Insufficient or noisy data
**Solutions**:
1. Increase `ML_MIN_SAMPLES`
2. Trade more diverse market conditions
3. Check strategy signal quality

### Model Not Retraining
**Cause**: `ML_RETRAIN_HOURS` not reached
**Solution**: Use `ml_manager.force_retrain()`

---

## 🔬 Advanced Usage

### Custom Features
Edit `feature_engineering.py`:
```python
def _extract_custom_features(self, df: pd.DataFrame) -> Dict:
    return {
        'my_indicator': calculate_my_indicator(df),
        'my_ratio': df['close'].iloc[-1] / df['close'].mean()
    }
```

### Custom Model
Edit `model_training.py`:
```python
def _create_model(self):
    if self.model_type == 'my_model':
        return MyCustomModel(...)
```

### Position Size Adjustment
```python
# In bot_main.py
adjusted_size = ml_manager.adjust_position_size(
    signal, df, base_size
)
```

---

## 📊 Performance Tips

### 1. Data Quality
- Ensure clean, complete price data
- Filter out anomalous trades
- Use consistent timeframes

### 2. Training Frequency
- Too frequent: Overfitting risk
- Too rare: Stale model
- Recommended: 24-48 hours

### 3. Blend Weight
- Start with 0.5 (50/50)
- Increase to 0.7 as model improves
- Never exceed 0.9 (keep some traditional logic)

### 4. Sample Size
- Minimum: 50 trades
- Optimal: 200+ trades
- Maximum: 1000 trades (configurable)

---

## 🎯 Expected Results

### Phase 1: Initial Training (0-50 trades)
- No ML predictions yet
- Data collection only
- Traditional strategy runs

### Phase 2: First Model (50-200 trades)
- Basic predictions available
- Accuracy: 55-65%
- Conservative blending (50%)

### Phase 3: Mature Model (200+ trades)
- Reliable predictions
- Accuracy: 65-75%
- Aggressive blending (70%)

---

## 🔐 Safety Features

1. **Minimum Accuracy Check**: Won't use model below 55%
2. **Confidence Threshold**: Only acts on high-confidence predictions
3. **Blend Weight Limit**: Never 100% ML, always keeps traditional logic
4. **Graceful Degradation**: Falls back to traditional if ML fails
5. **Data Validation**: Rejects invalid/infinite features

---

## 📚 References

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [XGBoost](https://xgboost.readthedocs.io/)
- [LightGBM](https://lightgbm.readthedocs.io/)

---

## 🆘 Support

For issues or questions:
1. Check logs for ML-related errors
2. Review `ml_manager.export_model_report()`
3. Verify configuration in `.env`
4. Check data files in `ml_data/`

---

**Last Updated**: 2025-01-15
**Version**: 1.0.0