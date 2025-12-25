# test_ml_trade.py
import pandas as pd
import numpy as np
from order_manager import _load_ml_manager

print("Loading ML Manager...")
ml = _load_ml_manager()

if not ml:
    print("ERROR: ML Manager not loaded!")
    exit(1)

print(f"ML Manager loaded, enabled={ml.enabled}")

# Fake price data
df = pd.DataFrame({
    'open': np.random.uniform(100, 105, 100),
    'high': np.random.uniform(105, 110, 100),
    'low': np.random.uniform(95, 100, 100),
    'close': np.random.uniform(100, 105, 100),
    'volume': np.random.uniform(1000, 5000, 100),
})

signal = {
    'symbol': 'BTCUSDT',
    'side': 'BUY',
    'price': 103.5,
    'stop_loss': 100.0,
    'take_profit': 110.0,
    'strength': 75,
}

# Record fake trade
print("\nRecording fake trade...")
success = ml.record_trade_outcome(
    signal=signal,
    df=df,
    entry_price=103.5,
    exit_price=108.0,
    pnl=4.5,
    sl_used=100.0,
    tp_used=110.0
)

print(f"Trade recorded: {success}")

# Check stats
stats = ml.get_model_stats()
print(f"Total trades in ML: {stats['training_samples']}")
print("\n✅ Test complete!")