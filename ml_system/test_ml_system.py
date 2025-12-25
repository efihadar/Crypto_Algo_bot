# test_ml_system.py
import sys
sys.path.insert(0, 'E:/crypto_algo_bot')

try:
    from ml_system import MLManager, MLPredictor, FeatureEngineer
    from ml_system.config import ml_config
    
    print("✅ All imports successful!")
    print(f"   ML Enabled: {ml_config.enabled}")
    print(f"   Model Type: {ml_config.model_type}")
    print(f"   Feature Window: {ml_config.feature_window}")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()