# ml_system/__init__.py
"""
ML System for Trading Bot
Centralized machine learning module for trade prediction and optimization
"""
from .ml_manager import MLManager
from .predictor import MLPredictor
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer

__all__ = [
    'MLManager',
    'MLPredictor', 
    'FeatureEngineer',
    'ModelTrainer'
]

__version__ = '1.0.0'