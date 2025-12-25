# ml_system/model_training.py
"""
Professional ML Model Training System
Handles model training, evaluation, validation, and persistence with production-grade reliability.
Supports multiple algorithms with automatic fallback and comprehensive monitoring.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List, Any
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

# Try to import ML libraries with graceful fallbacks
SKLEARN_AVAILABLE = False
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    SKLEARN_AVAILABLE = True
    logger.info("✅ scikit-learn loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ scikit-learn not available: {e}")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    logger.info("✅ XGBoost loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ XGBoost not available: {e}")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    logger.info("✅ LightGBM loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ LightGBM not available: {e}")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("⚠️ joblib not available, using pickle instead")

from .config import ml_config


class ModelTrainer:
    """Professional ML model trainer with comprehensive validation and monitoring"""
    
    def __init__(self):
        self.models_dir = Path(ml_config.models_dir)
        self.model_type = ml_config.model_type
        self.test_size = ml_config.test_size
        self.random_state = ml_config.random_state
        
        self.current_model = None
        self.scaler = None  # For feature scaling
        self.feature_selector = None  # For feature selection
        self.model_metadata = {}
        self.last_training_time = None
        self.training_history = []
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎯 Model Trainer initialized with model type: {self.model_type}")
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Train a new model on provided data with comprehensive validation
        
        Args:
            X: Feature DataFrame
            y: Labels Series (0 = loss, 1 = profit)
        
        Returns:
            Dictionary with training results or None on failure
        """
        try:
            if not SKLEARN_AVAILABLE:
                logger.error("❌ scikit-learn not available for training")
                return None
            
            # Validate input data
            if not self._validate_training_data(X, y):
                return None
            
            logger.info(f"🎓 Starting training of {self.model_type} model with {len(X)} samples and {len(X.columns)} features...")
            
            # Create copy to avoid modifying original data
            X_clean = X.copy()
            y_clean = y.copy()
            
            # Handle missing values
            X_clean = self._handle_missing_values(X_clean)
            
            # Scale features
            X_scaled = self._scale_features(X_clean)
            
            # Feature selection (optional)
            X_selected, selected_features = self._select_features(X_scaled, y_clean)
            
            # Split data with stratification
            X_train, X_test, y_train, y_test = self._split_data(X_selected, y_clean)
            
            # Create and train model
            model = self._create_model(len(X_selected.columns))
            if model is None:
                return None
            
            # Train with validation monitoring
            training_result = self._train_with_validation(model, X_train, X_test, y_train, y_test)
            if training_result is None:
                return None
            
            model, metrics, cv_scores = training_result
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(model, selected_features)
            
            # Check for overfitting
            overfitting_score = self._detect_overfitting(metrics)
            if overfitting_score > 0.2:  # More than 20% difference
                logger.warning(f"⚠️ Potential overfitting detected: {overfitting_score:.2%} difference between train/test")
            
            # Save model and metadata
            metadata = self._save_model(model, X_selected, y_clean, metrics, cv_scores, feature_importance, selected_features)
            if metadata is None:
                return None
            
            # Update instance state
            self.current_model = model
            self.model_metadata = metadata
            self.last_training_time = datetime.utcnow()
            
            # Add to training history
            self.training_history.append({
                'timestamp': self.last_training_time.isoformat(),
                'test_accuracy': metrics['test_accuracy'],
                'overfitting_score': overfitting_score,
                'model_path': metadata['model_path']
            })
            
            # Cleanup old models
            self.cleanup_old_models(keep_n=5)
            
            logger.success(
                f"✅ Model training completed successfully!\n"
                f"   Model: {self.model_type}\n"
                f"   Train Accuracy: {metrics['train_accuracy']:.2%}\n"
                f"   Test Accuracy: {metrics['test_accuracy']:.2%}\n"
                f"   CV Score: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}\n"
                f"   Overfitting: {overfitting_score:.2%}\n"
                f"   Saved to: {metadata['model_path']}"
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _validate_training_data(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Validate input data before training"""
        try:
            if len(X) < ml_config.min_samples_to_train:
                logger.warning(f"⚠️ Insufficient samples: {len(X)} < {ml_config.min_samples_to_train}")
                return False
            
            if len(X) != len(y):
                logger.error(f"❌ Feature-label mismatch: {len(X)} vs {len(y)}")
                return False
            
            if len(X.columns) == 0:
                logger.error("❌ No features provided")
                return False
            
            # Check for class imbalance
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                logger.error(f"❌ Only one class found in labels: {unique_classes}")
                return False
            
            class_counts = pd.Series(y).value_counts()
            minority_class_ratio = class_counts.min() / len(y)
            if minority_class_ratio < 0.1:  # Less than 10%
                logger.warning(f"⚠️ Severe class imbalance detected: minority class = {minority_class_ratio:.1%}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Data validation failed: {e}")
            return False
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        try:
            initial_missing = X.isnull().sum().sum()
            if initial_missing > 0:
                logger.warning(f"⚠️ Found {initial_missing} missing values in features")
            
            # Fill numeric columns with median
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if X[col].isnull().any():
                    median_val = X[col].median()
                    if pd.isna(median_val):
                        median_val = 0.0
                    X[col] = X[col].fillna(median_val)
            
            # Fill categorical/object columns with mode
            categorical_columns = X.select_dtypes(exclude=[np.number]).columns
            for col in categorical_columns:
                if X[col].isnull().any():
                    mode_val = X[col].mode()
                    if len(mode_val) > 0:
                        X[col] = X[col].fillna(mode_val[0])
                    else:
                        X[col] = X[col].fillna('Unknown')
            
            remaining_missing = X.isnull().sum().sum()
            if remaining_missing > 0:
                logger.error(f"❌ Still {remaining_missing} missing values after handling")
                return None
            
            if initial_missing > 0:
                logger.info(f"✅ Handled all {initial_missing} missing values")
            
            return X
            
        except Exception as e:
            logger.error(f"❌ Failed to handle missing values: {e}")
            return X
    
    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features for better model performance"""
        try:
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) == 0:
                return X
            
            # Create scaler
            self.scaler = StandardScaler()
            
            # Fit and transform
            X_scaled = X.copy()
            X_scaled[numeric_columns] = self.scaler.fit_transform(X[numeric_columns])
            
            logger.info(f"📊 Scaled {len(numeric_columns)} numeric features")
            return X_scaled
            
        except Exception as e:
            logger.error(f"❌ Failed to scale features: {e}")
            return X
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """Select most important features (optional)"""
        try:
            # For now, use all features - can be extended with SelectKBest or other methods
            selected_features = list(X.columns)
            return X, selected_features
            
        except Exception as e:
            logger.error(f"❌ Failed to select features: {e}")
            return X, list(X.columns)
    
    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data with stratification"""
        try:
            # Ensure we have enough samples for stratification
            unique_classes = np.unique(y)
            min_class_count = min(pd.Series(y).value_counts())
            
            if min_class_count >= 2:
                stratify_param = y
            else:
                stratify_param = None
                logger.warning("⚠️ Cannot stratify due to insufficient samples in some classes")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify_param
            )
            
            logger.info(
                f"SplitOptions: {len(X_train)} train, {len(X_test)} test "
                f"({self.test_size*100:.0f}% test)"
            )
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"❌ Failed to split data: {e}")
            # Fallback to simple split
            split_idx = int(len(X) * (1 - self.test_size))
            return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]
    
    def _create_model(self, n_features: int):
        """Create model instance based on config with optimal parameters"""
        try:
            common_params = {
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            if self.model_type == 'random_forest':
                return RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features='sqrt',
                    class_weight='balanced',
                    **common_params
                )
            
            elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
                return xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    eval_metric='logloss',
                    **common_params
                )
            
            elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                return lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    verbose=-1,
                    **common_params
                )
            
            else:
                # Fallback with warning
                actual_type = self.model_type
                if self.model_type == 'xgboost' and not XGBOOST_AVAILABLE:
                    logger.warning("⚠️ XGBoost not available, falling back to RandomForest")
                elif self.model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
                    logger.warning("⚠️ LightGBM not available, falling back to RandomForest")
                
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    **common_params
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to create model: {e}")
            return None
    
    def _train_with_validation(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                             y_train: pd.Series, y_test: pd.Series) -> Optional[Tuple]:
        """Train model with cross-validation and comprehensive evaluation"""
        try:
            # Perform cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            
            logger.info(f"🔍 Cross-validation scores: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate comprehensive metrics
            metrics = {
                'train_accuracy': accuracy_score(y_train, y_pred_train),
                'test_accuracy': accuracy_score(y_test, y_pred_test),
                'train_precision': precision_score(y_train, y_pred_train, zero_division=0),
                'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
                'train_recall': recall_score(y_train, y_pred_train, zero_division=0),
                'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
                'train_f1': f1_score(y_train, y_pred_train, zero_division=0),
                'test_f1': f1_score(y_test, y_pred_test, zero_division=0),
                'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
            }
            
            # Log detailed results
            logger.info(
                f"📈 Training Results:\n"
                f"   Train Accuracy: {metrics['train_accuracy']:.2%}\n"
                f"   Test Accuracy: {metrics['test_accuracy']:.2%}\n"
                f"   Test Precision: {metrics['test_precision']:.2%}\n"
                f"   Test Recall: {metrics['test_recall']:.2%}\n"
                f"   Test F1: {metrics['test_f1']:.2%}"
            )
            
            # Check minimum accuracy requirement
            if metrics['test_accuracy'] < ml_config.min_model_accuracy:
                logger.warning(
                    f"⚠️ Model accuracy {metrics['test_accuracy']:.2%} below minimum "
                    f"{ml_config.min_model_accuracy:.2%} - consider collecting more data"
                )
            
            return model, metrics, cv_scores
            
        except Exception as e:
            logger.error(f"❌ Training with validation failed: {e}")
            return None
    
    def _calculate_feature_importance(self, model, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Calculate feature importance if available"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                if len(importances) != len(feature_names):
                    logger.warning("⚠️ Feature importance length mismatch")
                    return None
                
                feature_importance = dict(zip(feature_names, importances))
                
                # Log top 10 features
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                logger.info("🔝 Top 10 Most Important Features:")
                for i, (feat, imp) in enumerate(sorted_features[:10]):
                    logger.info(f"   {i+1:2d}. {feat:<30} {imp:.4f}")
                
                return feature_importance
            elif hasattr(model, 'coef_'):
                # For linear models
                coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                if len(coef) == len(feature_names):
                    feature_importance = dict(zip(feature_names, np.abs(coef)))
                    return feature_importance
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate feature importance: {e}")
            return None
    
    def _detect_overfitting(self, metrics: Dict) -> float:
        """Detect potential overfitting"""
        try:
            train_acc = metrics.get('train_accuracy', 0)
            test_acc = metrics.get('test_accuracy', 0)
            overfitting_score = train_acc - test_acc
            return max(0, overfitting_score)
        except:
            return 0.0
    
    def _save_model(self, model, X: pd.DataFrame, y: pd.Series, metrics: Dict, 
                   cv_scores: np.ndarray, feature_importance: Optional[Dict], 
                   feature_names: List[str]) -> Optional[Dict[str, Any]]:
        """Save model and comprehensive metadata"""
        try:
            timestamp = datetime.utcnow().isoformat()
            safe_timestamp = timestamp.replace(':', '-').replace('.', '_')
            model_filename = f"model_{self.model_type}_{safe_timestamp}.{'joblib' if JOBLIB_AVAILABLE else 'pkl'}"
            model_path = self.models_dir / model_filename
            
            # Save model
            if JOBLIB_AVAILABLE:
                joblib.dump(model, model_path)
            else:
                import pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Prepare metadata
            metadata = {
                'model_type': self.model_type,
                'timestamp': timestamp,
                'n_samples': len(X),
                'n_features': len(X.columns),
                'feature_names': feature_names,
                'metrics': self._make_serializable(metrics),
                'cv_scores': self._make_serializable(cv_scores.tolist()),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'feature_importance': self._make_serializable(feature_importance),
                'model_path': str(model_path),
                'scaler_params': self._get_scaler_params() if self.scaler else None,
                'training_config': {
                    'test_size': self.test_size,
                    'random_state': self.random_state,
                    'min_samples_to_train': ml_config.min_samples_to_train,
                    'min_model_accuracy': ml_config.min_model_accuracy,
                },
                'data_stats': {
                    'target_distribution': y.value_counts().to_dict(),
                    'feature_ranges': {col: [float(X[col].min()), float(X[col].max())] for col in X.columns if X[col].dtype in [np.float64, np.float32, np.int64, np.int32]},
                }
            }
            
            # Save metadata
            metadata_path = model_path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.success(f"💾 Model and metadata saved successfully")
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            return None
    
    def _get_scaler_params(self) -> Optional[Dict]:
        """Get scaler parameters for reproducibility"""
        try:
            if self.scaler is None:
                return None
            
            return {
                'mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
                'scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
                'var': self.scaler.var_.tolist() if hasattr(self.scaler, 'var_') else None,
            }
        except:
            return None
    
    def load_latest_model(self) -> bool:
        """Load the most recent trained model with comprehensive validation"""
        try:
            # Find all model files
            model_extensions = ['.joblib', '.pkl']
            model_files = []
            
            for ext in model_extensions:
                model_files.extend([
                    f for f in os.listdir(self.models_dir)
                    if f.endswith(ext) and f.startswith('model_')
                ])
            
            if not model_files:
                logger.warning("⚠️  No trained models found")
                return False
            
            # Sort by modification time (most recent first)
            model_files.sort(
                key=lambda f: os.path.getmtime(self.models_dir / f),
                reverse=True
            )
            
            latest_model_file = model_files[0]
            model_path = self.models_dir / latest_model_file
            
            # Load model
            try:
                if JOBLIB_AVAILABLE:
                    self.current_model = joblib.load(model_path)
                else:
                    import pickle
                    with open(model_path, 'rb') as f:
                        self.current_model = pickle.load(f)
            except Exception as e:
                logger.error(f"❌ Failed to load model from {model_path}: {e}")
                return False
            
            # Load metadata
            metadata_path = model_path.with_suffix('.json')
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        self.model_metadata = json.load(f)
                    
                    self.feature_names = self.model_metadata.get('feature_names', [])
                    
                    # Load scaler if available
                    if 'scaler_params' in self.model_metadata and self.model_metadata['scaler_params']:
                        self._load_scaler(self.model_metadata['scaler_params'])
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load metadata: {e}")
            
            logger.success(f"✅ Successfully loaded model from: {model_path}")
            
            # Log comprehensive model info
            if self.model_metadata:
                metrics = self.model_metadata.get('metrics', {})
                logger.info(
                    f"📋 Model Information:\n"
                    f"   Type: {self.model_metadata.get('model_type', 'unknown')}\n"
                    f"   Trained on: {self.model_metadata.get('n_samples', 0):,} samples\n"
                    f"   Features: {self.model_metadata.get('n_features', 0)}\n"
                    f"   Test Accuracy: {metrics.get('test_accuracy', 0):.2%}\n"
                    f"   Test F1: {metrics.get('test_f1', 0):.2%}\n"
                    f"   CV Score: {self.model_metadata.get('cv_mean', 0):.2%} ± {self.model_metadata.get('cv_std', 0):.2%}"
                )
            
            # Set last training time
            if 'timestamp' in self.model_metadata:
                try:
                    self.last_training_time = datetime.fromisoformat(self.model_metadata['timestamp'])
                except:
                    self.last_training_time = datetime.utcnow()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def _load_scaler(self, scaler_params: Dict):
        """Load scaler from parameters"""
        try:
            if not SKLEARN_AVAILABLE:
                return
            
            self.scaler = StandardScaler()
            if scaler_params.get('mean'):
                self.scaler.mean_ = np.array(scaler_params['mean'])
            if scaler_params.get('scale'):
                self.scaler.scale_ = np.array(scaler_params['scale'])
            if scaler_params.get('var'):
                self.scaler.var_ = np.array(scaler_params['var'])
            
            logger.debug("✅ Scaler loaded successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load scaler: {e}")
    
    def needs_retraining(self) -> bool:
        """Check if model needs retraining based on age and performance"""
        try:
            if self.last_training_time is None:
                logger.debug("🔄 Model needs retraining: no previous training")
                return True
            
            # Check age
            age = datetime.utcnow() - self.last_training_time
            max_age = timedelta(hours=ml_config.retrain_interval_hours)
            
            if age > max_age:
                logger.debug(f"🔄 Model needs retraining: age {age} > max age {max_age}")
                return True
            
            # Check performance degradation (if we have history)
            if len(self.training_history) >= 2:
                recent_accuracy = self.training_history[-1]['test_accuracy']
                previous_accuracy = self.training_history[-2]['test_accuracy']
                if recent_accuracy < previous_accuracy * 0.9:  # 10% drop
                    logger.debug(f"🔄 Model needs retraining: performance dropped from {previous_accuracy:.2%} to {recent_accuracy:.2%}")
                    return True
            
            logger.debug("✅ Model does not need retraining")
            return False
            
        except Exception as e:
            logger.error(f"❌ Retraining check failed: {e}")
            return True  # Default to retraining on error
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about current model"""
        try:
            if self.current_model is None:
                return {
                    'loaded': False,
                    'message': 'No model loaded',
                    'can_train': SKLEARN_AVAILABLE,
                    'available_models': self._get_available_models()
                }
            
            info = {
                'loaded': True,
                'model_type': self.model_type,
                'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
                'needs_retraining': self.needs_retraining(),
                'can_retrain': SKLEARN_AVAILABLE,
                'available_models': self._get_available_models(),
            }
            
            if self.model_metadata:
                info.update({
                    'n_samples': self.model_metadata.get('n_samples', 0),
                    'n_features': self.model_metadata.get('n_features', 0),
                    'feature_names': self.model_metadata.get('feature_names', []),
                    'metrics': self.model_metadata.get('metrics', {}),
                    'cv_scores': {
                        'mean': self.model_metadata.get('cv_mean', 0),
                        'std': self.model_metadata.get('cv_std', 0),
                    },
                    'training_history': self.training_history[-5:] if len(self.training_history) > 0 else [],
                })
            
            return info
            
        except Exception as e:
            logger.error(f"❌ Failed to get model info: {e}")
            return {'error': str(e)}
    
    def _get_available_models(self) -> List[str]:
        """Get list of available model types"""
        available = ['random_forest']
        if XGBOOST_AVAILABLE:
            available.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            available.append('lightgbm')
        return available
    
    def _make_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def cleanup_old_models(self, keep_n: int = 5):
        """Remove old model files, keep only N most recent"""
        try:
            model_extensions = ['.joblib', '.pkl']
            all_model_files = []
            
            for ext in model_extensions:
                all_model_files.extend([
                    f for f in os.listdir(self.models_dir)
                    if f.endswith(ext) and f.startswith('model_')
                ])
            
            if len(all_model_files) <= keep_n:
                return
            
            # Sort by modification time
            all_model_files.sort(
                key=lambda f: os.path.getmtime(self.models_dir / f),
                reverse=True
            )
            
            # Remove old files
            removed_count = 0
            for old_file in all_model_files[keep_n:]:
                model_path = self.models_dir / old_file
                metadata_path = model_path.with_suffix('.json')
                
                try:
                    if model_path.exists():
                        model_path.unlink()
                        removed_count += 1
                        logger.debug(f"🗑️ Removed old model: {old_file}")
                    
                    if metadata_path.exists():
                        metadata_path.unlink()
                        logger.debug(f"🗑️ Removed metadata: {metadata_path.name}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to remove {old_file}: {e}")
            
            if removed_count > 0:
                logger.success(f"🧹 Cleaned up {removed_count} old models, kept {keep_n} most recent")
            
        except Exception as e:
            logger.error(f"❌ Model cleanup failed: {e}")
    
    def compare_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Compare different model types on the same dataset"""
        try:
            if not SKLEARN_AVAILABLE:
                return {"error": "scikit-learn not available"}
            
            logger.info("🔬 Starting model comparison...")
            
            # Validate data
            if not self._validate_training_data(X, y):
                return {"error": "Invalid training data"}
            
            # Handle missing values and scale features
            X_clean = self._handle_missing_values(X.copy())
            X_scaled = self._scale_features(X_clean)
            
            # Split data
            X_train, X_test, y_train, y_test = self._split_data(X_scaled, y)
            
            results = {}
            available_models = self._get_available_models()
            
            for model_type in available_models:
                logger.info(f"🧪 Testing {model_type}...")
                
                # Create model
                model = None
                if model_type == 'random_forest':
                    model = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
                elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
                    model = xgb.XGBClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
                elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                    model = lgb.LGBMClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1, verbose=-1)
                
                if model is None:
                    continue
                
                # Train and evaluate
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    results[model_type] = {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1),
                        'feature_importance': dict(zip(X.columns, model.feature_importances_)) if hasattr(model, 'feature_importances_') else None
                    }
                    
                    logger.info(f"   {model_type}: Accuracy={accuracy:.2%}, F1={f1:.2%}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to train {model_type}: {e}")
                    results[model_type] = {"error": str(e)}
            
            # Find best model
            best_model = None
            best_score = 0
            for model_type, metrics in results.items():
                if 'error' not in metrics:
                    score = metrics['f1_score']  # Use F1 as primary metric
                    if score > best_score:
                        best_score = score
                        best_model = model_type
            
            comparison_result = {
                'results': results,
                'best_model': best_model,
                'best_score': best_score,
                'recommendation': f"Use {best_model} for best performance" if best_model else "No clear winner"
            }
            
            logger.success(f"✅ Model comparison completed. Best model: {best_model} (F1={best_score:.2%})")
            return comparison_result
            
        except Exception as e:
            logger.error(f"❌ Model comparison failed: {e}")
            return {"error": str(e)}

# Global trainer instance
try:
    model_trainer = ModelTrainer()
    logger.success("✅ Model Trainer initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Model Trainer: {e}")
    model_trainer = None