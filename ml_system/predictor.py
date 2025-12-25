# ml_system/predictor.py
"""
Professional ML Predictor
Handles predictions using trained models with comprehensive validation and safety checks.
Thread-safe and production-ready for live trading environments.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any, List
from loguru import logger
import threading
from datetime import datetime
import time
from .config import ml_config
from .feature_engineering import feature_engineer
from .ml_safety_validator import safety_validator

class MLPredictor:
    """Professional ML predictor for trading with safety validation and monitoring"""
    
    def __init__(self):
        self.model = None
        self.feature_names: List[str] = []
        self.config = ml_config
        self._lock = threading.Lock()  # For thread safety
        self._prediction_cache = {}  # Simple cache for repeated predictions
        self._last_prediction_time = None
        self._total_predictions = 0
        self.feature_engineer = feature_engineer
        
        logger.info("🤖 ML Predictor initialized")
    
    def set_model(self, model: Any, feature_names: Optional[List[str]] = None):
        """Set the model to use for predictions with warm-up and validation"""
        with self._lock:
            self.model = model
            if feature_names:
                self.feature_names = feature_names

            # Validate model
            if not self._validate_model():
                logger.error("❌ Model validation failed")
                return

            # Warm-up prediction if possible
            try:
                if self.feature_names:
                    dummy_features = {name: 0.0 for name in self.feature_names}
                else:
                    # Get feature names from model if available
                    if hasattr(model, 'feature_names_in_'):
                        dummy_features = {name: 0.0 for name in model.feature_names_in_}
                        self.feature_names = list(model.feature_names_in_)
                    else:
                        dummy_features = {"dummy_feature": 0.0}
                
                X = self._prepare_features(dummy_features)
                if X is not None:
                    _ = self.model.predict(X)
                    logger.info("✅ Model warmed up successfully")
                    
                    # Test predict_proba if available
                    if hasattr(self.model, 'predict_proba'):
                        _ = self.model.predict_proba(X)
                        logger.debug("✅ Model probability prediction tested successfully")
                        
            except Exception as e:
                logger.warning(f"⚠️ Model warm-up failed: {e}")

            logger.info(f"🤖 ML Predictor: Model loaded - Type: {type(model).__name__}, Features: {len(self.feature_names)}")
    
    def _validate_model(self) -> bool:
        """Validate that the model is properly configured"""
        try:
            if self.model is None:
                return False
            
            # Check if model has required methods
            if not hasattr(self.model, 'predict'):
                logger.error("❌ Model missing 'predict' method")
                return False
            
            return True
        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}")
            return False
    
    def clear_model(self):
        """Clear currently loaded model and features"""
        with self._lock:
            self.model = None
            self.feature_names = []
            self._prediction_cache.clear()
            logger.info("🧹 ML Predictor: Model cleared")
    
    def predict_optimal_stops(self,signal: Dict[str, Any],df: pd.DataFrame,entry_price: float) -> Optional[Dict[str, Any]]:
        """
        Predict optimal stop loss and take profit levels with comprehensive safety validation
        
        Args:
            signal: Signal dictionary
            df: Price dataframe
            entry_price: Proposed entry price
        
        Returns:
            Dict with sl_price, tp_price, confidence, multipliers, timestamp or None
        """
        start_time = time.time()
        try:
            with self._lock:
                # Validate inputs
                if not self._validate_prediction_inputs(signal, df, entry_price):
                    return None
                
                if self.model is None:
                    logger.warning("⚠️ No model loaded for prediction")
                    return None
                
                # Create cache key
                cache_key = self._create_cache_key(signal, entry_price)
                if cache_key in self._prediction_cache:
                    logger.debug("🎯 Using cached prediction")
                    return self._prediction_cache[cache_key].copy()
                
                # Extract features
                features = feature_engineer.extract_features(df, signal)
                if not features or not isinstance(features, dict):
                    logger.warning("⚠️ Failed to extract features")
                    return None
                
                # Prepare feature vector
                X = self._prepare_features(features)
                if X is None:
                    return None
                
                # Get prediction probability
                confidence, is_profitable = self._get_prediction_confidence(X)
                
                # Calculate base stops from signal or use defaults
                side = str(signal.get('side', 'BUY')).upper()
                base_sl_pct = float(signal.get('sl_pct', self.config.sl_min_pct * 2))
                base_tp_pct = float(signal.get('tp_pct', self.config.tp_min_pct * 2))
                
                # Adjust based on prediction confidence
                sl_multiplier, tp_multiplier = self._calculate_adjustment_multipliers(
                    is_profitable, confidence, side
                )
                
                # Apply adjustment ranges
                sl_multiplier = np.clip(
                    sl_multiplier,
                    self.config.sl_adjustment_range[0],
                    self.config.sl_adjustment_range[1]
                )
                tp_multiplier = np.clip(
                    tp_multiplier,
                    self.config.tp_adjustment_range[0],
                    self.config.tp_adjustment_range[1]
                )
                
                # Calculate adjusted percentages
                adjusted_sl_pct = base_sl_pct * sl_multiplier
                adjusted_tp_pct = base_tp_pct * tp_multiplier
                
                # Apply hard boundaries from config
                adjusted_sl_pct = np.clip(adjusted_sl_pct, self.config.sl_min_pct, self.config.sl_max_pct)
                adjusted_tp_pct = np.clip(adjusted_tp_pct, self.config.tp_min_pct, self.config.tp_max_pct)
                
                # Calculate final prices
                if side == 'BUY':
                    sl_price = entry_price * (1 - adjusted_sl_pct / 100)
                    tp_price = entry_price * (1 + adjusted_tp_pct / 100)
                else:  # SELL
                    sl_price = entry_price * (1 + adjusted_sl_pct / 100)
                    tp_price = entry_price * (1 - adjusted_tp_pct / 100)
                
                # Apply safety validation if available
                if safety_validator:
                    validated_sl, validated_tp, is_safe, reason = safety_validator.validate_stops(
                        ml_sl=sl_price,
                        ml_tp=tp_price,
                        entry_price=entry_price,
                        side=side,
                        atr=features.get('atr_pct', 0) * entry_price if 'atr_pct' in features else None,
                        volatility=features.get('volatility', 0) * 100 if 'volatility' in features else None,
                        symbol=str(signal.get('symbol', 'UNKNOWN'))
                    )
                    
                    if not is_safe:
                        logger.warning(f"⚠️ Safety validation failed: {reason}")
                        # Still return the validated values (they were adjusted to be safe)
                        sl_price, tp_price = validated_sl, validated_tp
                    else:
                        sl_price, tp_price = validated_sl, validated_tp
                else:
                    # Basic validation
                    if not self._validate_final_prices(sl_price, tp_price, entry_price, side):
                        logger.error("❌ Invalid final prices after calculation")
                        return None
                
                # Calculate risk/reward ratio
                risk_reward_ratio = self._calculate_risk_reward_ratio(
                    sl_price, tp_price, entry_price, side
                )
                
                result = {
                    'sl_price': float(sl_price),
                    'tp_price': float(tp_price),
                    'confidence': float(confidence),
                    'multipliers': {
                        'sl': float(sl_multiplier),
                        'tp': float(tp_multiplier)
                    },
                    'timestamp': datetime.utcnow().isoformat(),
                    'is_profitable': bool(is_profitable),
                    'risk_reward_ratio': float(risk_reward_ratio) if risk_reward_ratio is not None else None,
                    'model_type': type(self.model).__name__,
                    'feature_count': len(features),
                    'processing_time_ms': (time.time() - start_time) * 1000,
                    'version': getattr(ml_config, '__version__', '1.0.0')
                }
                
                # Cache result
                self._prediction_cache[cache_key] = result.copy()
                if len(self._prediction_cache) > 1000:  # Limit cache size
                    self._prediction_cache.pop(next(iter(self._prediction_cache)))
                
                # Update statistics
                self._last_prediction_time = datetime.utcnow()
                self._total_predictions += 1
                
                rr_display = risk_reward_ratio if risk_reward_ratio is not None else 0.0
                logger.debug(
                    f"🎯 ML Stops: SL={sl_price:.4f} (x{sl_multiplier:.2f}), "
                    f"TP={tp_price:.4f} (x{tp_multiplier:.2f}), conf={confidence:.2%}, "
                    f"RR={rr_display:.2f}"
                )
                
                return result
                
        except Exception as e:
            logger.error(f"❌ Stop prediction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _validate_prediction_inputs(self, signal: Dict[str, Any], df: pd.DataFrame, entry_price: float) -> bool:
        """Validate prediction inputs"""
        try:
            if not isinstance(signal, dict):
                logger.error("❌ Invalid signal: must be dictionary")
                return False
            
            if not isinstance(df, pd.DataFrame):
                logger.error("❌ Invalid dataframe")
                return False
            
            if entry_price <= 0:
                logger.error(f"❌ Invalid entry price: {entry_price}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"❌ Input validation failed: {e}")
            return False
    
    def _create_cache_key(self, signal: Dict[str, Any], entry_price: float) -> str:
        """Create cache key for prediction caching"""
        try:
            side = str(signal.get('side', 'BUY')).upper()
            strength = str(signal.get('strength', 50))
            symbol = str(signal.get('symbol', 'UNKNOWN'))
            return f"{symbol}_{side}_{strength}_{entry_price:.4f}"
        except:
            return f"prediction_{int(time.time())}"
    
    def _get_prediction_confidence(self, X: np.ndarray) -> Tuple[float, bool]:
        """Get prediction confidence and profitability flag"""
        try:
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)[0]
                confidence = float(max(proba))
                is_profitable = np.argmax(proba) == 1
            else:
                prediction = self.model.predict(X)[0]
                confidence = 0.6  # Default confidence
                is_profitable = prediction == 1
            
            # Ensure confidence is within bounds
            confidence = np.clip(confidence, 0.0, 1.0)
            return confidence, bool(is_profitable)
            
        except Exception as e:
            logger.error(f"❌ Failed to get prediction confidence: {e}")
            return 0.6, False
    
    def _calculate_adjustment_multipliers(self, is_profitable: bool, confidence: float, side: str) -> Tuple[float, float]:
        """Calculate SL/TP adjustment multipliers based on confidence"""
        try:
            if is_profitable and confidence > self.config.confidence_threshold:
                # High confidence profit - can be more aggressive
                sl_multiplier = 1.0  # Keep SL tight
                # Extend TP based on confidence (more confidence = more aggressive)
                tp_multiplier = 1.0 + (confidence - 0.5) * 1.0  # More aggressive than before
            else:
                # Lower confidence - be more conservative
                sl_multiplier = 0.8  # Tighter SL
                tp_multiplier = 0.8  # Lower TP
            
            return float(sl_multiplier), float(tp_multiplier)
        except Exception as e:
            logger.error(f"❌ Failed to calculate adjustment multipliers: {e}")
            return 1.0, 1.0
    
    def _validate_final_prices(self, sl_price: float, tp_price: float, entry_price: float, side: str) -> bool:
        """Validate final prices are logical"""
        try:
            if side == 'BUY':
                if sl_price >= entry_price:
                    logger.error(f"❌ Invalid BUY stops: SL({sl_price}) >= Entry({entry_price})")
                    return False
                if tp_price <= entry_price:
                    logger.error(f"❌ Invalid BUY stops: TP({tp_price}) <= Entry({entry_price})")
                    return False
            else:  # SELL
                if sl_price <= entry_price:
                    logger.error(f"❌ Invalid SELL stops: SL({sl_price}) <= Entry({entry_price})")
                    return False
                if tp_price >= entry_price:
                    logger.error(f"❌ Invalid SELL stops: TP({tp_price}) >= Entry({entry_price})")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"❌ Final price validation failed: {e}")
            return False
    
    def _calculate_risk_reward_ratio(self, sl_price: float, tp_price: float, entry_price: float, side: str) -> Optional[float]:
        """Calculate risk/reward ratio"""
        try:
            if side == 'BUY':
                risk = entry_price - sl_price
                reward = tp_price - entry_price
            else:  # SELL
                risk = sl_price - entry_price
                reward = entry_price - tp_price
            
            if risk <= 0 or reward <= 0:
                return None
            
            return reward / risk
        except Exception:
            return None
    
    def analyze_trade_quality(self, signal: Dict[str, Any], df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze trade quality using ML with comprehensive insights
        """
        start_time = time.time()
        try:
            with self._lock:
                # Validate inputs
                if not isinstance(signal, dict) or not isinstance(df, pd.DataFrame):
                    logger.error("❌ Invalid inputs for trade analysis")
                    return None
                
                if self.model is None:
                    logger.warning("⚠️ No model loaded for trade analysis")
                    return None
                
                # Extract features
                features = feature_engineer.extract_features(df, signal)
                if not features or not isinstance(features, dict):
                    logger.warning("⚠️ Failed to extract features for trade analysis")
                    return None
                
                # Prepare feature vector
                X = self._prepare_features(features)
                if X is None:
                    return None
                
                # Get prediction
                try:
                    prediction = self.model.predict(X)[0]
                except Exception as e:
                    logger.error(f"❌ Prediction failed: {e}")
                    return None
                
                # Get confidence
                confidence, profit_prob = self._get_prediction_confidence(X)
                
                # Determine recommendation based on configurable thresholds
                if profit_prob >= self.config.confidence_threshold:
                    recommendation = "TAKE"
                    quality = "HIGH"
                elif profit_prob >= getattr(self.config, 'min_prediction_confidence', 0.5):
                    recommendation = "CONSIDER"
                    quality = "MEDIUM"
                else:
                    recommendation = "SKIP"
                    quality = "LOW"
                
                # Feature importance insights
                insights = []
                top_features = []
                
                if hasattr(self.model, 'feature_importances_'):
                    top_features = self._get_top_features(features, 5)  # Show top 5
                    for feat_name, feat_value, importance in top_features:
                        insights.append({
                            'feature': feat_name,
                            'value': float(feat_value),
                            'importance': float(importance),
                            'percent': f"{importance:.2%}"
                        })
                
                # Calculate expected value
                expected_value = self._calculate_expected_value(profit_prob, features, signal)
                
                result = {
                    'prediction': 'PROFIT' if prediction == 1 else 'LOSS',
                    'confidence': float(confidence),
                    'profit_probability': float(profit_prob),
                    'recommendation': recommendation,
                    'quality': quality,
                    'insights': insights,
                    'top_features': [
                        {'name': name, 'value': float(value), 'importance': float(imp)}
                        for name, value, imp in top_features
                    ],
                    'features_used': features,
                    'timestamp': datetime.utcnow().isoformat(),
                    'processing_time_ms': (time.time() - start_time) * 1000,
                    'expected_value': expected_value,
                    'model_info': {
                        'type': type(self.model).__name__,
                        'feature_count': len(self.feature_names) if self.feature_names else len(features)
                    }
                }
                
                logger.debug(
                    f"🔍 Trade Analysis: {recommendation} ({quality}) - "
                    f"Profit Prob: {profit_prob:.2%}, Confidence: {confidence:.2%}"
                )
                
                return result
                
        except Exception as e:
            logger.error(f"❌ Trade analysis failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _calculate_expected_value(self, profit_prob: float, features: Dict, signal: Dict) -> Optional[float]:
        """Calculate expected value of trade"""
        try:
            # Get risk/reward from features or signal
            if 'risk_reward_ratio' in features:
                rr_ratio = features['risk_reward_ratio']
            else:
                # Estimate from signal or use default
                rr_ratio = 2.0  # Default risk/reward ratio
            
            if rr_ratio <= 0:
                return None
            
            # Expected value formula: (win_prob * reward) - (loss_prob * risk)
            # Assuming risk = 1, reward = rr_ratio
            loss_prob = 1.0 - profit_prob
            expected_value = (profit_prob * rr_ratio) - (loss_prob * 1.0)
            
            return float(expected_value)
        except Exception:
            return None
    
    def should_adjust_position_size(self, features: Dict[str, Any], base_size: float) -> float:
        """
        Adjust position size based on ML confidence with safety limits
        Args:
            features: Extracted features
            base_size: Base position size
        Returns:Adjusted position size within configured bounds
        """
        try:
            with self._lock:
                if self.model is None:
                    logger.debug("⚠️ No model loaded, returning base size")
                    return float(base_size)
                
                if not isinstance(features, dict):
                    logger.error("❌ Invalid features for position sizing")
                    return float(base_size)
                
                if base_size <= 0:
                    logger.error(f"❌ Invalid base size: {base_size}")
                    return float(base_size)
                
                # Prepare feature vector
                X = self._prepare_features(features)
                if X is None:
                    return float(base_size)
                
                # Get confidence
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(X)[0]
                    profit_prob = proba[1] if len(proba) > 1 else proba[0]
                else:
                    logger.debug("⚠️ Model doesn't support predict_proba, returning base size")
                    return float(base_size)
                
                # Adjust size based on confidence with smooth transitions
                multiplier = self._calculate_position_multiplier(float(profit_prob))
                
                # Apply absolute bounds from config
                min_multiplier = self.config.min_position_multiplier
                max_multiplier = self.config.max_position_multiplier
                multiplier = np.clip(multiplier, min_multiplier, max_multiplier)
                
                adjusted_size = base_size * multiplier
                
                # Apply safety validation if available
                if safety_validator:
                    symbol = features.get('symbol', 'UNKNOWN') if isinstance(features, dict) else 'UNKNOWN'
                    balance = features.get('account_balance', 10000.0) if isinstance(features, dict) else 10000.0
                    
                    validated_size, is_safe, reason = safety_validator.validate_position_size(
                        ml_size=adjusted_size,
                        balance=float(balance),
                        symbol=str(symbol)
                    )
                    
                    if not is_safe:
                        logger.warning(f"⚠️ Position size validation: {reason}")
                    
                    final_size = float(validated_size)
                else:
                    final_size = float(adjusted_size)
                
                logger.debug(
                    f"📊 Position size adjusted: {base_size:.4f} -> {final_size:.4f} "
                    f"(x{multiplier:.2f}, prob={profit_prob:.2%})"
                )
                
                return final_size
                
        except Exception as e:
            logger.error(f"❌ Position size adjustment failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return float(base_size)
    
    def _calculate_position_multiplier(self, profit_prob: float) -> float:
        """Calculate position size multiplier based on profit probability"""
        try:
            # Smooth transition between multipliers
            if profit_prob >= 0.75:
                return 1.5  # Maximum aggression
            elif profit_prob >= 0.7:
                return 1.3
            elif profit_prob >= 0.65:
                return 1.1
            elif profit_prob >= 0.6:
                return 1.0  # Base size
            elif profit_prob >= 0.55:
                return 0.8
            elif profit_prob >= 0.5:
                return 0.6
            else:
                return 0.4  # Minimum size
                
        except Exception as e:
            logger.error(f"❌ Failed to calculate position multiplier: {e}")
            return 1.0
    
    def _prepare_features(self, features: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Prepare features for prediction as DataFrame with correct column names.
        Robust version: Handles feature mismatch using reindex (auto-fill missing with 0).
        """
        try:
            if not features or not isinstance(features, dict):
                logger.error("❌ Invalid features for preparation")
                return None
            
            # 1. Create initial DataFrame from current available features
            current_keys = list(features.keys())
            current_values = list(features.values())
            X_df = pd.DataFrame([current_values], columns=current_keys)
            
            # 2. Determine the target feature structure expected by the model
            expected_features = []
            
            if hasattr(self.model, 'feature_names_in_'):
                # Best source: The specific features the model was trained on
                expected_features = list(self.model.feature_names_in_)
            elif self.feature_names:
                # Fallback source: The feature names list stored in the class
                expected_features = self.feature_names
            
            # 3. Align DataFrame to the model's expectation
            if expected_features:
                # CRITICAL FIX: Use reindex instead of direct bracket selection.
                # - If a column is missing (model needs it, but we don't have it): Fills with 0.0
                # - If a column is extra (we have it, model doesn't need it): It gets dropped
                # - Ensures the order is exactly what the model expects
                X_df = X_df.reindex(columns=expected_features, fill_value=0.0)
            
            return X_df
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare features: {e}")
            return None
    
    def _get_top_features(self, features: Dict[str, Any], n: int = 3) -> List[Tuple[str, float, float]]:
        """Get top N important features with their values - ROBUST VERSION"""
        try:
            if not hasattr(self.model, 'feature_importances_'):
                return []
            
            importances = self.model.feature_importances_
            
            if hasattr(self.model, 'feature_names_in_'):
                feature_names = list(self.model.feature_names_in_)
            else:
                feature_names = list(features.keys())
            
            if len(feature_names) != len(importances):

                return []
            
            feature_data = []
            for name, imp in zip(feature_names, importances):

                value = features.get(name, 0.0)
                if isinstance(value, (int, float, np.number)):
                    feature_data.append((str(name), float(value), float(imp)))
            
            feature_data.sort(key=lambda x: x[2], reverse=True)
            
            return feature_data[:n]
            
        except Exception as e:
            logger.error(f"❌ Failed to get top features: {e}")
            return []
    
    def get_predictor_status(self) -> Dict[str, Any]:
        """Get current predictor status"""
        try:
            return {
                'model_loaded': self.model is not None,
                'feature_count': len(self.feature_names),
                'model_type': type(self.model).__name__ if self.model else None,
                'last_prediction': self._last_prediction_time.isoformat() if self._last_prediction_time else None,
                'total_predictions': self._total_predictions,
                'cache_size': len(self._prediction_cache),
                'thread_safe': True,
                'config_version': getattr(ml_config, '__version__', 'unknown')
            }
        except Exception as e:
            logger.error(f"❌ Failed to get predictor status: {e}")
            return {'error': str(e)}
    
    def clear_cache(self):
        """Clear prediction cache"""
        with self._lock:
            self._prediction_cache.clear()
            logger.debug("🧹 Prediction cache cleared")

# Global predictor instance
try:
    ml_predictor = MLPredictor()
    logger.success("✅ ML Predictor initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize ML Predictor: {e}")
    ml_predictor = None