# ml_system/ml_manager.py
"""
Professional ML Manager - Central coordinator for all ML operations
Orchestrates feature engineering, training, predictions, and system health monitoring.
Thread-safe and production-ready for live trading environments.
"""
import pandas as pd
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime, timedelta
from loguru import logger
from .config import ml_config
from .data_storage import DataStorage 
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer
from .predictor import MLPredictor
from .drift_detector import DataDriftDetector

class MLManager:
    """
    Central ML Manager - Production Grade
    Coordinates all ML operations with robust error handling and monitoring.
    """
    
    def __init__(self):
        self.enabled = ml_config.enabled
        self.config = ml_config
        self.last_retrain_time: Optional[datetime] = None
        self.model_version: int = 0

        self.execution_filter_enabled = getattr(self.config, 'ENABLE_ML_EXECUTION_FILTER', False)
        
        # Initialize components (don't rely on globals)
        try:
            self.storage = DataStorage()
            self.feature_engineer = FeatureEngineer()
            self.trainer = ModelTrainer()
            self.predictor = MLPredictor()
            self.drift_detector = DataDriftDetector(threshold=0.05)
            
            logger.info("🤖 Initializing Professional ML Manager...")
            
            if self.enabled:
                self._initialize()
            else:
                logger.warning("⚠️ ML system is disabled by configuration")
                
        except Exception as e:
            logger.error(f"❌ Critical failure initializing ML components: {e}")
            self.enabled = False
    
    def _initialize(self):
        """Initialize ML system with validation and fallbacks"""
        try:
            # Validate config first
            if not hasattr(self.config, 'validate') or not self.config.validate():
                logger.error("❌ ML config validation failed - disabling ML system")
                self.enabled = False
                return
            
            # Load existing model if available
            if self._load_existing_model():
                logger.success("✅ ML system initialized with existing model")
            else:
                logger.info("ℹ️ No existing model found - will train when sufficient data available")
            
            # Load and display statistics
            stats = self._get_safe_statistics()
            logger.info(
                f"📊 ML Data Stats:\n"
                f"   Total trades: {stats['total_trades']}\n"
                f"   Win rate: {stats['win_rate']:.1f}%\n"
                f"   Avg PnL: ${stats['avg_pnl']:.2f}"
            )
            
            # Check if immediate retraining is needed
            if self._needs_immediate_retraining():
                logger.info("🎓 Immediate model retraining recommended")
                self._attempt_training()
            
            # Initialize drift detector with recent data if available
            self._initialize_drift_detector()
            
        except Exception as e:
            logger.error(f"❌ ML initialization failed: {e}")
            self.enabled = False
    
    def _load_existing_model(self) -> bool:
        """Attempt to load the most recent model"""
        try:
            if not self.trainer.load_latest_model():
                return False
            
            # Validate model before using
            if not hasattr(self.trainer, 'current_model') or self.trainer.current_model is None:
                logger.warning("⚠️ Loaded model is invalid")
                return False
            
            # Connect to predictor
            self.predictor.set_model(self.trainer.current_model)
            
            # Update metadata
            self.last_retrain_time = datetime.utcnow()
            self.model_version += 1
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing model: {e}")
            return False
    
    def _get_safe_statistics(self) -> Dict[str, Any]:
        """Get statistics with error handling"""
        try:
            stats = self.storage.get_statistics()
            return {
                'total_trades': stats.get('total_trades', 0),
                'profitable': stats.get('profitable', 0),
                'losing': stats.get('losing', 0),
                'win_rate': stats.get('win_rate', 0.0),
                'avg_pnl': stats.get('avg_pnl', 0.0),
            }
        except Exception as e:
            logger.warning(f"⚠️ Failed to get statistics: {e}")
            return {
                'total_trades': 0,
                'profitable': 0,
                'losing': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
            }
    
    def _needs_immediate_retraining(self) -> bool:
        """Check if immediate retraining is needed"""
        try:
            # Check if we have enough data
            trades = self.storage.load_trades()
            if len(trades) < self.config.min_samples_to_train:
                return False
            
            # Check if model exists
            if not hasattr(self.trainer, 'current_model') or self.trainer.current_model is None:
                return True
            
            # Check retraining interval
            if self.last_retrain_time is None:
                return True
            
            hours_since_retrain = (datetime.utcnow() - self.last_retrain_time).total_seconds() / 3600
            return hours_since_retrain >= self.config.retrain_interval_hours
            
        except Exception as e:
            logger.error(f"❌ Error checking retraining need: {e}")
            return False
    
    def _initialize_drift_detector(self):
        """Initialize drift detector with recent training data"""
        try:
            trades = self.storage.load_trades()
            if len(trades) < 50:
                return
            
            # Extract features from recent trades
            recent_features = []
            for trade in trades[-50:]:
                if 'features' in trade and isinstance(trade['features'], dict):
                    recent_features.append(trade['features'])
            
            if len(recent_features) < 20:
                return
            
            # Create DataFrame and fit drift detector
            df_features = pd.DataFrame(recent_features)
            self.drift_detector.fit(df_features)
            logger.debug("✅ Drift detector initialized with recent data")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize drift detector: {e}")
    
    def record_trade_outcome(
        self,
        signal: Dict,
        df: pd.DataFrame,
        entry_price: float,
        exit_price: float,
        pnl: float,
        sl_used: float,
        tp_used: float
    ) -> bool:
        """
        Record a completed trade for ML learning with comprehensive validation
        
        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            if not self.enabled:
                return False
            
            # Validate inputs
            if not isinstance(signal, dict):
                logger.error("❌ Invalid signal: must be dictionary")
                return False
            
            if not isinstance(df, pd.DataFrame) or df.empty:
                logger.error("❌ Invalid dataframe")
                return False
            
            # Extract features
            features = self.feature_engineer.extract_features(df, signal)
            if not features or not isinstance(features, dict):
                logger.warning("⚠️ Could not extract valid features from trade")
                return False
            
            # Calculate outcome metrics
            side = str(signal.get('side', 'BUY')).upper()
            if side == 'BUY':
                price_move_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0
            else:  # SELL
                price_move_pct = ((entry_price - exit_price) / entry_price * 100) if entry_price > 0 else 0.0
            
            # Build trade record with validation
            trade_record = {
                'symbol': str(signal.get('symbol', 'UNKNOWN')),
                'side': side,
                'entry_price': float(entry_price),
                'exit_price': float(exit_price),
                'sl_used': float(sl_used),
                'tp_used': float(tp_used),
                'pnl': float(pnl),
                'outcome': 'profit' if pnl > 0 else 'loss',
                'price_move_pct': float(price_move_pct),
                'signal_strength': float(signal.get('strength', 50.0)),
                'features': features,
                'timestamp': datetime.utcnow().isoformat(),
                'model_version': self.model_version,
            }
            
            # Validate critical fields
            if any(pd.isna([trade_record[k] for k in ['entry_price', 'exit_price', 'pnl']])):
                logger.error("❌ Trade record contains NaN values")
                return False
            
            # Save to storage
            if not self.storage.save_trade(trade_record):
                logger.error("❌ Failed to save trade to storage")
                return False
            
            logger.debug(
                f"📝 Recorded trade outcome: {trade_record['symbol']} "
                f"{'PROFIT' if pnl > 0 else 'LOSS'} ${pnl:.2f} "
                f"(Model v{self.model_version})"
            )
            
            # Check if we should retrain (async in production)
            self._check_and_retrain()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to record trade outcome: {e}")
            return False
    
    def predict_stops(self,signal: Dict,df: pd.DataFrame,entry_price: float) -> Optional[Tuple[float, float]]:
        """
        Predict optimal SL/TP using ML with comprehensive error handling
        Returns:
            Tuple of (sl_price, tp_price) or None if prediction unavailable
        """
        try:
            if not self.enabled:
                logger.debug("ML system disabled - skipping stop prediction")
                return None
            
            if self.predictor.model is None:
                logger.debug("No ML model loaded - skipping stop prediction")
                return None
            
            if not isinstance(df, pd.DataFrame) or df.empty:
                logger.error("❌ Invalid dataframe for stop prediction")
                return None
            
            if entry_price <= 0:
                logger.error(f"❌ Invalid entry price: {entry_price}")
                return None
            
            result = self.predictor.predict_optimal_stops(signal, df, entry_price)
            
            # Validate result
            if result is None:
                return None
            
            if isinstance(result, dict):
                sl_price = result.get('sl_price')
                tp_price = result.get('tp_price')
                
                # Apply hard boundaries from config
                side = signal.get('side', 'BUY').upper()
                
                if side == 'BUY':
                    # Validate BUY stops: SL < entry < TP
                    if not (sl_price < entry_price < tp_price):
                        logger.warning(f"❌ Invalid BUY stops: SL={sl_price:.4f}, Entry={entry_price:.4f}, TP={tp_price:.4f}")
                        return None
                    
                    # Apply percentage boundaries
                    sl_distance_pct = (entry_price - sl_price) / entry_price * 100
                    tp_distance_pct = (tp_price - entry_price) / entry_price * 100
                    
                    if not (self.config.sl_min_pct <= sl_distance_pct <= self.config.sl_max_pct):
                        logger.warning(f"❌ SL distance {sl_distance_pct:.2f}% outside bounds [{self.config.sl_min_pct}, {self.config.sl_max_pct}]")
                        return None
                    
                    if not (self.config.tp_min_pct <= tp_distance_pct <= self.config.tp_max_pct):
                        logger.warning(f"❌ TP distance {tp_distance_pct:.2f}% outside bounds [{self.config.tp_min_pct}, {self.config.tp_max_pct}]")
                        return None
                
                else:  # SELL
                    # Validate SELL stops: TP < entry < SL
                    if not (tp_price < entry_price < sl_price):
                        logger.warning(f"❌ Invalid SELL stops: TP={tp_price:.4f}, Entry={entry_price:.4f}, SL={sl_price:.4f}")
                        return None
                    
                    # Apply percentage boundaries
                    sl_distance_pct = (sl_price - entry_price) / entry_price * 100
                    tp_distance_pct = (entry_price - tp_price) / entry_price * 100
                    
                    if not (self.config.sl_min_pct <= sl_distance_pct <= self.config.sl_max_pct):
                        logger.warning(f"❌ SL distance {sl_distance_pct:.2f}% outside bounds [{self.config.sl_min_pct}, {self.config.sl_max_pct}]")
                        return None
                    
                    if not (self.config.tp_min_pct <= tp_distance_pct <= self.config.tp_max_pct):
                        logger.warning(f"❌ TP distance {tp_distance_pct:.2f}% outside bounds [{self.config.tp_min_pct}, {self.config.tp_max_pct}]")
                        return None
                
                return (float(sl_price), float(tp_price))
            
            elif isinstance(result, tuple) and len(result) == 2:
                return (float(result[0]), float(result[1]))
            
            return None
            
        except Exception as e:
            logger.error(f"❌ ML stop prediction failed: {e}")
            return None
    
    def analyze_trade(self, signal: Dict, df: pd.DataFrame) -> Optional[Dict]:

        try:
            if not self.enabled or self.predictor.model is None:
                return None
            
            analysis = self.predictor.analyze_trade_quality(signal, df)
            if analysis is None:
                return None
            
            analysis['model_version'] = self.model_version
            
            profit_prob = analysis.get('profit_probability', 0.5)
            confidence_threshold = self.config.min_prediction_confidence
            
            should_skip = profit_prob < confidence_threshold
            
            if should_skip:
                analysis['quality'] = "LOW"

                if self.execution_filter_enabled:
                    analysis['recommendation'] = "SKIP"
                    logger.info(f"🚫 ML Filter: Execution blocked (Prob: {profit_prob:.2%})")
                else:
                    analysis['recommendation'] = "ALLOW_WITHOUT_ML"
                    logger.debug(f"⚠️ ML Filter: Low confidence ({profit_prob:.2%}) but execution allowed by config")
            else:
                analysis['quality'] = "HIGH"
                analysis['recommendation'] = "EXECUTE"
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ ML trade analysis failed: {e}")
            return None
    
    def adjust_position_size(self, signal: Dict, df: pd.DataFrame, base_size: float) -> float:
        """
        Adjust position size based on ML confidence with safety limits
        Returns:
            Adjusted size within configured bounds
        """
        try:
            if not self.enabled or self.predictor.model is None:
                return float(base_size)
            
            if not isinstance(df, pd.DataFrame) or df.empty:
                logger.error("❌ Invalid dataframe for position sizing")
                return float(base_size)
            
            if base_size <= 0:
                logger.error(f"❌ Invalid base size: {base_size}")
                return float(base_size)
            
            features = self.feature_engineer.extract_features(df, signal)
            if not features or not isinstance(features, dict):
                return float(base_size)
            
            adjusted_size = self.predictor.should_adjust_position_size(features, base_size)
            
            # Apply absolute bounds
            min_size = base_size * self.config.min_position_multiplier
            max_size = base_size * self.config.max_position_multiplier
            
            final_size = max(min_size, min(float(adjusted_size), max_size))
            
            if final_size != adjusted_size:
                logger.debug(f"📏 Position size clamped from {adjusted_size:.4f} to {final_size:.4f}")
            
            return final_size
            
        except Exception as e:
            logger.error(f"❌ ML size adjustment failed: {e}")
            return float(base_size)
    
    def _check_and_retrain(self):
        """Check if retraining is needed and execute if appropriate"""
        try:
            if not self.enabled:
                return
            
            # Check if enough new data accumulated
            trades = self.storage.load_trades()
            if len(trades) < self.config.min_samples_to_train:
                logger.debug(f"ℹ️ Not enough trades for training: {len(trades)}")
                return
            
            # Check if retraining interval passed
            if not self._needs_immediate_retraining():
                return
            
            logger.info("🎓 Automatic retraining triggered")
            success = self._attempt_training()
            
            if success:
                # Re-initialize drift detector with new data
                self._initialize_drift_detector()
            
        except Exception as e:
            logger.error(f"❌ Retraining check failed: {e}")
    
    def _attempt_training(self) -> bool:
        """Attempt to train a new model with comprehensive error handling"""
        try:
            # Load trade history
            trades = self.storage.load_trades()
            if len(trades) < self.config.min_samples_to_train:
                logger.warning(
                    f"⚠️ Insufficient trades for training: "
                    f"{len(trades)} < {self.config.min_samples_to_train}"
                )
                return False
            
            # Prepare features and labels
            result = self.feature_engineer.prepare_features_for_training(trades)
            if result is None:
                logger.error("❌ Failed to prepare training data")
                return False
            
            X, y = result
            
            # Validate training data
            if len(X) != len(y):
                logger.error(f"❌ Feature-label mismatch: {len(X)} vs {len(y)}")
                return False
            
            if len(X) < self.config.min_samples_to_train:
                logger.error(f"❌ Insufficient valid samples after preparation: {len(X)}")
                return False
            
            # Check for class imbalance
            unique_classes = y.nunique() if hasattr(y, 'nunique') else len(set(y))
            if unique_classes < 2:
                logger.error("❌ Training data has only one class - cannot train classifier")
                return False
            
            # Train model
            metadata = self.trainer.train_model(X, y)
            if metadata is None:
                logger.error("❌ Model training failed")
                return False
            
            # Validate that training produced a usable model
            if not hasattr(self.trainer, 'current_model') or self.trainer.current_model is None:
                logger.error("❌ Training completed but no model was created")
                return False
            
            # Validate model performance
            test_accuracy = metadata.get('test_accuracy', 0.0)
            if test_accuracy < self.config.min_model_accuracy:
                logger.warning(
                    f"⚠️ Model accuracy {test_accuracy:.2%} below minimum {self.config.min_model_accuracy:.2%} - "
                    f"not activating new model"
                )
                return False
            
            # Connect new model to predictor
            self.predictor.set_model(self.trainer.current_model)
            
            # Update metadata
            self.last_retrain_time = datetime.utcnow()
            self.model_version += 1
            
            # Cleanup old models
            self.trainer.cleanup_old_models(keep_n=5)
            
            logger.success(
                f"✅ Model retraining completed successfully (v{self.model_version}) - "
                f"Accuracy: {test_accuracy:.2%}"
            )
            return True
            
        except Exception as e:
            logger.error(f"❌ Training attempt failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def force_retrain(self) -> bool:
        """Force immediate model retraining"""
        logger.info("🎓 Manual retraining triggered")
        return self._attempt_training()
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about ML system"""
        try:
            stats = self._get_safe_statistics()
            model_info = {}
            
            if hasattr(self.trainer, 'get_model_info'):
                model_info = self.trainer.get_model_info()
            
            return {
                'enabled': self.enabled,
                'model_version': self.model_version,
                'training_samples': stats['total_trades'],
                'profit_trades': stats['profitable'],
                'loss_trades': stats['losing'],
                'win_rate': stats['win_rate'],
                'avg_pnl': stats['avg_pnl'],
                'model_loaded': model_info.get('loaded', False),
                'models_trained': self.model_version,
                'last_training': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
                'needs_retraining': self._needs_immediate_retraining(),
                'model_metrics': model_info.get('metrics', {}),
                'feature_count': len(self.feature_engineer.get_feature_names()) if hasattr(self.feature_engineer, 'get_feature_names') else 0,
            }
        except Exception as e:
            logger.error(f"❌ Failed to get model stats: {e}")
            return {
                'enabled': self.enabled,
                'model_version': 0,
                'training_samples': 0,
                'profit_trades': 0,
                'loss_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'model_loaded': False,
                'models_trained': 0,
                'last_training': None,
                'needs_retraining': True,
                'model_metrics': {},
                'feature_count': 0,
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get ML system health status with detailed diagnostics"""
        try:
            stats = self.get_model_stats()
            
            # Determine health status
            issues = []
            warnings = []
            
            if not self.enabled:
                issues.append("ML system disabled by configuration")
            
            if not stats['model_loaded']:
                issues.append("No model loaded")
            
            if stats['training_samples'] < self.config.min_samples_to_train:
                issues.append(f"Insufficient training data: {stats['training_samples']}/{self.config.min_samples_to_train}")
            
            if stats['needs_retraining']:
                warnings.append("Model needs retraining")
            
            if stats['model_loaded']:
                test_accuracy = stats['model_metrics'].get('test_accuracy', 0.0)
                if test_accuracy < self.config.min_model_accuracy:
                    warnings.append(f"Model accuracy ({test_accuracy:.2%}) below minimum ({self.config.min_model_accuracy:.2%})")
            
            # Calculate health score
            if not issues and not warnings:
                health = "HEALTHY"
                score = 100
            elif not issues and warnings:
                health = "DEGRADED"
                score = 70
            elif issues and not warnings:
                health = "UNHEALTHY"
                score = 40
            else:
                health = "CRITICAL"
                score = 20
            
            return {
                'health': health,
                'score': score,
                'issues': issues,
                'warnings': warnings,
                'stats': stats,
                'timestamp': datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get system health: {e}")
            return {
                'health': 'CRITICAL',
                'score': 0,
                'issues': ['System health check failed'],
                'warnings': [],
                'stats': {},
                'timestamp': datetime.utcnow().isoformat(),
            }
    
    def export_model_report(self) -> str:
        """Generate detailed professional model report"""
        try:
            health = self.get_system_health()
            stats = health['stats']
            
            report_lines = [
                "=" * 80,
                "PROFESSIONAL ML SYSTEM REPORT",
                "=" * 80,
                f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
                f"System Health: {health['health']} (Score: {health['score']}/100)",
                "=" * 80,
            ]
            
            # System Status
            report_lines.extend([
                "\n🟢 SYSTEM STATUS:",
                f"   ML Enabled: {'YES' if self.enabled else 'NO'}",
                f"   Model Version: {stats.get('model_version', 0)}",
                f"   Last Training: {stats.get('last_training', 'Never')}",
                f"   Needs Retraining: {'YES' if stats.get('needs_retraining', True) else 'NO'}",
                f"   Features Count: {stats.get('feature_count', 0)}",
            ])
            
            # Training Data
            report_lines.extend([
                "\n📊 TRAINING DATA:",
                f"   Total Samples: {stats['training_samples']:,}",
                f"   Profitable Trades: {stats['profit_trades']:,} ({stats['win_rate']:.1f}%)",
                f"   Losing Trades: {stats['loss_trades']:,}",
                f"   Average PnL: ${stats['avg_pnl']:.2f}",
            ])
            
            # Model Performance
            if stats['model_loaded']:
                metrics = stats.get('model_metrics', {})
                report_lines.extend([
                    "\n🧠 MODEL PERFORMANCE:",
                    f"   Test Accuracy: {metrics.get('test_accuracy', 0):.2%}",
                    f"   Test Precision: {metrics.get('test_precision', 0):.2%}",
                    f"   Test Recall: {metrics.get('test_recall', 0):.2%}",
                    f"   Test F1-Score: {metrics.get('test_f1', 0):.2%}",
                ])
            else:
                report_lines.append("\n⚠️ NO MODEL CURRENTLY LOADED")
            
            # Issues and Warnings
            if health['issues']:
                report_lines.append("\n🔴 ISSUES:")
                for issue in health['issues']:
                    report_lines.append(f"   • {issue}")
            
            if health['warnings']:
                report_lines.append("\n🟡 WARNINGS:")
                for warning in health['warnings']:
                    report_lines.append(f"   • {warning}")
            
            # Configuration Summary
            report_lines.extend([
                "\n⚙️ CONFIGURATION SUMMARY:",
                f"   Min Samples to Train: {self.config.min_samples_to_train:,}",
                f"   Min Model Accuracy: {self.config.min_model_accuracy:.2%}",
                f"   Retrain Interval: {self.config.retrain_interval_hours} hours",
                f"   Confidence Threshold: {self.config.confidence_threshold:.2%}",
                f"   Feature Window: {self.config.feature_window} periods",
            ])
            
            report_lines.append("\n" + "=" * 80)
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate model report: {e}")
            return f"ERROR GENERATING REPORT: {str(e)}"
    
    def get_prediction_metadata(self) -> Dict[str, Any]:
        """Get metadata about current prediction capabilities"""
        try:
            return {
                'can_predict_stops': self.enabled and self.predictor.model is not None,
                'can_analyze_trades': self.enabled and self.predictor.model is not None,
                'can_adjust_positions': self.enabled and self.predictor.model is not None,
                'model_type': getattr(self.config, 'model_type', 'unknown'),
                'confidence_threshold': self.config.confidence_threshold,
                'feature_window': self.config.feature_window,
                'last_retrain': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
                'model_version': self.model_version,
            }
        except Exception as e:
            logger.error(f"❌ Failed to get prediction metadata: {e}")
            return {}

# Global ML Manager instance
try:
    ml_manager = MLManager() if ml_config.enabled else None
    if ml_config.enabled and ml_manager:
        logger.success("✅ ML Manager initialized successfully")
    elif ml_config.enabled:
        logger.error("❌ ML Manager initialization failed")
        ml_manager = None
except Exception as e:
    logger.error(f"❌ Critical failure creating ML Manager: {e}")
    ml_manager = None