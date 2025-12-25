# ml_system/config.py
"""
ML System Configuration — Professional Trading Edition
Centralized, validated, immutable configuration for ML trading components.
"""
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from loguru import logger
import configparser

@dataclass(frozen=True)
class MLConfig:
    """Immutable ML Trading Configuration — Safe for live trading."""

    # ═══════════════════════════════════════════════════════════
    # MASTER SWITCHES
    # ═══════════════════════════════════════════════════════════
    enabled: bool = True
    use_ml_predictions: bool = True  # New from your INI

    # ═══════════════════════════════════════════════════════════
    # PATHS — Auto-resolved
    # ═══════════════════════════════════════════════════════════
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.resolve())
    models_dir: Path = field(init=False)
    data_dir: Path = field(init=False)

    # ═══════════════════════════════════════════════════════════
    # MODEL SETTINGS
    # ═══════════════════════════════════════════════════════════
    model_type: str = "random_forest"
    test_size: float = 0.2
    random_state: int = 42
    ml_blend_weight: float = 0.7  # New: weight of ML signal vs other signals

    # ═══════════════════════════════════════════════════════════
    # TRAINING THRESHOLDS
    # ═══════════════════════════════════════════════════════════
    min_samples_to_train: int = 100
    min_model_accuracy: float = 0.55
    retrain_interval_hours: int = 24

    # ═══════════════════════════════════════════════════════════
    # PREDICTION SETTINGS
    # ═══════════════════════════════════════════════════════════
    confidence_threshold: float = 0.6
    min_prediction_confidence: float = 0.6 
    feature_window: int = 50
    use_technical_indicators: bool = True
    use_volume_features: bool = True
    use_price_features: bool = True
    lookback_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])

    # ═══════════════════════════════════════════════════════════
    # POSITION SIZING & RISK
    # ═══════════════════════════════════════════════════════════
    max_position_multiplier: float = 1.5
    min_position_multiplier: float = 0.3
    max_daily_trades: int = 15
    max_drawdown_pct: float = 3.0
    require_confirmation: bool = True

    # ═══════════════════════════════════════════════════════════
    # STOP LOSS / TAKE PROFIT — HARD BOUNDARIES (in %)
    # ═══════════════════════════════════════════════════════════
    sl_min_pct: float = 0.5      # Minimum allowed SL distance from entry (%)
    sl_max_pct: float = 3.0      # Maximum allowed SL distance from entry (%)
    tp_min_pct: float = 1.0      # Minimum allowed TP distance from entry (%)
    tp_max_pct: float = 5.0      # Maximum allowed TP distance from entry (%)
    min_risk_reward: float = 1.5 # Min RR ratio (TP/SL) to allow trade
    max_risk_reward: float = 5.0 # Max RR ratio to avoid overextension

    # ═══════════════════════════════════════════════════════════
    # ATR-BASED DYNAMIC BOUNDARIES
    # ═══════════════════════════════════════════════════════════
    enable_atr_bounds: bool = True
    sl_min_atr_multiplier: float = 0.5
    sl_max_atr_multiplier: float = 2.5
    tp_min_atr_multiplier: float = 1.0
    tp_max_atr_multiplier: float = 4.0

    # ═══════════════════════════════════════════════════════════
    # ML STOP ADJUSTMENT RANGES — REQUIRED BY PREDICTOR
    # ═══════════════════════════════════════════════════════════
    sl_adjustment_range: Tuple[float, float] = field(default_factory=lambda: (0.8, 2.0))
    tp_adjustment_range: Tuple[float, float] = field(default_factory=lambda: (1.5, 4.0))
    
    def __post_init__(self):
        object.__setattr__(self, 'models_dir', self.base_dir / "models")
        object.__setattr__(self, 'data_dir', self.base_dir / "data")

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if not self.validate():
            raise ValueError("MLConfig validation failed — check logs for details.")

    def validate(self) -> bool:
        """
        Comprehensive validation of ML configuration parameters.
        Returns True if all validations pass, False otherwise.
        Logs detailed error messages for any failed validations.
        """
        errors = []

        def _err(msg: str):
            errors.append(msg)
            logger.error(f"❌ CONFIG VALIDATION FAILED: {msg}")

        # --- Basic parameter validations ---
        if self.min_samples_to_train < 20:
            _err("min_samples_to_train must be >= 20")

        if not (0.01 <= self.test_size <= 0.5):
            _err("test_size must be between 0.01 and 0.5")

        if not (0.5 <= self.min_model_accuracy <= 0.95):
            _err("min_model_accuracy must be between 0.5 and 0.95")

        if self.model_type not in {'random_forest', 'xgboost', 'lightgbm'}:
            _err(f"Unsupported model_type: {self.model_type}")

        if self.feature_window < 20:
            _err("feature_window must be >= 20")

        if not (0.0 <= self.confidence_threshold <= 1.0):
            _err("confidence_threshold must be between 0.0 and 1.0")

        if not (0.0 < self.ml_blend_weight <= 1.0):
            _err("ml_blend_weight must be between 0.0 and 1.0")

        # --- Stop Loss / Take Profit percentage boundaries ---
        if not (0.0 < self.sl_min_pct < self.sl_max_pct):
            _err("SL boundaries invalid: sl_min_pct < sl_max_pct required")

        if not (0.0 < self.tp_min_pct < self.tp_max_pct):
            _err("TP boundaries invalid: tp_min_pct < tp_max_pct required")

        if not (1.0 <= self.min_risk_reward <= self.max_risk_reward <= 10.0):
            _err("Risk-reward boundaries invalid: min_risk_reward <= max_risk_reward and within [1.0, 10.0]")

        # --- ATR Multiplier validations ---
        if not (self.sl_min_atr_multiplier > 0 and self.sl_min_atr_multiplier <= self.sl_max_atr_multiplier):
            _err("Invalid SL ATR multiplier range: must be positive and min <= max")

        if not (self.tp_min_atr_multiplier > 0 and self.tp_min_atr_multiplier <= self.tp_max_atr_multiplier):
            _err("Invalid TP ATR multiplier range: must be positive and min <= max")

        # --- NEW: ML STOP ADJUSTMENT RANGE VALIDATIONS ---
        if not hasattr(self, 'sl_adjustment_range') or len(self.sl_adjustment_range) != 2:
            _err("sl_adjustment_range must be a tuple/list of 2 elements")
        else:
            sl_min, sl_max = self.sl_adjustment_range
            if not (isinstance(sl_min, (int, float)) and isinstance(sl_max, (int, float))):
                _err("sl_adjustment_range values must be numeric")
            elif not (sl_min > 0 and sl_min <= sl_max):
                _err("Invalid SL adjustment range: must be positive and min <= max")

        if not hasattr(self, 'tp_adjustment_range') or len(self.tp_adjustment_range) != 2:
            _err("tp_adjustment_range must be a tuple/list of 2 elements")
        else:
            tp_min, tp_max = self.tp_adjustment_range
            if not (isinstance(tp_min, (int, float)) and isinstance(tp_max, (int, float))):
                _err("tp_adjustment_range values must be numeric")
            elif not (tp_min > 0 and tp_min <= tp_max):
                _err("Invalid TP adjustment range: must be positive and min <= max")

        # --- Final check ---
        if len(errors) > 0:
            return False

        logger.debug("✅ MLConfig passed all validations")
        return True

    @classmethod
    def from_ini(cls, ini_path: Optional[Path] = None) -> Optional['MLConfig']:
        ini_path = ini_path or Path.cwd() / "strategy_config.ini"
        if not ini_path.exists():
            logger.debug(f"INI config not found at: {ini_path}")
            return None

        config = configparser.ConfigParser()
        try:
            config.read(ini_path, encoding='utf-8')
            if 'machine_learning' not in config:
                logger.warning("INI has no [machine_learning] section")
                return None

            ml = config['machine_learning']
            bounds = config['ml_boundaries'] if 'ml_boundaries' in config else {}

            def get(key: str, default: Any, type_cast):
                try:
                    if key in ml:
                        val = ml[key]
                        if type_cast is bool:
                            return val.lower() in ('true', '1', 'yes', 'on')
                        return type_cast(val)
                except Exception as e:
                    logger.debug(f"Failed parsing {key} in [machine_learning]: {e}")
                return default

            def get_bound(key: str, default: float) -> float:
                try:
                    if key in bounds:
                        return float(bounds[key])
                except:
                    pass
                return default

            # Adjustment Ranges and Boundaries
            def get_range(key_min: str, key_max: str, default_tuple: Tuple[float, float]) -> Tuple[float, float]:
                try:
                    return (float(bounds.get(key_min, default_tuple[0])), 
                            float(bounds.get(key_max, default_tuple[1])))
                except:
                    return default_tuple

            instance = cls(
                enabled=get('ENABLED', True, bool),
                use_ml_predictions=get('USE_ML_PREDICTIONS', True, bool),
                model_type=get('MODEL_TYPE', 'random_forest', str),
                min_samples_to_train=get('MIN_TRAINING_SAMPLES', 100, int),
                min_model_accuracy=get('MIN_MODEL_ACCURACY', 0.55, float),
                retrain_interval_hours=get('RETRAIN_INTERVAL_HOURS', 24, int),
                confidence_threshold=get('CONFIDENCE_THRESHOLD', 0.6, float),
                min_prediction_confidence=get('MIN_PREDICTION_CONFIDENCE', 0.6, float), 
                feature_window=get('FEATURE_WINDOW', 50, int),
                use_technical_indicators=get('USE_TECHNICAL_INDICATORS', True, bool),
                use_volume_features=get('USE_VOLUME_FEATURES', True, bool),
                use_price_features=get('USE_PRICE_FEATURES', True, bool),
                ml_blend_weight=get('ML_BLEND_WEIGHT', 0.7, float),
                
                # Position Multipliers
                max_position_multiplier=get('MAX_POSITION_MULTIPLIER', 1.5, float),
                min_position_multiplier=get('MIN_POSITION_MULTIPLIER', 0.3, float),
                max_daily_trades=get('MAX_DAILY_TRADES', 15, int),

                # Boundaries[ml_boundaries]
                sl_min_pct=get_bound('SL_MIN_PCT', 0.5),
                sl_max_pct=get_bound('SL_MAX_PCT', 3.0),
                tp_min_pct=get_bound('TP_MIN_PCT', 1.0),
                tp_max_pct=get_bound('TP_MAX_PCT', 5.0),
                min_risk_reward=get_bound('MIN_RISK_REWARD', 1.5),
                max_risk_reward=get_bound('MAX_RISK_REWARD', 5.0),

                # ATR settings
                enable_atr_bounds=get_bound('ENABLE_ATR_BOUNDS', True),
                sl_min_atr_multiplier=get_bound('SL_MIN_ATR_MULTIPLIER', 0.5),
                sl_max_atr_multiplier=get_bound('SL_MAX_ATR_MULTIPLIER', 2.5),
                tp_min_atr_multiplier=get_bound('TP_MIN_ATR_MULTIPLIER', 1.0),
                tp_max_atr_multiplier=get_bound('TP_MAX_ATR_MULTIPLIER', 4.0),

                # Predictor
                sl_adjustment_range=get_range('SL_ADJUST_MIN', 'SL_ADJUST_MAX', (0.8, 2.0)),
                tp_adjustment_range=get_range('TP_ADJUST_MIN', 'TP_ADJUST_MAX', (1.5, 4.0))
            )

            logger.info(f"✅ Loaded MLConfig from INI: {ini_path}")
            return instance

        except Exception as e:
            logger.error(f"⚠️ Critical failure parsing INI config: {e}")
            return None
        
    @classmethod
    def load(cls, ini_path: Optional[Path] = None) -> 'MLConfig':
        config = cls.from_ini(ini_path)
        if config:
            return config

        try:
            config = cls.from_env()
            return config
        except Exception as e:
            logger.warning(f"⚠️ Failed to load from environment: {e}")

        logger.info("ℹ️ Using default MLConfig")
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def override(self, **kwargs) -> 'MLConfig':
        current = self.to_dict()
        current.update(kwargs)
        return MLConfig(**current)

    def __str__(self) -> str:
        d = self.to_dict()
        return f"MLConfig(enabled={d['enabled']}, model='{d['model_type']}', conf_thresh={d['confidence_threshold']})"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.to_dict().items())))

    def __eq__(self, other) -> bool:
        if not isinstance(other, MLConfig):
            return False
        return self.to_dict() == other.to_dict()

# Global config instance
ml_config: MLConfig = MLConfig.load()

def reload_ml_config(ini_path: Optional[Path] = None) -> MLConfig:
    global ml_config
    new_config = MLConfig.load(ini_path)
    ml_config = new_config
    logger.info("🔄 ML Config reloaded successfully")
    return ml_config