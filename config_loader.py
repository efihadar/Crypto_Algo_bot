# root/config_loader.py
import configparser
import os
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple
from collections import OrderedDict
from loguru import logger

class ConfigLoader:
    """
    High-performance, thread-safe, production-grade config loader for trading bots.
    Supports hot-reload, caching, variants, health checks, and graceful degradation.
    """

    RELOAD_INTERVAL = 300

    def __init__(self, config_path: str = None):
        # Thread safety
        self._lock = threading.RLock()

        # Cache layer (key → (value, timestamp))
        self._cache: Dict[str, Tuple[Any, float]] = OrderedDict()
        self._cache_ttl = 5.0  # seconds — configurable via env if needed
        self._max_cache_size = 500

        env_path = os.getenv("CONFIG_PATH")
        root_path = Path("/app/strategy_config.ini")
        project_root = Path(__file__).parent.parent / "strategy_config.ini"

        candidates = []

        if env_path:
            candidates.append(Path(env_path))

        if config_path:
            candidates.append(Path(config_path))

        candidates.extend([root_path, project_root])

        self.config_path: Path | None = None
        for p in candidates:
            if p.exists():
                self.config_path = p
                logger.success(f"🔧 Found strategy config at: {p}")
                break

        if not self.config_path:
            logger.error("❌ Could NOT find strategy_config.ini! Using default path (/app/strategy_config.ini)")
            self.config_path = root_path

        self.config = configparser.ConfigParser(interpolation=None)  # Disable interpolation for performance
        self.active_variant: str | None = None
        self.last_loaded_ts: float = 0.0
        self._last_modified: float = 0.0

        self._load_config()

    def _snapshot(self) -> dict:
        """Take a full snapshot of all config values (section/key → value)."""
        snap = {}
        for section in self.config.sections():
            for key, value in self.config.items(section):
                snap[f"{section}.{key}"] = value
        return snap

    def _detect_changes(self, before: dict, after: dict):
        """Detect and print all changed INI values."""
        for key, old_val in before.items():
            new_val = after.get(key)
            if new_val is None:
                continue
            if str(old_val).strip() != str(new_val).strip():
                logger.warning(f"🔄 INI changed: {key}: {old_val} → {new_val}")

    def _load_config(self) -> bool:
        """Load configuration from disk — thread-safe and cached-aware."""
        try:
            if not self.config_path.exists():
                logger.error(f"❌ Config file missing at: {self.config_path}")
                return False

            self._last_modified = self.config_path.stat().st_mtime
            before = self._snapshot()

            # Clear cache on reload
            with self._lock:
                self._cache.clear()

            files_read = self.config.read(self.config_path, encoding="utf-8")

            if not files_read:
                logger.error(f"❌ Failed to read config file: {self.config_path}")
                logger.error(f"   File exists: {self.config_path.exists()}")
                logger.error(f"   File readable: {os.access(self.config_path, os.R_OK)}")
                return False

            if not self.config.sections():
                logger.error(f"❌ Config file is empty or invalid: {self.config_path}")
                logger.error(f"   No sections found in INI file!")
                return False

            logger.success(f"✅ Successfully loaded {len(self.config.sections())} sections from config")
            after = self._snapshot()
            self._detect_changes(before, after)

            # Update metadata
            self.active_variant = self.get("bot_core", "STRATEGY_VARIANT", "default", str)
            self.RELOAD_INTERVAL = self.get("bot_core", "RELOAD_INTERVAL_SECONDS", 300, int)
            self.last_loaded_ts = time.time()

            logger.info(f"🎯 Active strategy variant: {self.active_variant}")
            logger.info(
                f"⚙️  Core: interval={self.get_cycle_interval()}s, "
                f"batch={self.get_batch_size()}, "
                f"pos_size={self.get('risk_management', 'POSITION_SIZE_USDT', 1.0, float)} USDT, "
                f"min_strength={self.get('enhanced_strategy', 'MIN_SIGNAL_STRENGTH', 40, float)}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Failed to read config: {e}")
            logger.exception(e)
            return False

    # ------------------------------------------------------------------ #
    # 🔄 Reload Methods — THREAD SAFE
    # ------------------------------------------------------------------ #
    def file_was_modified(self) -> bool:
        """Check if config file was modified or deleted."""
        try:
            if not self.config_path.exists():
                logger.critical(f"🚨 Config file DELETED or MOVED: {self.config_path} — continuing with last known config")
                return False
            current_mtime = self.config_path.stat().st_mtime
            return current_mtime > self._last_modified
        except Exception as e:
            logger.error(f"⚠️ Failed to check file modification: {e}")
            return False

    def should_reload(self) -> bool:
        """Check if reload interval passed OR file was modified — thread-safe."""
        try:
            time_passed = (time.time() - self.last_loaded_ts) > self.RELOAD_INTERVAL
            file_changed = self.file_was_modified()
            if file_changed:
                logger.info("📝 Config file was modified on disk!")
            return time_passed or file_changed
        except Exception as e:
            logger.error(f"⚠️ should_reload failed: {e}")
            return False

    def reload_if_needed(self):
        """Reload config if needed — thread-safe and cache-aware."""
        if self.should_reload():
            logger.info("♻️  Auto-reloading strategy_config.ini ...")
            with self._lock:
                success = self._load_config()
            logger.info("─" * 50)
            if success:
                logger.success("✅ strategy_config.ini reloaded successfully!")
            else:
                logger.error("❌ Failed to reload config!")
            logger.info("─" * 50)

    def force_reload(self):
        """Force reload config — thread-safe."""
        logger.warning("🔄 FORCE RELOAD requested!")
        with self._lock:
            success = self._load_config()
        if success:
            logger.success("✅ Config force-reloaded!")
        else:
            logger.error("❌ Force reload failed!")
        return success

    # ------------------------------------------------------------------ #
    # 🔑 GET WITH TTL CACHE — MAJOR PERFORMANCE UPGRADE
    # ------------------------------------------------------------------ #
    def get(self, section: str, key: str, fallback: Any = None, cast: Union[type, str] = str) -> Any:
        """
        Thread-safe get with TTL cache. Returns cached value if fresh, otherwise reloads.
        Format: "{section}.{key}.{cast}" as cache key.
        """
        cache_key = f"{section}.{key}.{cast.__name__ if isinstance(cast, type) else str(cast)}"

        with self._lock:
            now = time.time()

            # Return from cache if valid
            if cache_key in self._cache:
                value, timestamp = self._cache[cache_key]
                if now - timestamp < self._cache_ttl:
                    return value

            # Compute fresh value
            value = self._get_fresh_value(section, key, fallback, cast)

            # Store in cache
            self._cache[cache_key] = (value, now)

            # Enforce max cache size (LRU-like)
            if len(self._cache) > self._max_cache_size:
                # Remove oldest 20%
                to_remove = max(1, len(self._cache) // 5)
                for _ in range(to_remove):
                    self._cache.pop(next(iter(self._cache)))

            return value

    def _get_fresh_value(self, section: str, key: str, fallback: Any, cast: Union[type, str]) -> Any:
        """Actual logic to fetch value from config — uncached."""
        try:
            if self.active_variant and self.active_variant != "default":
                if self.config.has_section(self.active_variant):
                    if self.config.has_option(self.active_variant, key):
                        return self._convert_value(
                            self.config.get(self.active_variant, key),
                            cast
                        )

            if not self.config.has_section(section):
                logger.debug(f"⚠️ Section [{section}] not found, using fallback for {key}")
                return self._convert_value(fallback, cast)

            raw = self.config.get(section, key, fallback=None)
            if raw is None or str(raw).strip() == "":
                logger.debug(f"⚠️ Key {key} in [{section}] is empty, using fallback")
                return self._convert_value(fallback, cast)

            return self._convert_value(raw, cast)

        except Exception as e:
            logger.warning(f"⚠️ Error reading [{section}] {key}: {e}")
            return self._convert_value(fallback, cast)

    # ------------------------------------------------------------------ #
    # 🎯 Type Conversions
    # ------------------------------------------------------------------ #
    def _convert_value(self, value: Any, cast: Union[type, str]):
        if value is None:
            return self._default_for_type(cast)

        try:
            if cast == bool:
                return str(value).lower() in ("1", "true", "yes", "on", "y")
            if cast == int:
                return int(float(value))  # handle "3.0" → 3
            if cast == float:
                return float(value)
            if cast == list:
                return [x.strip() for x in str(value).split(",") if x.strip()]
            if cast == "float_list":
                cleaned = str(value).replace("[", "").replace("]", "")
                parts = [x.strip() for x in cleaned.split(",") if x.strip()]
                return [float(x) for x in parts]
            return str(value)

        except Exception as e:
            logger.warning(f"⚠️ Conversion failed for '{value}' to {cast}: {e}")
            return self._default_for_type(cast)

    def _default_for_type(self, cast: Union[type, str]):
        if cast == bool:
            return False
        if cast == int:
            return 0
        if cast == float:
            return 0.0
        if cast in (list, "float_list"):
            return []
        return ""

    # ------------------------------------------------------------------ #
    # 🧪 HEALTH CHECK — NEW FEATURE
    # ------------------------------------------------------------------ #
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of config system.
        Returns status dict for monitoring systems.
        """
        now = time.time()
        file_exists = self.config_path.exists() if self.config_path else False
        age_seconds = now - self.last_loaded_ts
        stale = age_seconds > (self.RELOAD_INTERVAL * 3)
        cache_size = len(self._cache)

        healthy = file_exists and not stale

        report = {
            "healthy": healthy,
            "file_exists": file_exists,
            "config_age_seconds": age_seconds,
            "stale": stale,
            "last_reload": self.last_loaded_ts,
            "reload_interval": self.RELOAD_INTERVAL,
            "cache_size": cache_size,
            "active_variant": self.active_variant,
            "sections_loaded": len(self.config.sections()) if hasattr(self, 'config') else 0,
        }

        if not healthy:
            logger.critical(f"🆘 Config Health Check FAILED: {report}")

        return report

    # ------------------------------------------------------------------ #
    # 📦 GROUPED PARAMS — UNCHANGED (backward compatible)
    # ------------------------------------------------------------------ #
    def get_bot_core_params(self) -> Dict[str, Any]:
        return {
            "DEBUG_MODE": self.get("debug", "DEBUG_MODE", False, bool),
            "CYCLE_INTERVAL_SECONDS": self.get("bot_core", "CYCLE_INTERVAL_SECONDS", 5, int),
            "BATCH_SIZE": self.get("bot_core", "BATCH_SIZE", 3, int),
            "GENERATE_SIGNALS_IF_NONE": self.get("bot_core", "GENERATE_SIGNALS_IF_NONE", True, bool),
            "ENABLE_SESSIONS": self.get("bot_core", "ENABLE_SESSIONS", True, bool),
            "RELOAD_INTERVAL_SECONDS": self.get("bot_core", "RELOAD_INTERVAL_SECONDS", 300, int),
            "STRATEGY_VARIANT": self.get("bot_core", "STRATEGY_VARIANT", "default", str),
        }

    def get_trend_params(self) -> Dict[str, Any]:
        return {
            "DETECT_RANGING": self.get("trend", "DETECT_RANGING", True, bool),
            "SKIP_RANGING_MARKETS": self.get("trend", "SKIP_RANGING_MARKETS", False, bool),
            "RANGING_ATR_THRESHOLD": self.get("trend", "RANGING_ATR_THRESHOLD", 0.15, float),
            "MIN_TREND_STRENGTH": self.get("trend", "MIN_TREND_STRENGTH", 0.30, float),
        }

    def get_trading_block(self) -> Dict[str, Any]:
        return {
            "SYMBOLS": self.get("trading", "SYMBOLS", [], list),
        }

    def get_enhanced_strategy_params(self) -> Dict[str, Any]:
        return {
            "EMA_FAST": self.get("enhanced_strategy", "EMA_FAST", 9, int),
            "EMA_SLOW": self.get("enhanced_strategy", "EMA_SLOW", 21, int),
            "EMA_LONG": self.get("enhanced_strategy", "EMA_LONG", 50, int),
            "RSI_PERIOD": self.get("enhanced_strategy", "RSI_PERIOD", 14, int),
            "RSI_OVERBOUGHT": self.get("enhanced_strategy", "RSI_OVERBOUGHT", 75, int),
            "RSI_OVERSOLD": self.get("enhanced_strategy", "RSI_OVERSOLD", 25, int),
            "ATR_PERIOD": self.get("enhanced_strategy", "ATR_PERIOD", 14, int),
            "VOLUME_MA_PERIOD": self.get("enhanced_strategy", "VOLUME_MA_PERIOD", 20, int),
            "ENABLE_VOLUME_FILTER": self.get("enhanced_strategy", "ENABLE_VOLUME_FILTER", True, bool),
            "MIN_VOLUME_RATIO": self.get("enhanced_strategy", "MIN_VOLUME_RATIO", 0.15, float),
            "STRICTNESS_MODE": self.get("enhanced_strategy", "STRICTNESS_MODE", "NORMAL", str),
            "STRATEGY_STYLE": self.get("enhanced_strategy", "STRATEGY_STYLE", "HYBRID", str),
            "ENABLE_BREAKOUT_FILTER": self.get("enhanced_strategy", "ENABLE_BREAKOUT_FILTER", True, bool),
            "BREAKOUT_LOOKBACK_BARS": self.get("enhanced_strategy", "BREAKOUT_LOOKBACK_BARS", 20, int),
            "BREAKOUT_MIN_PCT": self.get("enhanced_strategy", "BREAKOUT_MIN_PCT", 0.08, float),
            "TARGET_SIGNALS_PER_DAY": self.get("enhanced_strategy", "TARGET_SIGNALS_PER_DAY", 25, int),
            "MIN_SIGNAL_STRENGTH": self.get("enhanced_strategy", "MIN_SIGNAL_STRENGTH", 40, float),
            "REQUIRE_EMA_CROSS_FOR_ENTRY": self.get("enhanced_strategy", "REQUIRE_EMA_CROSS_FOR_ENTRY", False, bool),
            "ENABLE_WEAK_EMA_ENTRY": self.get("enhanced_strategy", "ENABLE_WEAK_EMA_ENTRY", True, bool),
            "WEAK_EMA_TOLERANCE": self.get("enhanced_strategy", "WEAK_EMA_TOLERANCE", 0.15, float),
            "MIN_CANDLES_REQUIRED": self.get("enhanced_strategy", "MIN_CANDLES_REQUIRED", 100, int),
            "KLINES_LIMIT": self.get("enhanced_strategy", "KLINES_LIMIT", 200, int),
            "KLINES_INTERVAL": self.get("enhanced_strategy", "KLINES_INTERVAL", 15, int),
            "EMA_WEIGHT": self.get("enhanced_strategy", "EMA_WEIGHT", 35, float),
            "RSI_WEIGHT": self.get("enhanced_strategy", "RSI_WEIGHT", 15, float),
            "TREND_WEIGHT": self.get("enhanced_strategy", "TREND_WEIGHT", 20, float),
            "VOLUME_WEIGHT": self.get("enhanced_strategy", "VOLUME_WEIGHT", 15, float),
            "TREND_STRENGTH_WEIGHT": self.get("enhanced_strategy", "TREND_STRENGTH_WEIGHT", 15, float),
            "SL_ATR_MULTIPLIER": self.get("enhanced_strategy", "SL_ATR_MULTIPLIER", 2.0, float),
            "TP_ATR_MULTIPLIER": self.get("enhanced_strategy", "TP_ATR_MULTIPLIER", 2.0, float),
            "BUY_STRICTNESS": self.get("enhanced_strategy", "BUY_STRICTNESS", 0.25, float),
            "SELL_STRICTNESS": self.get("enhanced_strategy", "SELL_STRICTNESS", 0.20, float),
        }

    def get_strategy_filters(self) -> Dict[str, Any]:
        return {
            "ALLOW_BUY_SIGNALS": self.get("strategy_filters", "ALLOW_BUY_SIGNALS", True, bool),
            "ALLOW_SELL_SIGNALS": self.get("strategy_filters", "ALLOW_SELL_SIGNALS", True, bool),
            "ENABLE_RANGE_SCALPING": self.get("strategy_filters", "ENABLE_RANGE_SCALPING", False, bool),
        }

    def get_risk_params(self) -> Dict[str, Any]:
        return {
            "POSITION_SIZE_USDT": self.get("risk_management", "POSITION_SIZE_USDT", 1.0, float),
            "MIN_POSITION_USDT": self.get("risk_management", "MIN_POSITION_USDT", 1.0, float),
            "MAX_POSITION_USDT": self.get("risk_management", "MAX_POSITION_USDT", 20.0, float),
            "MIN_NOTIONAL_USDT": self.get("risk_management", "MIN_NOTIONAL_USDT", 5.0, float),
            "USE_KELLY_CRITERION": self.get("risk_management", "USE_KELLY_CRITERION", False, bool),
            "MAX_KELLY_FRACTION": self.get("risk_management", "MAX_KELLY_FRACTION", 0.10, float),
            "MAX_POSITIONS_PER_SYMBOL": self.get("risk_management", "MAX_POSITIONS_PER_SYMBOL", 1, int),
            "MAX_TOTAL_POSITIONS": self.get("risk_management", "MAX_TOTAL_POSITIONS", 2, int),
            "POSITION_COOLDOWN_MINUTES": self.get("risk_management", "POSITION_COOLDOWN_MINUTES", 1, int),
            "MAX_DAILY_LOSS_PCT": self.get("risk_management", "MAX_DAILY_LOSS_PCT", 5.0, float),
            "MAX_DRAWDOWN_PCT": self.get("risk_management", "MAX_DRAWDOWN_PCT", 10.0, float),
            "RISK_PER_TRADE_PCT": self.get("risk_management", "RISK_PER_TRADE_PCT", 1.0, float),
            "SL_METHOD": self.get("risk_management", "SL_METHOD", "dynamic_atr", str),
            "SL_ATR_MULTIPLIER": self.get("risk_management", "SL_ATR_MULTIPLIER", 1.0, float),
            "SL_PCT_DEFAULT": self.get("risk_management", "SL_PCT_DEFAULT", 1.6, float),
            "TP_METHOD": self.get("risk_management", "TP_METHOD", "ladder", str),
            "TP_LADDER_PCTS": self.get("risk_management", "TP_LADDER_PCTS", "float_list"),
            "TP_RISK_REWARD_RATIO": self.get("risk_management", "TP_RISK_REWARD_RATIO", 2.0, float),
            "TRAILING_STOP_ENABLED": self.get("risk_management", "TRAILING_STOP_ENABLED", False, bool),
            "TRAILING_STOP_ACTIVATION_PCT": self.get("risk_management", "TRAILING_STOP_ACTIVATION_PCT", 1.0, float),
            "TRAILING_STOP_DISTANCE_PCT": self.get("risk_management", "TRAILING_STOP_DISTANCE_PCT", 0.8, float),
        }

    def get_notifications_params(self) -> Dict[str, Any]:
        return {
            "ALERT_ON_SIGNAL": self.get("notifications", "ALERT_ON_SIGNAL", True, bool),
            "ALERT_ON_ENTRY": self.get("notifications", "ALERT_ON_ENTRY", True, bool),
            "ALERT_ON_EXIT": self.get("notifications", "ALERT_ON_EXIT", True, bool),
            "ALERT_ON_ERROR": self.get("notifications", "ALERT_ON_ERROR", True, bool),
            "ALERT_ON_RISK_BREACH": self.get("notifications", "ALERT_ON_RISK_BREACH", True, bool),
        }

    def get_api_params(self) -> Dict[str, Any]:
        return {
            "API_RATE_LIMIT_CALLS_PER_SECOND": self.get("api", "API_RATE_LIMIT_CALLS_PER_SECOND", 5, int),
            "API_TIMEOUT_SECONDS": self.get("api", "API_TIMEOUT_SECONDS", 30, int),
            "ENABLE_API_HEALTH_CHECKS": self.get("api", "ENABLE_API_HEALTH_CHECKS", True, bool),
        }

    def get_ml_config(self) -> Dict[str, Any]:
        ml_section = "machine_learning"
        boundaries_section = "ml_boundaries"
        ml_config = {
            "ML_ENABLED": self.getboolean(ml_section, "ML_ENABLED", True),
            "USE_ML_PREDICTIONS": self.getboolean(ml_section, "USE_ML_PREDICTIONS", True),
            "MIN_TRAINING_SAMPLES": self.getint(ml_section, "MIN_TRAINING_SAMPLES", 100),
            "RETRAIN_INTERVAL_HOURS": self.getint(ml_section, "RETRAIN_INTERVAL_HOURS", 24),
            "ML_BLEND_WEIGHT": self.getfloat(ml_section, "ML_BLEND_WEIGHT", 0.7),
            "MODEL_TYPE": self.get(ml_section, "MODEL_TYPE", "random_forest"),
            "MIN_MODEL_ACCURACY": self.getfloat(ml_section, "MIN_MODEL_ACCURACY", 0.55),
            "CONFIDENCE_THRESHOLD": self.getfloat(ml_section, "CONFIDENCE_THRESHOLD", 0.6),
            "FEATURE_WINDOW": self.getint(ml_section, "FEATURE_WINDOW", 50),
            "USE_TECHNICAL_INDICATORS": self.getboolean(ml_section, "USE_TECHNICAL_INDICATORS", True),
            "USE_VOLUME_FEATURES": self.getboolean(ml_section, "USE_VOLUME_FEATURES", True),
            "USE_PRICE_FEATURES": self.getboolean(ml_section, "USE_PRICE_FEATURES", True),
            "SL_MIN_PCT": self.getfloat(boundaries_section, "SL_MIN_PCT", 0.5),
            "SL_MAX_PCT": self.getfloat(boundaries_section, "SL_MAX_PCT", 3.0),
            "TP_MIN_PCT": self.getfloat(boundaries_section, "TP_MIN_PCT", 1.0),
            "TP_MAX_PCT": self.getfloat(boundaries_section, "TP_MAX_PCT", 5.0),
            "MIN_RISK_REWARD": self.getfloat(boundaries_section, "MIN_RISK_REWARD", 1.5),
            "MAX_RISK_REWARD": self.getfloat(boundaries_section, "MAX_RISK_REWARD", 5.0),
            "ENABLE_ATR_BOUNDS": self.getboolean(boundaries_section, "ENABLE_ATR_BOUNDS", True),
            "SL_MIN_ATR_MULTIPLIER": self.getfloat(boundaries_section, "SL_MIN_ATR_MULTIPLIER", 0.5),
            "SL_MAX_ATR_MULTIPLIER": self.getfloat(boundaries_section, "SL_MAX_ATR_MULTIPLIER", 2.5),
            "TP_MIN_ATR_MULTIPLIER": self.getfloat(boundaries_section, "TP_MIN_ATR_MULTIPLIER", 1.0),
            "TP_MAX_ATR_MULTIPLIER": self.getfloat(boundaries_section, "TP_MAX_ATR_MULTIPLIER", 4.0),
        }
        logger.debug(f"🧠 Loaded ML config: {ml_config}")
        return ml_config

    # ------------------------------------------------------------------ #
    # 🎯 Convenience Getters
    # ------------------------------------------------------------------ #
    def get_symbols(self) -> List[str]:
        return self.get("trading", "SYMBOLS", [], list)

    def get_cycle_interval(self) -> int:
        return self.get("bot_core", "CYCLE_INTERVAL_SECONDS", 5, int)

    def get_batch_size(self) -> int:
        return self.get("bot_core", "BATCH_SIZE", 3, int)

    def get_min_signal_strength(self) -> float:
        return self.get("enhanced_strategy", "MIN_SIGNAL_STRENGTH", 40.0, float)

    def get_min_notional(self) -> float:
        return self.get("risk_management", "MIN_NOTIONAL_USDT", 5.0, float)

    # ------------------------------------------------------------------ #
    # 🧩 Legacy Compatibility — NO CHANGES
    # ------------------------------------------------------------------ #
    def get_trading_params(self) -> Dict[str, Any]:
        params = {
            "DEBUG_MODE": self.get("debug", "DEBUG_MODE", False, bool),
            "CYCLE_INTERVAL_SECONDS": self.get("bot_core", "CYCLE_INTERVAL_SECONDS", 5, int),
            "BATCH_SIZE": self.get("bot_core", "BATCH_SIZE", 5, int),
            "GENERATE_SIGNALS_IF_NONE": self.get("bot_core", "GENERATE_SIGNALS_IF_NONE", True, bool),
            "HTF_TIMEFRAME": self.get("timeframes", "HTF_TIMEFRAME", "4h", str),
            "ENTRY_TIMEFRAME": self.get("timeframes", "ENTRY_TIMEFRAME", "1h", str),
            "ENTRY_LOOKBACK": self.get("timeframes", "ENTRY_LOOKBACK", 200, int),
            "EMA_FAST": self.get("indicators", "EMA_FAST", 9, int),
            "EMA_SLOW": self.get("indicators", "EMA_SLOW", 21, int),
            "RSI_PERIOD": self.get("indicators", "RSI_PERIOD", 14, int),
            "RSI_OVERBOUGHT": self.get("indicators", "RSI_OVERBOUGHT", 70, int),
            "RSI_OVERSOLD": self.get("indicators", "RSI_OVERSOLD", 30, int),
            "MIN_VOLUME_USDT": self.get("indicators", "MIN_VOLUME_USDT", 10000.0, float),
            "POSITION_SIZE_USDT": self.get("risk_management", "POSITION_SIZE_USDT", 1.0, float),
            "SL_PCT_DEFAULT": self.get("risk_management", "SL_PCT_DEFAULT", 1.0, float),
            "TP_LADDER_PCTS": self.get("risk_management", "TP_LADDER_PCTS", "float_list"),
            "MAX_POSITIONS_PER_SYMBOL": self.get("risk_management", "MAX_POSITIONS_PER_SYMBOL", 1, int),
            "MAX_TOTAL_POSITIONS": self.get("risk_management", "MAX_TOTAL_POSITIONS", 2, int),
            "LOOKBACK_BARS": self.get("data", "LOOKBACK_BARS", 200, int),
        }
        return params

    def getboolean(self, section: str, key: str, fallback: bool = False) -> bool:
        return self.get(section, key, fallback, bool)

    def getint(self, section: str, key: str, fallback: int = 0) -> int:
        return self.get(section, key, fallback, int)

    def getfloat(self, section: str, key: str, fallback: float = 0.0) -> float:
        return self.get(section, key, fallback, float)

# ---------------------- SINGLETON HELPERS — THREAD SAFE ---------------------- #

_config_loader_instance: Optional[ConfigLoader] = None
_singleton_lock = threading.RLock()

def get_config_loader() -> ConfigLoader:
    global _config_loader_instance
    with _singleton_lock:
        if _config_loader_instance is None:
            _config_loader_instance = ConfigLoader()
        else:
            _config_loader_instance.reload_if_needed()
        return _config_loader_instance

def force_reload_config():
    global _config_loader_instance
    with _singleton_lock:
        if _config_loader_instance:
            return _config_loader_instance.force_reload()
    return False

def get_trading_params() -> Dict[str, Any]:
    return get_config_loader().get_trading_params()

def get_position_size() -> float:
    return get_config_loader().get("risk_management", "POSITION_SIZE_USDT", 1.0, float)

def get_sl_pct() -> float:
    return get_config_loader().get("risk_management", "SL_PCT_DEFAULT", 1.0, float)

def get_tp_ladder() -> List[float]:
    return get_config_loader().get("risk_management", "TP_LADDER_PCTS", "float_list")

def get_batch_size() -> int:
    return get_config_loader().get("bot_core", "BATCH_SIZE", 5, int)

def get_generate_signals_if_none() -> bool:
    return get_config_loader().get("bot_core", "GENERATE_SIGNALS_IF_NONE", True, bool)

def get_cycle_interval() -> int:
    return get_config_loader().get("bot_core", "CYCLE_INTERVAL_SECONDS", 5, int)

def get_active_variant_name() -> str:
    return get_config_loader().active_variant or "default"

def get_timeframe() -> str:
    return get_config_loader().get("timeframes", "ENTRY_TIMEFRAME", "1h", str)

def get_lookback_bars() -> int:
    return get_config_loader().get("data", "LOOKBACK_BARS", 200, int)