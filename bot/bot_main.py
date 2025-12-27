# bot_main.py
"""
Professional Crypto Trading Bot
Main entry point for multi-account trading system with ML integration.
Production-ready with comprehensive error handling and monitoring.
"""
import os
import sys
import time
import signal
import copy
import traceback
import inspect
import requests
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from loguru import logger
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from datetime import datetime, timezone
from heartbeat_manager import HeartbeatManager
from db_utils import record_metrics, record_execution, get_open_trades, insert_signal
from smart_safety import SmartSafetyManager
from sessions import BybitSession
from get_symbols_list import run_update_process
from multi_account import get_multi_account_manager, MultiAccountManager
from correlation_manager import CorrelationManager
from time_stop_manager import TimeStopManager
from regime_detector import RegimeDetector, MarketRegime
from datetime import datetime, timezone
from risk_manager import RiskManager

# Global session for ML API
ml_session = requests.Session()
ml_session.timeout = 5.0

# Global multi-account manager
multi_account: Optional[MultiAccountManager] = None

# ============================================================
# 📁 PATH CONFIGURATION - ENHANCED
# ============================================================
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

ML_SYSTEM_PATH = os.path.join(project_root, "ml_system")
if ML_SYSTEM_PATH not in sys.path:
    sys.path.insert(0, ML_SYSTEM_PATH)
    logger.debug(f"🔧 Added to PYTHONPATH: {ML_SYSTEM_PATH}")

# ============================================================
# 🎨 CONFIGURE LOGGER - PROFESSIONAL MODE
# ============================================================
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
    colorize=True
)

# Add file logging
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
logger.add(
    log_dir / "trading_bot_{time}.log",
    rotation="100 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

# ============================================================
# 📦 IMPORTS WITH COMPREHENSIVE ERROR HANDLING
# ============================================================
try:
    from config_loader import get_config_loader, ConfigLoader

    # Load config loader
    loader = get_config_loader()
    logger.info("📄 Config loaded successfully")

    # Load ML configuration from INI (MUST happen before ML imports)
    ml_config_values = loader.get_ml_config()
    
    ML_ENABLED = ml_config_values["ML_ENABLED"]
    USE_ML_PREDICTIONS = ml_config_values["USE_ML_PREDICTIONS"]
    ML_BLEND_WEIGHT = ml_config_values["ML_BLEND_WEIGHT"]
    CONFIDENCE_THRESHOLD = ml_config_values["CONFIDENCE_THRESHOLD"]
    MIN_TRAINING_SAMPLES = ml_config_values["MIN_TRAINING_SAMPLES"]
    MODEL_TYPE = ml_config_values["MODEL_TYPE"]
    FEATURE_WINDOW = ml_config_values["FEATURE_WINDOW"]

    logger.info(f"🧠 ML Config loaded from INI: Enabled={ML_ENABLED}, "
               f"Predictions={USE_ML_PREDICTIONS}, BlendWeight={ML_BLEND_WEIGHT}, "
               f"ConfidenceThreshold={CONFIDENCE_THRESHOLD}")

    # Strategy imports
    try:
        from trade_strategy import EnhancedBybitStrategy as BybitStrategy
        ENHANCED_STRATEGY = True
        logger.success("✅ Enhanced strategy loaded")
    except ImportError:
        from trade_strategy import BybitStrategy
        ENHANCED_STRATEGY = False
        logger.warning("⚠️ Using basic strategy (EnhancedBybitStrategy not available)")

    # Risk Manager
    try:
        from risk_manager import EnhancedRiskManager as RiskManager
        ENHANCED_RISK = True
        logger.success("✅ Enhanced risk manager loaded")
    except ImportError:
        from risk_manager import RiskManager
        ENHANCED_RISK = False
        logger.warning("⚠️ Using basic risk manager (EnhancedRiskManager not available)")

    # Smart Safety
    from smart_safety import SmartSafetyManager

    # Order Manager
    from order_manager import OrderManager

    # Emergency Exit Manager
    try:
        from emergency_exit_manager import EmergencyExitManager
        EMERGENCY_EXIT_AVAILABLE = True
        logger.success("✅ EmergencyExitManager loaded")
    except ImportError:
        logger.warning("⚠️ EmergencyExitManager not found - emergency protections disabled")
        EmergencyExitManager = None
        EMERGENCY_EXIT_AVAILABLE = False

    # Telegram
    try:
        from telegram.telegram_reporter import TelegramReporter
        TELEGRAM_AVAILABLE = True
        logger.success("✅ Telegram reporter loaded")
    except ImportError:
        logger.warning("⚠️ Telegram reporter not found")
        TelegramReporter = None
        TELEGRAM_AVAILABLE = False

    # Database
    try:
        from order_manager import record_execution
        logger.success("✅ record_execution function imported successfully")
    except ImportError as e:
        logger.error(f"❌ Could not import record_execution: {e}")

    # Ensure DB_AVAILABLE is properly set
    try:
        from order_manager import DB_INTEGRATION
        DB_AVAILABLE = DB_INTEGRATION
        logger.info(f"✅ Database availability fixed: DB_AVAILABLE = {DB_AVAILABLE}")
    except Exception as e:
        logger.error(f"❌ Could not sync database availability: {e}")

    try:
        from correlation_manager import CorrelationManager
        CORRELATION_AVAILABLE = True
        logger.success("✅ CorrelationManager loaded")
    except ImportError:
        logger.warning("⚠️ CorrelationManager not available")
        CORRELATION_AVAILABLE = False
        CorrelationManager = None

    try:
        from time_stop_manager import TimeStopManager
        TIME_STOPS_AVAILABLE = True
        logger.success("✅ TimeStopManager loaded")
    except ImportError:
        logger.warning("⚠️ TimeStopManager not available")
        TIME_STOPS_AVAILABLE = False
        TimeStopManager = None

    try:
        from regime_detector import RegimeDetector, MarketRegime
        REGIME_AVAILABLE = True
        logger.success("✅ RegimeDetector loaded")
    except ImportError:
        logger.warning("⚠️ RegimeDetector not available")
        REGIME_AVAILABLE = False
        RegimeDetector = None
        MarketRegime = None

    try:
        from performance_attribution import PerformanceAttribution
        ATTRIBUTION_AVAILABLE = True
        logger.success("✅ PerformanceAttribution loaded")
    except ImportError:
        logger.warning("⚠️ PerformanceAttribution not available")
        ATTRIBUTION_AVAILABLE = False
        PerformanceAttribution = None

    try:
        from walk_forward_tester import WalkForwardTester
        WALK_FORWARD_AVAILABLE = True
        logger.success("✅ WalkForwardTester loaded")
    except ImportError:
        logger.warning("⚠️ WalkForwardTester not available")
        WALK_FORWARD_AVAILABLE = False
        WalkForwardTester = None
    
    # ML System - MUST come after ML config is loaded
    ML_AVAILABLE = False
    MLManager = None
    ml_config = None
    
    try:
        # Method 1: Try direct import first
        try:
            from ml_system import MLManager, ml_config
            ML_AVAILABLE = True
            logger.success("✅ ML System loaded (direct import)")
        except ImportError:
            # Method 2: Fallback to order_manager's loader
            logger.debug("🔁 Falling back to order_manager loader...")
            from order_manager import _load_ml_manager
            ml_manager_instance = _load_ml_manager()
            if ml_manager_instance is not None and getattr(ml_manager_instance, 'enabled', False):
                MLManager = type(ml_manager_instance)  # Get the class
                ml_config = getattr(ml_manager_instance, 'config', None)
                ML_AVAILABLE = True
                logger.success("✅ ML System loaded via order_manager fallback")
            else:
                raise ImportError("ML Manager disabled or not initialized")

        # Test instantiation only if enabled by INI config
        if ML_AVAILABLE and ML_ENABLED:
            test_mgr = MLManager()
            if not getattr(test_mgr, 'enabled', False):
                logger.warning("⚠️ ML System loaded but disabled in config")
            else:
                logger.info("🤖 ML System status: ACTIVE and ENABLED (via INI config)")
        elif ML_AVAILABLE and not ML_ENABLED:
            logger.info("ℹ️ ML System loaded but DISABLED by INI config")
            
    except Exception as e:
        logger.warning(f"⚠️ ML System not available: {e}")
        ML_AVAILABLE = False
        MLManager = None
        ml_config = None

except Exception as e:
    logger.error(f"❌ Critical import error: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# ============================================================
# ⚙️ GLOBAL CONFIGURATION - ENHANCED
# ============================================================
ENABLE_LIVE_TRADING = os.getenv("ENABLE_LIVE_TRADING", "false").lower() in ("1", "true", "yes", "y")
BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "false").lower() in ("1", "true", "yes", "y")

# --- Symbol update configuration ---
LAST_SYMBOLS_UPDATE = 0
UPDATE_INTERVAL = loader.get("bot_core", "SYMBOLS_UPDATE_INTERVAL", 43200, int)  # 12 hours

# --- Failure management ---
MAX_FAILURES = 3
MAX_CONSECUTIVE_ERRORS = 5

# --- ML Configuration (already loaded from INI above) ---
ML_API_URL = os.getenv("ML_API_URL", "http://localhost:8000")  # Only used if you have external API

CIRCUIT_BREAKER_ACTIVE = False
CIRCUIT_BREAKER_UNTIL = 0

# ⚠️ SAFETY CHECK - ENHANCED
if not BYBIT_TESTNET and not ENABLE_LIVE_TRADING:
    logger.error("=" * 70)
    logger.error("🚨 CRITICAL WARNING: DANGEROUS CONFIGURATION!")
    logger.error("=" * 70)
    logger.error("You are connected to LIVE API (BYBIT_TESTNET=false)")
    logger.error("But live trading is DISABLED (ENABLE_LIVE_TRADING=false)")
    logger.error("")
    logger.error("This means:")
    logger.error("  ❌ Bot will analyze LIVE market")
    logger.error("  ❌ Bot will NOT execute trades")
    logger.error("  ❌ Manual positions may be left unmanaged!")
    logger.error("")
    logger.error("Recommended settings:")
    logger.error("  1. For testing: BYBIT_TESTNET=true, ENABLE_LIVE_TRADING=false")
    logger.error("  2. For live: BYBIT_TESTNET=false, ENABLE_LIVE_TRADING=true")
    logger.error("=" * 70)

    for i in range(15, 0, -1):  # Extended to 15 seconds
        logger.warning(f"Continuing in {i} seconds... (Ctrl+C to abort)")
        time.sleep(1)

logger.info("🔧 Configuration Summary:")
logger.info(f"   Live Trading: {ENABLE_LIVE_TRADING}")
logger.info(f"   Testnet Mode: {BYBIT_TESTNET}")
logger.info(f"   Database: {DB_AVAILABLE}")
logger.info(f"   Emergency Exit: {EMERGENCY_EXIT_AVAILABLE}")
logger.info(f"   ML System: {ML_AVAILABLE} (Enabled: {ML_ENABLED})")
logger.info(f"   ML API URL: {ML_API_URL}")

# ============================================================
# 🌐 GLOBAL STATE - ENHANCED
# ============================================================
running = True
cycle_count = 0
consecutive_failures = 0
consecutive_errors = 0

heartbeat: Optional[HeartbeatManager] = None
last_heartbeat = 0

reporter: Optional["TelegramReporter"] = None
strategy: Optional["BybitStrategy"] = None
risk: Optional["RiskManager"] = None
order_manager: Optional["OrderManager"] = None
session: Optional["BybitSession"] = None
smart_safety: Optional["SmartSafetyManager"] = None
emergency_exit: Optional["EmergencyExitManager"] = None
ml_manager: Optional["MLManager"] = None
correlation_manager: Optional["CorrelationManager"] = None
time_stop_manager: Optional["TimeStopManager"] = None
regime_detector: Optional["RegimeDetector"] = None
attribution: Optional["PerformanceAttribution"] = None
walk_forward_tester: Optional["WalkForwardTester"] = None

# Performance tracking
performance_stats = {
    'total_trades': 0,
    'successful_trades': 0,
    'failed_trades': 0,
    'total_pnl': 0.0,
    'start_time': time.time(),
    'last_trade_time': None
}

def get_ml_prediction(signal: Dict, symbol: str) -> Optional[Dict]:
    """Get ML prediction with comprehensive error handling"""
    if not ML_ENABLED or not ML_AVAILABLE:
        return None
    
    try:
        response = ml_session.post(
            f"{ML_API_URL}/predict",
            json={
                "symbol": symbol,
                "signal": signal,
                "entry_price": signal.get('price', signal.get('entry_price', 0))
            },
            timeout=3.0  # Increased timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success', False):
                logger.debug(f"🤖 ML prediction received for {symbol}")
                return result
            else:
                logger.warning(f"⚠️ ML prediction failed for {symbol}: {result.get('reason', 'Unknown')}")
        else:
            logger.warning(f"⚠️ ML API returned status {response.status_code} for {symbol}")
            
    except requests.exceptions.Timeout:
        logger.warning(f"⚠️ ML API timeout for {symbol}")
    except requests.exceptions.ConnectionError:
        logger.warning(f"⚠️ ML API connection error for {symbol}")
    except Exception as e:
        logger.error(f"❌ ML API error for {symbol}: {e}")
    
    return None

def send_trade_outcome_to_ml(trade_data: Dict):
    """Send completed trade data to ML system for learning"""
    if not ML_ENABLED or not ML_AVAILABLE:
        return
    
    try:
        response = requests.post(
            f"{ML_API_URL}/training/record",
            json=trade_data,
            timeout=5.0
        )
        if response.status_code == 200:
            logger.debug("📊 Trade outcome sent to ML system")
        else:
            logger.debug(f"⚠️ ML training endpoint returned {response.status_code}")
    except Exception as e:
        logger.debug(f"⚠️ Failed to send outcome to ML: {e}")

# ============================================================
# 🛡️ SIGNAL HANDLERS - ENHANCED
# ============================================================
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global running
    logger.warning(f"🛑 Received signal {signum}, initiating graceful shutdown...")
    running = False

    # Close all positions before shutdown
    if order_manager and ENABLE_LIVE_TRADING:
        try:
            logger.warning("🚨 Closing all positions before shutdown...")
            closed_count = order_manager.close_all_positions(reason="Bot shutdown (signal)")
            logger.success(f"✅ Closed {closed_count} positions during shutdown")
        except Exception as e:
            logger.error(f"❌ Failed to close positions: {e}")

    # Send shutdown notification
    safe_send(
        f"🛑 <b>Bot Shutdown Initiated</b>\n"
        f"Signal: {signum}\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        parse_mode="HTML"
    )

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================================
# 📱 TELEGRAM HELPER - ENHANCED
# ============================================================
def safe_send(msg: str, parse_mode: Optional[str] = None, max_retries: int = 3) -> bool:
    """
    Send message to Telegram safely with retry logic
    
    Returns:
        bool: True if message sent successfully
    """
    global reporter
    if not reporter or not TELEGRAM_AVAILABLE:
        return False
    
    for attempt in range(max_retries):
        try:
            if parse_mode:
                reporter.send_message(msg, parse_mode=parse_mode)
            else:
                reporter.send_message(msg)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"⚠️ Telegram send failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(1)
            else:
                logger.error(f"❌ Telegram send failed after {max_retries} attempts: {e}")
                return False
    return False

# ============================================================
# 🗄️ DB HEALTH CHECK - ENHANCED
# ============================================================
def check_db_health() -> bool:
    """Check if database connection is healthy"""
    if not DB_AVAILABLE:
        return False
    
    try:
        with get_connection() as conn:
            if not conn:
                return False
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                _ = cur.fetchone()
        return True
    except Exception as e:
        logger.debug(f"DB health check failed: {e}")
        return False

# ============================================================
# 🔧 QUANTITY PRECISION - ENHANCED
# ============================================================
def round_quantity(quantity: float, symbol: str) -> float:
    """Round quantity according to exchange requirements"""
    try:
        if quantity <= 0:
            return 0.0
            
        # Get symbol limits from exchange
        limits = session.get_symbol_limits(symbol)
        
        # Get quantity step (precision) and minimum quantity
        qty_step = limits.get("qty_step", 0.01)
        min_qty = limits.get("min_qty", 0.0)
        
        # Round according to tick size
        qty_decimal = Decimal(str(quantity))
        step_decimal = Decimal(str(qty_step))
        
        # If quantity is close to minimum, round up to ensure we meet minimum
        if min_qty > 0 and quantity < min_qty * 1.5:
            rounded = (qty_decimal / step_decimal).quantize(
                Decimal('1'), rounding=ROUND_UP
            ) * step_decimal
        else:
            # Otherwise round down for safety
            rounded = (qty_decimal / step_decimal).quantize(
                Decimal('1'), rounding=ROUND_DOWN
            ) * step_decimal
                
        final_qty = float(rounded)
        
        # Ensure minimum quantity is met
        if min_qty > 0 and final_qty < min_qty:
            logger.warning(
                f"⚠️ Rounded qty {final_qty} below min_qty {min_qty} for {symbol}, "
                f"adjusting to minimum"
            )
            final_qty = min_qty
        
        # Final validation
        if final_qty <= 0:
            logger.warning(f"⚠️ Final quantity is 0 for {symbol}")
            return 0.0
            
        return final_qty
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to round quantity for {symbol}: {e}")
        # Fallback to simple rounding
        return round(quantity, 2)

# ============================================================
# 🎯 HYBRID POSITION SIZING - ENHANCED
# ============================================================
def calculate_dynamic_position_size(symbol: str, balance: float, equity: float, entry_price: float, signal: Dict[str, Any]) -> float:
    """
    Calculate dynamic position size with comprehensive risk management.
    Fully aligned with EnhancedRiskManager and INI configuration.
    Uses fee-aware net calculations, regime adjustments, and strength-based overrides.
    Risk percentage is primary driver; all other factors only increase size (never dilute risk).
    """
    try:
        # Early validation
        if not symbol or not isinstance(signal, dict):
            logger.error(f"❌ Invalid input for {symbol}: missing symbol or invalid signal")
            return 0.0

        cfg = loader
        section = "risk_management"

        # Load ALL values from INI to ensure full alignment
        base_usdt = max(0.0, cfg.get(section, "POSITION_SIZE_USDT", 35.0, float))
        min_usdt = max(0.0, cfg.get(section, "MIN_POSITION_USDT", 20.0, float))
        max_usdt = max(min_usdt, cfg.get(section, "MAX_POSITION_USDT", 50.0, float))
        risk_pct = max(0.0, cfg.get(section, "RISK_PER_TRADE_PCT", 2.0, float)) / 100.0
        max_balance_fraction = max(0.0, min(1.0, cfg.get(section, "MAX_BALANCE_FRACTION", 0.85, float)))
        min_notional_usdt = max(0.0, cfg.get(section, "MIN_NOTIONAL_USDT", 6.0, float))

        # Extract side
        side = (signal.get("side") or "").upper()
        if side not in ("BUY", "SELL"):
            logger.warning(f"⚠️ Invalid side '{side}' for {symbol} - blocking trade")
            return 0.0

        # Use EQUITY for conservative risk calculation (NOT max(balance, equity))
        effective_balance = equity
        if effective_balance <= 0:
            logger.warning(f"⚠️ Insufficient equity ({effective_balance:.2f}) for {symbol}")
            return 0.0

        # Check minimum balance requirement
        min_balance_needed = min_usdt / max_balance_fraction if max_balance_fraction > 0 else min_usdt
        if effective_balance < min_balance_needed:
            logger.warning(
                f"⚠️ Equity too low for safe trade on {symbol} "
                f"(equity={effective_balance:.2f}, min_needed={min_balance_needed:.2f})"
            )
            return 0.0

        # ------------------------------------------------------
        # 1) PRIMARY DRIVER: Risk Percentage (INI: RISK_PER_TRADE_PCT)
        # ------------------------------------------------------
        target_usdt = effective_balance * risk_pct if risk_pct > 0 else base_usdt

        # ------------------------------------------------------
        # 2) Apply Adaptive Position Size from RiskManager (if available)
        # ------------------------------------------------------
        if 'risk' in globals() and risk and hasattr(risk, "get_adaptive_position_size"):
            try:
                adaptive = risk.get_adaptive_position_size(symbol, signal)  # ← Now uses your EnhancedRiskManager
                if isinstance(adaptive, (int, float)) and adaptive > 0:
                    # Only allow adaptive size to INCREASE the position (never reduce below risk_pct baseline)
                    target_usdt = max(target_usdt, float(adaptive))
                    logger.debug(f"🔍 Adaptive size applied for {symbol}: ${adaptive:.2f} → new target: ${target_usdt:.2f}")
            except Exception as e:
                logger.debug(f"Adaptive sizing failed for {symbol}: {e}")

        # ------------------------------------------------------
        # 3) Apply Signal Strength Multiplier (ONLY increases size)
        # ------------------------------------------------------
        try:
            strength = float(signal.get("strength", 0) or 0.0)
        except (ValueError, TypeError):
            strength = 0.0

        if strength > 0:
            # Normalize strength to 0-1 range, then scale multiplier from 1.0 to 1.5 (only upward adjustment)
            norm = max(0.0, min(strength, 100.0)) / 100.0
            factor = 1.0 + (norm * 0.5)  # Range: 1.0 to 1.5
            original_target = target_usdt
            target_usdt *= factor

            logger.debug(
                f"💡 Signal strength boost for {symbol}: strength={strength:.1f}, factor={factor:.2f} "
                f"→ {original_target:.2f} → {target_usdt:.2f}"
            )

        # ------------------------------------------------------
        # 4) Apply Regime-Based Adjustments (if available)
        # ------------------------------------------------------
        if 'regime_detector' in globals() and regime_detector and hasattr(regime_detector, 'detect_regime'):
            try:
                regime = regime_detector.detect_regime(symbol)
                adjustments = regime_detector.get_regime_adjustments(regime) or {}
                size_mult = float(adjustments.get("position_size_mult", 1.0))
                
                if size_mult > 1.0:  # Only allow regime to INCREASE size (not decrease)
                    original_target = target_usdt
                    target_usdt *= size_mult
                    
                    logger.info(
                        f"📈 Regime boost for {symbol}: {regime} → size x{size_mult:.2f} "
                        f"({original_target:.2f} → {target_usdt:.2f})"
                    )
                    
                # Skip trade if regime says to avoid
                if adjustments.get("skip_trade", False):
                    logger.warning(f"⏭️ {symbol} skipped due to unfavorable regime: {regime}")
                    return 0.0
                    
            except Exception as e:
                logger.debug(f"Regime adjustment failed for {symbol}: {e}")

        # ------------------------------------------------------
        # 5) Limit by Maximum Balance Fraction
        # ------------------------------------------------------
        max_by_balance = effective_balance * max_balance_fraction
        if target_usdt > max_by_balance:
            logger.debug(f"📉 Capping position by balance limit: {target_usdt:.2f} → {max_by_balance:.2f}")
            target_usdt = max_by_balance

        # ------------------------------------------------------
        # 6) Enforce Configuration Limits
        # ------------------------------------------------------
        target_usdt = max(min_usdt, min(target_usdt, max_usdt))

        # ------------------------------------------------------
        # 7) Enforce Exchange Minimum (Bybit typically 5 USDT)
        # ------------------------------------------------------
        EXCHANGE_MIN = 5.0
        if target_usdt < EXCHANGE_MIN:
            logger.debug(f"⚠️ Target ${target_usdt:.2f} below exchange minimum ${EXCHANGE_MIN:.2f} for {symbol}")

            if effective_balance < EXCHANGE_MIN:
                logger.warning(f"⚠️ Equity ${effective_balance:.2f} too low for minimum trade size ${EXCHANGE_MIN:.2f}")
                return 0.0

            target_usdt = EXCHANGE_MIN
            logger.debug(f"✅ Set to exchange minimum: ${target_usdt:.2f}")

        # ------------------------------------------------------
        # 8) Auto-Boost to Minimum Notional if Needed
        # ------------------------------------------------------
        if target_usdt < min_notional_usdt:
            before = target_usdt
            target_usdt = min_notional_usdt
            logger.info(
                f"🟠 Auto-boost: {symbol} size {before:.2f} → {target_usdt:.2f} "
                f"(min_notional={min_notional_usdt})"
            )

        # ------------------------------------------------------
        # 9) Validate SL/TP Distances (FEE-AWARE)
        # ------------------------------------------------------
        stop_loss = signal.get("stop_loss", 0)
        take_profit = signal.get("take_profit", 0)

        if entry_price > 0 and stop_loss > 0 and take_profit > 0:

            # Validate logical consistency
            if side == "BUY":
                if not (stop_loss < entry_price < take_profit):
                    logger.warning(f"❌ Invalid price levels for BUY: SL={stop_loss}, EP={entry_price}, TP={take_profit}")
                    return 0.0
                gross_sl_dist_pct = ((entry_price - stop_loss) / entry_price) * 100
                gross_tp_dist_pct = ((take_profit - entry_price) / entry_price) * 100
            elif side == "SELL":
                if not (take_profit < entry_price < stop_loss):
                    logger.warning(f"❌ Invalid price levels for SELL: TP={take_profit}, EP={entry_price}, SL={stop_loss}")
                    return 0.0
                gross_sl_dist_pct = ((stop_loss - entry_price) / entry_price) * 100
                gross_tp_dist_pct = ((entry_price - take_profit) / entry_price) * 100
            else:
                return 0.0

            # Load fees from INI
            taker_fee_pct = max(0.0, cfg.get("fees", "TAKER_FEE_PCT", 0.06, float))
            total_fees_pct = taker_fee_pct * 2  # Open + Close

            # Calculate NET distances after fees
            net_sl_dist_pct = gross_sl_dist_pct - total_fees_pct
            net_tp_dist_pct = gross_tp_dist_pct - total_fees_pct

            logger.debug(
                f"[{symbol}] Gross SL: {gross_sl_dist_pct:.3f}%, Gross TP: {gross_tp_dist_pct:.3f}% | "
                f"After Fees ({total_fees_pct:.3f}%): Net SL: {net_sl_dist_pct:.3f}%, Net TP: {net_tp_dist_pct:.3f}%"
            )

            # Load configurable minimums from INI
            min_sl_pct = max(0.0, cfg.get("trading", "MIN_SL_PCT", 0.6, float))
            min_net_sl_pct = max(0.0, cfg.get("trading", "MIN_NET_SL_PCT", 0.12, float))
            min_tp_pct = max(0.0, cfg.get("trading", "MIN_TP_PCT", 1.2, float))
            min_net_tp_pct = max(0.0, cfg.get("trading", "MIN_NET_TP_PCT", 0.30, float))
            min_rr = max(0.0, cfg.get("trading", "MIN_RR_RATIO", 1.4, float))

            # Allow tighter stops for stronger signals (Smart Override)
            if strength >= 95:
                min_sl_pct = max(0.1, min_sl_pct * 0.3)
                min_net_sl_pct = max(0.1, min_net_sl_pct * 0.3)
            elif strength >= 90:
                min_sl_pct = max(0.15, min_sl_pct * 0.5)
                min_net_sl_pct = max(0.15, min_net_sl_pct * 0.5)
            elif strength >= 85:
                min_sl_pct = max(0.2, min_sl_pct * 0.7)
                min_net_sl_pct = max(0.2, min_net_sl_pct * 0.7)

            # Validate MINIMUM GROSS distances
            if gross_sl_dist_pct < min_sl_pct:
                logger.warning(f"❌ {symbol}: Gross SL too tight ({gross_sl_dist_pct:.2f}% < {min_sl_pct}%) - blocking trade")
                return 0.0

            if gross_tp_dist_pct < min_tp_pct:
                logger.warning(f"❌ {symbol}: Gross TP too close ({gross_tp_dist_pct:.2f}% < {min_tp_pct}%) - blocking trade")
                return 0.0

            # Validate MINIMUM NET distances (after fees)
            if net_sl_dist_pct < min_net_sl_pct:
                logger.warning(
                    f"❌ {symbol}: Net SL too tight after fees "
                    f"({gross_sl_dist_pct:.2f}% gross → {net_sl_dist_pct:.2f}% net < {min_net_sl_pct}%) - blocking trade"
                )
                return 0.0

            if net_tp_dist_pct < min_net_tp_pct:
                logger.warning(
                    f"❌ {symbol}: Net TP too small after fees "
                    f"({gross_tp_dist_pct:.2f}% gross → {net_tp_dist_pct:.2f}% net < {min_net_tp_pct}%) - blocking trade"
                )
                return 0.0

            # Validate Risk/Reward ratio (using NET values)
            rr_ratio = net_tp_dist_pct / net_sl_dist_pct if net_sl_dist_pct > 0 else 0

            if rr_ratio < min_rr:
                logger.warning(
                    f"❌ {symbol}: Poor NET R:R ({rr_ratio:.2f} < {min_rr}) after fees - blocking trade | "
                    f"Net TP: {net_tp_dist_pct:.2f}%, Net SL: {net_sl_dist_pct:.2f}%"
                )
                return 0.0

        # ------------------------------------------------------
        # 10) Final Sanity Checks
        # ------------------------------------------------------
        max_allowed = effective_balance * max_balance_fraction
        if target_usdt > max_allowed:
            logger.warning(f"⚠️ Size ${target_usdt:.2f} exceeds max allowed ${max_allowed:.2f} for {symbol}")
            return 0.0

        if target_usdt > effective_balance:
            logger.error(f"❌ Position size ${target_usdt:.2f} exceeds equity ${effective_balance:.2f} - blocking trade")
            return 0.0

        if target_usdt <= 0:
            logger.debug(f"ℹ️ Final calculated size is non-positive: {target_usdt:.2f}")
            return 0.0

        final_size = round(target_usdt, 2)
        logger.info(f"✅ Approved position size for {symbol}: ${final_size:.2f}")
        return final_size

    except Exception as e:
        logger.error(f"❌ Position size calculation FAILED for {symbol}: {str(e)}")
        logger.exception("Full traceback:")
        return 0.0

# ============================================================
# 🚀 INITIALIZATION - ENHANCED
# ============================================================
def initialize_bot() -> bool:
    """
    Initialize all bot components for all accounts with comprehensive error handling.
    """
    global session, strategy, risk, order_manager, smart_safety, emergency_exit
    global reporter, heartbeat, multi_account, ml_manager, loader
    global correlation_manager, time_stop_manager, regime_detector
    global attribution, walk_forward_tester

    try:
        logger.info("=" * 70)
        logger.info("🚀 INITIALIZING CRYPTO TRADING BOT (MULTI-ACCOUNT)")
        logger.info("=" * 70)

        # 0. Initialize Heartbeat Manager
        try:
            heartbeat = HeartbeatManager()
            logger.success("💓 HeartbeatManager initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize HeartbeatManager: {e}")
            return False

        # 1. Load configuration
        logger.info("📋 Loading configuration...")
        try:
            config = loader
            if not hasattr(config, 'get'):
                logger.error("❌ Invalid config loader")
                return False
            logger.success("✅ Configuration loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            return False

        # 2. Initialize Telegram (shared across accounts)
        if TelegramReporter:
            logger.info("📱 Initializing Telegram reporter...")
            try:
                reporter = TelegramReporter()
                logger.success("✅ Telegram reporter initialized")

                mode = "TESTNET" if BYBIT_TESTNET else "LIVE"
                trading = "ENABLED" if ENABLE_LIVE_TRADING else "DISABLED"
                safe_send(
                    f"🤖 <b>Trading Bot Starting...</b>\n"
                    f"Mode: {mode}\n"
                    f"Trading: {trading}",
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.warning(f"⚠️ Telegram init failed: {e}")
                reporter = None

        # 3. Initialize Multi-Account Manager
        logger.info("🔗 Loading accounts...")
        try:
            from bot.multi_account import get_multi_account_manager
            multi_account = get_multi_account_manager()
            num_accounts = multi_account.load_accounts()

            if num_accounts == 0:
                logger.error("❌ No valid accounts found!")
                return False

            logger.success(f"✅ Loaded {num_accounts} account(s)")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Multi-Account Manager: {e}")
            return False

        # 4. Initialize each account
        successful_accounts = 0
        for acc_id in multi_account.get_account_ids():
            acc = multi_account.get_account(acc_id)
            acc_name = acc['name']

            logger.info(f"")
            logger.info(f"━━━ Initializing Account: {acc_name} ━━━")

            try:
                # 5. Create session for this account
                try:
                    acc_session = BybitSession(
                        api_key=acc['api_key'],
                        api_secret=acc['api_secret']
                    )
                    # Test connection
                    test_info = acc_session.get_account_info()
                    if not test_info:
                        raise Exception("Failed to connect to exchange")
                    logger.success(f"   ✅ Session created and tested for {acc_name}")
                except Exception as e:
                    logger.error(f"   ❌ Session creation failed for {acc_name}: {e}")
                    continue

                # 6. Load config for this account
                try:
                    from config_loader import ConfigLoader
                    acc_config = ConfigLoader(acc['config_file'])
                    logger.success(f"   ✅ Config loaded: {acc['config_file']}")
                except Exception as e:
                    logger.error(f"   ❌ Config loading failed for {acc['config_file']}: {e}")
                    acc_config = loader  # Fallback to main config

                # 7. Create strategy
                try:
                    try:
                        acc_strategy = BybitStrategy(client=acc_session)
                        logger.success(f"   ✅ Strategy initialized with client")
                    except TypeError:
                        acc_strategy = BybitStrategy()
                        logger.success(f"   ✅ Strategy initialized (without client)")

                    if ENHANCED_STRATEGY:
                        logger.success(f"   ✨ Enhanced strategy features detected")
                except Exception as e:
                    logger.error(f"   ❌ Strategy initialization failed: {e}")
                    continue

                # 8. Create risk manager
                try:
                    # Pass config to RiskManager
                    acc_risk = RiskManager(config=acc_config)
                    logger.success(f"   ✅ Risk Manager initialized")

                    if ENHANCED_RISK and hasattr(acc_risk, "get_adaptive_position_size"):
                        logger.success(f"   ✨ Enhanced risk features detected")

                    # Register balance callback for risk manager
                    if hasattr(acc_risk, "set_balance_callback"):
                        def make_balance_callback(s):
                            def cb():
                                try:
                                    info = s.get_account_info()
                                    return info.get("balance") if info else None
                                except Exception:
                                    return None
                            return cb
                        acc_risk.set_balance_callback(make_balance_callback(acc_session))
                        logger.debug(f"   ✅ Balance callback registered")
                        
                except Exception as e:
                    logger.error(f"   ❌ Risk Manager initialization failed: {e}")
                    continue

                # 9. Create order manager
                try:
                    acc_order_manager = OrderManager(session=acc_session, risk_manager=acc_risk)
                    logger.success(f"   ✅ Order Manager initialized")
                except Exception as e:
                    logger.error(f"   ❌ Order Manager failed: {e}")
                    acc_order_manager = None
                    continue

                # 10. Create smart safety manager
                acc_smart_safety = None
                try:
                    acc_smart_safety = SmartSafetyManager(client=acc_session, strategy=acc_strategy)
                    logger.success(f"   ✅ Smart Safety initialized")
                except Exception as e:
                    logger.warning(f"   ⚠️ Smart Safety failed: {e}")

                # 11. Create emergency exit manager
                acc_emergency = None
                if EMERGENCY_EXIT_AVAILABLE and EmergencyExitManager:
                    try:
                        acc_emergency = EmergencyExitManager(
                            session=acc_session,
                            order_manager=acc_order_manager
                        )
                        logger.success(f"   ✅ Emergency Exit initialized")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Emergency Exit failed: {e}")

                # Store all components in multi_account manager
                try:
                    multi_account.initialize_account_components(
                        account_id=acc_id,
                        session=acc_session,
                        config=acc_config,
                        risk_manager=acc_risk,
                        order_manager=acc_order_manager,
                        strategy=acc_strategy,
                        smart_safety=acc_smart_safety,
                        emergency_exit=acc_emergency,
                    )
                    successful_accounts += 1
                except Exception as e:
                    logger.error(f"   ❌ Failed to store components for {acc_name}: {e}")
                    continue

                # 13. Display account balance
                try:
                    info = acc_session.get_account_info()
                    if info:
                        balance = info.get("balance", 0)
                        equity = info.get("equity", 0)
                        logger.info(f"   💰 Balance: ${balance:.2f}, Equity: ${equity:.2f}")

                        # Get symbols count
                        symbols_count = 0
                        if hasattr(acc_strategy, "symbols"):
                            symbols_count = len(acc_strategy.symbols)
                        elif hasattr(acc_strategy, "trading_symbols"):
                            symbols_count = len(acc_strategy.trading_symbols)
                        
                        logger.info(f"   📊 Monitoring {symbols_count} symbols")
                    else:
                        logger.warning(f"   ⚠️ Could not fetch account info")
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not fetch account info: {e}")

            except Exception as e:
                logger.error(f"❌ Failed to initialize account {acc_name}: {e}")
                logger.error(traceback.format_exc())
                continue

        if successful_accounts == 0:
            logger.error("❌ No accounts were successfully initialized!")
            return False

        # 14. Initialize Database (shared across accounts)
        if DB_AVAILABLE:
            logger.info("🗄️ Initializing database...")
            try:
                db_healthy = False
                for i in range(30):  # Wait up to 30 seconds
                    if check_db_health():
                        db_healthy = True
                        break
                    time.sleep(1)
                
                if not db_healthy:
                    logger.error("❌ Database unreachable after 30s")
                    safe_send(
                        "🚨 <b>Database unreachable</b> - running in degraded mode",
                        parse_mode="HTML",
                    )
                else:
                    init_tables()
                    logger.success("✅ Database initialized")
            except Exception as e:
                logger.warning(f"⚠️ Database initialization failed: {e}")

        # 15. Initialize ML Manager (shared)
        if ML_AVAILABLE and MLManager:
            logger.info("🤖 Initializing ML Manager...")
            try:
                ml_manager = MLManager()

                if ml_manager.enabled:
                    stats = ml_manager.get_model_stats()
                    logger.success("✅ ML Manager initialized")
                    logger.info(f"   Training samples: {stats['training_samples']}")
                    logger.info(f"   Models trained: {stats['models_trained']}")

                    if stats['models_trained']:
                        logger.info(f"   Profitable: {stats.get('profit_trades', 0)}/{stats['training_samples']}")
                else:
                    logger.info("ℹ️ ML features disabled in config")
                    ml_manager = None

            except Exception as e:
                logger.warning(f"⚠️ ML Manager init failed: {e}")
                ml_manager = None
        else:
            ml_manager = None

        # ═══════════════════════════════════════════════════════════════
        # 16. SET GLOBAL VARIABLES FROM PRIMARY ACCOUNT
        # ═══════════════════════════════════════════════════════════════
        logger.info("=" * 70)
        logger.info("🔧 Setting up global components...")
        logger.info("=" * 70)
        
        all_accounts = multi_account.get_all_accounts()
        num_accounts = len(all_accounts)

        if num_accounts == 0:
            logger.error("❌ No accounts available!")
            return False

        primary_account = all_accounts[0]
        
        session = primary_account.get("session")
        strategy = primary_account.get("strategy")
        risk = primary_account.get("risk_manager")
        order_manager = primary_account.get("order_manager")
        smart_safety = primary_account.get("smart_safety")
        emergency_exit = primary_account.get("emergency_exit")
        
        logger.success(f"✅ Primary account: {primary_account.get('name')}")
        
        # Verify critical components
        components_ok = True
        missing_components = []
        
        if session:
            logger.info(f"   ✅ Session initialized")
        else:
            logger.error(f"   ❌ Session NOT initialized!")
            components_ok = False
            missing_components.append("Session")
            
        if strategy:
            logger.info(f"   ✅ Strategy initialized")
        else:
            logger.warning(f"   ⚠️ Strategy not set")
            missing_components.append("Strategy")
            
        if order_manager:
            logger.info(f"   ✅ OrderManager initialized")
        else:
            logger.warning(f"   ⚠️ OrderManager not set")
            missing_components.append("OrderManager")
            
        if risk:
            logger.info(f"   ✅ RiskManager initialized")
        else:
            logger.warning(f"   ⚠️ RiskManager not set")
            missing_components.append("RiskManager")
        
        if not components_ok:
            logger.error(f"❌ Critical components missing: {', '.join(missing_components)}")
            return False

        # ═══════════════════════════════════════════════════════════════
        # 17. INITIALIZE ADVANCED MANAGERS (CORRELATION, TIME, REGIME)
        # ═══════════════════════════════════════════════════════════════
        logger.info("=" * 70)
        logger.info("🔧 Initializing Advanced Trading Managers...")
        logger.info("=" * 70)
        
        advanced_managers_ok = True
        
        # 17.1 Correlation Manager
        if CORRELATION_AVAILABLE and CorrelationManager:
            try:
                correlation_manager = CorrelationManager(session, loader)
                logger.success("✅ CorrelationManager initialized")
                logger.info(f"   Max correlation: {correlation_manager.max_correlation}")
                logger.info(f"   Max correlated positions: {correlation_manager.max_correlated_positions}")
            except Exception as e:
                logger.warning(f"⚠️ CorrelationManager init failed: {e}")
                logger.debug(traceback.format_exc())
                correlation_manager = None
        else:
            logger.info("ℹ️ CorrelationManager not available")
            correlation_manager = None
        
        # 17.2 Time Stop Manager
        if TIME_STOPS_AVAILABLE and TimeStopManager:
            try:
                time_stop_manager = TimeStopManager(loader)
                logger.success("✅ TimeStopManager initialized")
                logger.info(f"   Max hold time: {time_stop_manager.max_hold_hours}h")
                logger.info(f"   Time decay: {time_stop_manager.enable_time_decay}")
                logger.info(f"   Inactivity stop: {time_stop_manager.enable_inactivity_stop}")
            except Exception as e:
                logger.warning(f"⚠️ TimeStopManager init failed: {e}")
                logger.debug(traceback.format_exc())
                time_stop_manager = None
        else:
            logger.info("ℹ️ TimeStopManager not available")
            time_stop_manager = None
        
        # 17.3 Regime Detector
        if REGIME_AVAILABLE and RegimeDetector:
            try:
                regime_detector = RegimeDetector(session, loader)
                logger.success("✅ RegimeDetector initialized")
                logger.info(f"   Trend threshold: {regime_detector.trend_threshold}")
                logger.info(f"   High vol threshold: {regime_detector.high_vol_threshold}")
            except Exception as e:
                logger.warning(f"⚠️ RegimeDetector init failed: {e}")
                logger.debug(traceback.format_exc())
                regime_detector = None
        else:
            logger.info("ℹ️ RegimeDetector not available")
            regime_detector = None
        
        # 17.4 Performance Attribution
        if ATTRIBUTION_AVAILABLE and PerformanceAttribution:
            try:
                attribution = PerformanceAttribution(loader)
                logger.success("✅ PerformanceAttribution initialized")
                logger.info(f"   Reports directory: {attribution.results_dir}")
            except Exception as e:
                logger.warning(f"⚠️ PerformanceAttribution init failed: {e}")
                logger.debug(traceback.format_exc())
                attribution = None
        else:
            logger.info("ℹ️ PerformanceAttribution not available")
            attribution = None
        
        # 17.5 Walk-Forward Tester (optional, for manual testing)
        if WALK_FORWARD_AVAILABLE and WalkForwardTester and strategy:
            try:
                walk_forward_tester = WalkForwardTester(session, strategy, loader)
                logger.success("✅ WalkForwardTester initialized")
                logger.info(f"   Train: {walk_forward_tester.train_days}d, Test: {walk_forward_tester.test_days}d")
            except Exception as e:
                logger.warning(f"⚠️ WalkForwardTester init failed: {e}")
                logger.debug(traceback.format_exc())
                walk_forward_tester = None
        else:
            logger.info("ℹ️ WalkForwardTester not available")
            walk_forward_tester = None
        
        # Summary of advanced managers
        logger.success("📊 Advanced Managers Status:")
        logger.info(f"   Correlation:  {'✅ Active' if correlation_manager else '❌ Inactive'}")
        logger.info(f"   Time Stops:   {'✅ Active' if time_stop_manager else '❌ Inactive'}")
        logger.info(f"   Regime:       {'✅ Active' if regime_detector else '❌ Inactive'}")
        logger.info(f"   Attribution:  {'✅ Active' if attribution else '❌ Inactive'}")
        logger.info(f"   Walk-Forward: {'✅ Active' if walk_forward_tester else '❌ Inactive'}")
        
        # ═══════════════════════════════════════════════════════════════
        # 18. FINAL SUMMARY
        # ═══════════════════════════════════════════════════════════════
        logger.info("=" * 70)
        logger.success("✅ BOT INITIALIZATION COMPLETE")
        logger.info("=" * 70)

        if num_accounts == 1:
            acc = all_accounts[0]
            logger.info(f"📊 Active Account: {acc['name']}")
        else:
            logger.info(f"   📊 Active Accounts: {num_accounts}")

        # Show each account with balance
        for acc in all_accounts:
            acc_name = acc.get('name', 'Unknown')
            config_file = acc.get('config_file', 'config.ini')
            acc_session = acc.get('session')
            
            if acc_session:
                try:
                    info = acc_session.get_account_info()
                    balance = info.get("balance", 0) if info else 0
                    equity = info.get("equity", 0) if info else 0
                    logger.info(f"     • {acc_name}: ${balance:.2f} (Equity: ${equity:.2f}) [{config_file}]")
                except Exception:
                    logger.info(f"     • {acc_name} [{config_file}]")
            else:
                logger.info(f"     • {acc_name} [{config_file}] (no session)")

        # ═══════════════════════════════════════════════════════════════
        # 19. Send startup notification
        # ═══════════════════════════════════════════════════════════════
        startup_msg = (
            f"✅ <b>Bot Initialized Successfully!</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Account: {primary_account.get('name')}\n"
            f"Mode: {'TESTNET' if BYBIT_TESTNET else 'LIVE'}\n"
            f"Trading: {'ENABLED' if ENABLE_LIVE_TRADING else 'DISABLED'}\n"
            f"\n"
            f"📊 <b>Advanced Features:</b>\n"
            f"   {'✅' if correlation_manager else '❌'} Correlation Check\n"
            f"   {'✅' if time_stop_manager else '❌'} Time-Based Stops\n"
            f"   {'✅' if regime_detector else '❌'} Regime Detection\n"
            f"   {'✅' if attribution else '❌'} Performance Attribution\n"
            f"   {'✅' if ml_manager else '❌'} ML System\n"
            f"\n"
            f"🚀 Starting trading cycle..."
        )
        
        safe_send(startup_msg, parse_mode="HTML")

        # 20. Log cycle interval
        try:
            interval = loader.get("bot_core", "CYCLE_INTERVAL_SECONDS", 120, int)
            logger.info(f"⏰ Cycle interval: {interval} seconds")
        except Exception as e:
            logger.warning(f"⚠️ Failed to get cycle interval: {e}")
            interval = 120
            logger.info(f"⏰ Using default cycle interval: {interval} seconds")
        
        logger.info("🚀 Starting main loop...")
        logger.info("=" * 70)

        return True

    except Exception as e:
        logger.error(f"❌ CRITICAL: Bot initialization failed: {e}")
        logger.error(traceback.format_exc())
        safe_send(
            f"🚨 <b>CRITICAL ERROR</b>\n"
            f"Bot initialization failed:\n"
            f"{str(e)[:200]}",
            parse_mode="HTML",
        )
        return False

# ============================================================
# 📊 SIGNAL GENERATION - Standard
# ============================================================
def fetch_signals() -> List[Dict[str, Any]]:
    """
    Fetch trading signals from the strategy with comprehensive validation and error handling.
    
    Filters signals based on:
    - Valid symbol and price
    - Account balance requirements
    - Exchange minimum trade values
    - Risk/Reward ratio validation (including net values after fees)
    
    Returns:
        List of valid, tradable signals
    """
    try:
        # Validate strategy exists and has required method
        if not strategy or not hasattr(strategy, "generate_signals"):
            logger.error("❌ Strategy does not have generate_signals() method")
            return []

        logger.info("🔍 Generating new signals from strategy...")

        # Generate signals with error handling
        try:
            signals = strategy.generate_signals()
            if not isinstance(signals, list):
                logger.error(f"❌ Strategy returned invalid signal format: {type(signals)}")
                return []
        except Exception as e:
            logger.error(f"❌ Strategy.generate_signals() failed: {e}")
            logger.error(traceback.format_exc())
            return []

        if not signals:
            logger.info("⭕ No signals generated by strategy")
            return []

        # Get account balance for filtering with comprehensive error handling
        try:
            balance_info = session.get_account_info()
            
            if not balance_info or not isinstance(balance_info, dict):
                logger.warning("⚠️ get_account_info returned invalid data, using fallback balance")
                balance = 100.0
            else:
                # Use equity if available, otherwise use balance
                balance = float(balance_info.get("equity", 0) or balance_info.get("balance", 0))
                
                if balance <= 0:
                    logger.warning("⚠️ Account balance/equity is zero or negative, using fallback")
                    balance = 100.0
                    
        except Exception as e:
            logger.error(f"❌ Could not fetch account info: {e}")
            balance = 100.0 

        logger.info(f"💰 Using balance: ${balance:.2f} for signal filtering")

        filtered: List[Dict[str, Any]] = []
        cfg = loader

        for sig_raw in signals:
            try:
                # Create deep copy to avoid modifying original
                sig = copy.deepcopy(sig_raw)

                # Validate symbol
                symbol = sig.get("symbol")
                if not symbol or not isinstance(symbol, str):
                    logger.debug("⚠️ Signal without valid symbol → skipping")
                    continue

                # Validate price
                price = sig.get("price") or sig.get("entry_price")
                if price is None:
                    logger.debug(f"⚠️ Signal on {symbol} has no price → skipping")
                    continue

                try:
                    price = float(price)
                    if price <= 0:
                        logger.debug(f"⚠️ Signal on {symbol} has non-positive price={price} → skipping")
                        continue
                except (ValueError, TypeError):
                    logger.debug(f"⚠️ Signal on {symbol} has invalid price={price} → skipping")
                    continue

                # Get symbol limits from exchange
                try:
                    limits = session.get_symbol_limits(symbol)
                    if not isinstance(limits, dict):
                        logger.debug(f"⚠️ Invalid limits for {symbol} → skipping")
                        continue
                        
                    min_qty = float(limits.get("min_qty", 0.0))
                    min_notional = float(limits.get("min_notional", 5.0))
                    
                    if min_qty <= 0 or min_notional <= 0:
                        logger.debug(f"⚠️ Invalid limits for {symbol} → skipping")
                        continue
                        
                except Exception as e:
                    logger.debug(f"⚠️ Could not fetch limits for {symbol}: {e}")
                    continue

                # Calculate minimum trade value
                min_trade_value = price * min_qty
                
                # Check if balance is sufficient for minimum trade
                if min_trade_value > balance:
                    logger.warning(
                        f"⛔ Skipping {symbol}: balance ${balance:.2f} < min trade ${min_trade_value:.2f}"
                    )
                    continue

                # Validate stop loss and take profit
                stop_loss = sig.get("stop_loss") or sig.get("sl", 0)
                take_profit = sig.get("take_profit") or sig.get("tp", 0)
                
                side = str(sig.get("side", "BUY")).upper()
                
                # Calculate risk/reward ratio if possible
                if stop_loss and take_profit:
                    try:
                        stop_loss = float(stop_loss)
                        take_profit = float(take_profit)
                        
                        if side == "BUY":
                            gross_sl_dist_pct = ((price - stop_loss) / price) * 100 if price > 0 else 0
                            gross_tp_dist_pct = ((take_profit - price) / price) * 100 if price > 0 else 0
                        else:  # SELL
                            gross_sl_dist_pct = ((stop_loss - price) / price) * 100 if price > 0 else 0
                            gross_tp_dist_pct = ((price - take_profit) / price) * 100 if price > 0 else 0

                        # Load fee configuration
                        taker_fee_pct = cfg.get("fees", "TAKER_FEE_PCT", 0.06, float)  # Default Bybit Taker Fee
                        total_fees_pct = taker_fee_pct * 2  # Open + Close

                        # Calculate NET distances after fees
                        net_sl_dist_pct = gross_sl_dist_pct - total_fees_pct
                        net_tp_dist_pct = gross_tp_dist_pct - total_fees_pct

                        # Log net values for transparency
                        logger.debug(
                            f"[{symbol}] Gross SL: {gross_sl_dist_pct:.3f}%, Gross TP: {gross_tp_dist_pct:.3f}% | "
                            f"After Fees ({total_fees_pct:.3f}%): Net SL: {net_sl_dist_pct:.3f}%, Net TP: {net_tp_dist_pct:.3f}%"
                        )

                        # Validate MINIMUM NET distances (after fees)
                        min_net_sl_pct = cfg.get("trading", "MIN_NET_SL_PCT", 0.15, float)
                        min_net_tp_pct = cfg.get("trading", "MIN_NET_TP_PCT", 0.5, float)

                        if net_sl_dist_pct < min_net_sl_pct:
                            logger.warning(
                                f"❌ {symbol}: Net SL too tight after fees "
                                f"({gross_sl_dist_pct:.2f}% gross → {net_sl_dist_pct:.2f}% net < {min_net_sl_pct}%) - blocking trade"
                            )
                            continue

                        if net_tp_dist_pct < min_net_tp_pct:
                            logger.warning(
                                f"❌ {symbol}: Net TP too small after fees "
                                f"({gross_tp_dist_pct:.2f}% gross → {net_tp_dist_pct:.2f}% net < {min_net_tp_pct}%) - blocking trade"
                            )
                            continue

                        # Validate MINIMUM GROSS distances (optional legacy check)
                        min_sl_pct = cfg.get("trading", "MIN_SL_PCT", 0.5, float)
                        min_tp_pct = cfg.get("trading", "MIN_TP_PCT", 1.0, float)
                        
                        if gross_sl_dist_pct < min_sl_pct:
                            logger.warning(f"❌ {symbol}: Gross SL too tight ({gross_sl_dist_pct:.2f}% < {min_sl_pct}%) - blocking trade")
                            continue
                            
                        if gross_tp_dist_pct < min_tp_pct:
                            logger.warning(f"❌ {symbol}: Gross TP too close ({gross_tp_dist_pct:.2f}% < {min_tp_pct}%) - blocking trade")
                            continue
                        
                        # Validate risk/reward ratio (using NET values for smarter decisions)
                        rr_ratio = net_tp_dist_pct / net_sl_dist_pct if net_sl_dist_pct > 0 else 0
                        min_rr = cfg.get("trading", "MIN_RR_RATIO", 1.5, float)
                        
                        if rr_ratio < min_rr:
                            logger.warning(
                                f"❌ {symbol}: Poor NET R:R ({rr_ratio:.2f} < {min_rr}) after fees - blocking trade | "
                                f"Net TP: {net_tp_dist_pct:.2f}%, Net SL: {net_sl_dist_pct:.2f}%"
                            )
                            continue
                            
                    except Exception as e:
                        logger.debug(f"⚠️ Error calculating risk metrics for {symbol}: {e}")

                # Add timestamp if not present
                if "timestamp" not in sig:
                    sig["timestamp"] = datetime.now(timezone.utc).isoformat()
                
                # Add data version
                sig["data_version"] = getattr(ml_config, '__version__', '1.0.0') if 'ml_config' in globals() else '1.0.0'
                
                filtered.append(sig)

                # Record signal to database
                if DB_AVAILABLE:
                    try:
                        insert_signal(
                            symbol=symbol,
                            direction=str(sig.get("side", "BUY")),
                            price=float(price),
                            sl=float(stop_loss) if stop_loss else 0,
                            tp=float(take_profit) if take_profit else 0,
                            timeframe=str(sig.get("timeframe") or ""),
                            confidence=float(sig.get("strength") or 0),
                            session_tag=str(sig.get("session") or ""),
                            raw=sig,
                        )
                    except Exception as db_err:
                        logger.debug(f"⚠️ insert_signal failed for {symbol}: {db_err}")

            except Exception as e:
                logger.error(f"❌ Error processing signal: {e}")
                continue

        logger.success(
            f"📊 {len(filtered)}/{len(signals)} signals are tradable based on available balance and risk parameters"
        )
        return filtered

    except Exception as e:
        logger.error(f"❌ fetch_signals failed: {e}")
        logger.error(traceback.format_exc())
        return []

# ============================================================
# 🚀 ENHANCED SIGNAL GENERATION - With Momentum Detection
# ============================================================
def enhanced_generate_signals() -> List[Dict[str, Any]]:
    """
    Generate enhanced trading signals combining:
    - Regular strategy signals
    - Momentum detection for breakout opportunities
    - Comprehensive validation and error handling
    
    Returns:
        List of combined regular and momentum signals
    """
    try:
        logger.info("🔍 Generating enhanced signals (Strategy + Momentum)...")
        
        # Update symbols list periodically
        try:
            run_update_process()
            loader.reload_if_needed()
            logger.success("✅ Symbols list updated and config reloaded")
        except Exception as e:
            logger.error(f"⚠️ Failed to update symbols list: {e}")

        # Generate regular signals
        try:
            normal_signals = fetch_signals()
            logger.info(f"📊 Generated {len(normal_signals)} regular signals")
        except Exception as e:
            logger.error(f"❌ Failed to generate regular signals: {e}")
            normal_signals = []

        # Initialize momentum signals list
        momentum_signals = []
        
        # Load configuration safely
        try:
            cfg = loader
            min_momentum_pct = cfg.get("momentum_trading", "MIN_MOMENTUM_PCT", 20.0, float)
            min_volume_usdt = cfg.get("momentum_trading", "MIN_VOLUME_USDT", 2000000, float)
            momentum_symbols_str = cfg.get("momentum_trading", "MOMENTUM_SYMBOLS", "")
            momentum_enabled = cfg.get("momentum_trading", "ENABLED", False, bool)
            
            if not momentum_enabled:
                logger.info("ℹ️ Momentum trading disabled in configuration")
                return normal_signals
                
            if not momentum_symbols_str:
                logger.info("ℹ️ No momentum symbols configured")
                return normal_signals

            momentum_symbols = [s.strip() for s in momentum_symbols_str.split(",") if s.strip()]
            if not momentum_symbols:
                logger.info("ℹ️ No valid momentum symbols found")
                return normal_signals
                
        except Exception as e:
            logger.error(f"❌ Failed to load momentum configuration: {e}")
            return normal_signals

        # Analyze momentum symbols
        logger.info(f"🚀 Analyzing {len(momentum_symbols)} momentum symbols...")
        
        for symbol in momentum_symbols:
            try:
                # Skip if already have a regular signal or open position
                if any(s.get("symbol") == symbol for s in normal_signals):
                    logger.debug(f"⏭️ Skipping {symbol} - already has regular signal")
                    continue
                
                if order_manager and any(p.get("symbol") == symbol for p in order_manager.active_orders.values()):
                    logger.debug(f"⏭️ Skipping {symbol} - already has open position")
                    continue

                # Fetch kline data
                try:
                    interval = getattr(strategy, 'klines_interval', "15") if hasattr(strategy, 'klines_interval') else "15"
                    kl = session.client.get_kline(category="linear", symbol=symbol, interval=interval, limit=100)
                    
                    if kl.get("retCode") != 0:
                        logger.debug(f"⚠️ Failed to fetch kline data for {symbol}: {kl.get('retMsg', 'Unknown error')}")
                        continue
                        
                    lst = kl.get("result", {}).get("list", [])
                    if not lst:
                        logger.debug(f"⚠️ No kline data for {symbol}")
                        continue
                        
                except Exception as e:
                    logger.debug(f"⚠️ Error fetching kline data for {symbol}: {e}")
                    continue

                # Process data to DataFrame
                try:
                    df = pd.DataFrame(lst, columns=[
                        "open_time", "open", "high", "low", "close", "volume", "turnover"
                    ])
                    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric)
                    
                    # Reverse order to have most recent at the end
                    df = df.iloc[::-1].reset_index(drop=True)
                    
                    if len(df) < 50:
                        logger.debug(f"⚠️  Insufficient data for {symbol} (need 50, have {len(df)})")
                        continue
                        
                except Exception as e:
                    logger.error(f"❌ Error processing kline data for {symbol}: {e}")
                    continue

                # Calculate momentum metrics
                try:
                    lookback = min(96, len(df))
                    oldest_price = df.iloc[-lookback]["close"]
                    current_price = df.iloc[-1]["close"]
                    
                    if oldest_price <= 0 or current_price <= 0:
                        logger.debug(f"⚠️ Invalid price data for {symbol}")
                        continue
                        
                    momentum_24h = ((current_price - oldest_price) / oldest_price) * 100
                    
                    current_vol = df["volume"].iloc[-1]
                    avg_vol = df["volume"].iloc[-21:-1].mean() if len(df) >= 22 else df["volume"].iloc[:-1].mean()
                    volume_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
                    
                    notional_vol = current_vol * current_price
                    
                except Exception as e:
                    logger.error(f"❌ Error calculating momentum metrics for {symbol}: {e}")
                    continue

                # Apply momentum filters
                try:
                    if (momentum_24h > min_momentum_pct and 
                        volume_ratio > 1.5 and 
                        notional_vol > min_volume_usdt):
                        
                        # Calculate strength based on momentum
                        strength = min(100.0, max(60.0, 60.0 + (momentum_24h / 2.0)))
                        
                        # Calculate stops (7% SL, 15% TP)
                        stop_loss = current_price * 0.93
                        take_profit = current_price * 1.15
                        
                        momentum_signal = {
                            "symbol": symbol,
                            "side": "BUY",
                            "price": current_price,
                            "strength": strength,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "type": "MOMENTUM",
                            "is_momentum": True,
                            "momentum_pct": momentum_24h,
                            "volume_ratio": volume_ratio,
                            "notional_volume": notional_vol,
                            "reasons": [
                                f"Momentum: +{momentum_24h:.1f}%",
                                f"Volume Spike: {volume_ratio:.1f}x",
                                f"Notional Volume: ${notional_vol:,.0f}"
                            ],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "data_version": getattr(ml_config, '__version__', '1.0.0') if 'ml_config' in globals() else '1.0.0'
                        }
                        
                        momentum_signals.append(momentum_signal)
                        logger.success(f"🚀 Momentum Signal: {symbol} (+{momentum_24h:.1f}%)")
                        
                except Exception as e:
                    logger.error(f"❌ Error creating momentum signal for {symbol}: {e}")
                    continue

            except Exception as e:
                logger.error(f"❌ Error checking momentum for {symbol}: {e}")
                logger.error(traceback.format_exc())
                continue

        # Combine and return signals
        total_signals = normal_signals + momentum_signals
        logger.info(f"🎯 Total signals generated: {len(total_signals)} "
                   f"({len(normal_signals)} regular + {len(momentum_signals)} momentum)")
        
        return total_signals
        
    except Exception as e:
        logger.error(f"❌ Enhanced signals generation failed: {e}")
        logger.error(traceback.format_exc())
        # Fallback to regular signals
        try:
            return fetch_signals()
        except:
            return [] 

# ============================================================
# 💰 TRADE EXECUTION - UNIFIED & BACKWARD COMPATIBLE
# ============================================================
def execute_trade(signal: Dict[str, Any], is_momentum: bool = None) -> Optional[Dict[str, Any]]:
    """
    Main entry point for trade execution with comprehensive validation and error handling.
    
    Routes to appropriate handler based on signal type and respects overrides.
    
    Args:
        signal: Trading signal dictionary
        is_momentum: Explicitly specify if this is a momentum trade (overrides signal detection)
    
    Returns:
        Trade result dictionary or None if failed
    """
    try:
        # Validate signal structure
        if not isinstance(signal, dict):
            logger.error("❌ Invalid signal: must be a dictionary")
            return None

        # Extract and validate symbol and side
        symbol = str(signal.get("symbol", "")).upper().strip()
        side = str(signal.get("side", "BUY")).upper().strip()
        
        if not symbol:
            logger.error("❌ Invalid signal: missing or empty symbol")
            return None
            
        if side not in ["BUY", "SELL"]:
            logger.error(f"❌ Invalid signal: invalid side '{side}' (must be BUY or SELL)")
            return None

        # Extract and validate strength
        try:
            strength = float(signal.get("strength") or signal.get("data", {}).get("strength", 0))
            # Clamp strength to reasonable range
            strength = max(0.0, min(100.0, strength))
            signal['strength'] = strength
        except (ValueError, TypeError):
            strength = 0.0
            signal['strength'] = strength
            logger.warning(f"⚠️ Invalid strength in signal for {symbol}, using default: 0.0")

        # Determine if this is a momentum trade
        if is_momentum is not None:
            is_momentum_trade = bool(is_momentum)
        else:
            signal_type = str(signal.get("type", "")).lower().strip()
            is_momentum_trade = bool(signal.get("is_momentum", False)) or signal_type == "momentum"

        # Log trade routing
        if is_momentum_trade:
            logger.info(f"🚀 Routing {symbol} to MOMENTUM handler (Strength: {strength:.1f}%)")
            return execute_momentum_trade(signal)
        else:
            logger.info(f"📊 Routing {symbol} to REGULAR handler (Strength: {strength:.1f}%)")
            return execute_regular_trade(signal)
            
    except Exception as e:
        logger.error(f"❌ Trade execution routing error for {signal.get('symbol', 'Unknown')}: {e}")
        logger.error(traceback.format_exc())
        return None

def execute_regular_trade(signal: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
    """
    Execute regular (non-momentum) trades with comprehensive validation.
    
    Args:
        signal: Trading signal dictionary 
        **kwargs: Additional keyword arguments
        
    Returns:
        Trade result dictionary or None if failed
    """
    try:
        logger.debug(f"📋 Executing regular trade for {signal.get('symbol', 'Unknown')}")
        return _execute_trade_internal(signal, is_momentum=False)
    except Exception as e:
        logger.error(f"❌ Regular trade execution failed: {e}")
        logger.error(traceback.format_exc())
        return None

def execute_momentum_trade(signal: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
    """
    Execute momentum trades with specialized logic and validation.
    
    Args:
        signal: Trading signal dictionary
        **kwargs: Additional keyword arguments
        
    Returns:
        Trade result dictionary or None if failed
    """
    try:
        logger.debug(f"🚀 Executing momentum trade for {signal.get('symbol', 'Unknown')}")
        return _execute_trade_internal(signal, is_momentum=True)
    except Exception as e:
        logger.error(f"❌ Momentum trade execution failed: {e}")
        logger.error(traceback.format_exc())
        return None

def _execute_trade_internal(signal: Dict[str, Any], is_momentum: bool = False) -> Optional[Dict[str, Any]]:
    """
    Internal unified trade execution logic with comprehensive validation and error handling.
    Handles both regular and momentum trades with conditional logic.
    """
    global risk, ml_manager
    
    try:
        # ========================================
        # 1. EXTRACT & VALIDATE SIGNAL - Enhanced Validation
        # ========================================
        if not isinstance(signal, dict):
            logger.error("❌ Invalid signal: must be a dictionary")
            return None

        symbol = str(signal.get("symbol", "")).upper().strip()
        side = str(signal.get("side", "BUY")).upper().strip()
        
        if not symbol:
            logger.error("❌ Invalid signal: missing symbol")
            return None
            
        if side not in ["BUY", "SELL"]:
            logger.error(f"❌ Invalid signal: invalid side '{side}'")
            return None

        # Validate strength
        try:
            strength = float(signal.get("strength") or signal.get("data", {}).get("strength", 0))
            strength = max(0.0, min(100.0, strength))  # Clamp to 0-100
        except (ValueError, TypeError):
            strength = 0.0
            logger.warning(f"⚠️ Invalid strength for {symbol}, using default: 0.0")

        trade_type = "MOMENTUM" if is_momentum else "REGULAR"
        logger.info(f"📋 Processing {trade_type} trade: {symbol} {side} (Strength: {strength:.1f}%)")
        
        # ========================================
        # 2. FETCH ACCOUNT INFO - With Comprehensive Error Handling
        # ========================================
        try:
            info = session.get_account_info()
            
            if not info or not isinstance(info, dict):
                logger.error("❌ Could not fetch valid account info")
                return None
            
            # Extract balance and equity with fallbacks
            try:
                balance = float(info.get("balance", 0) or 0.0)
                equity = float(info.get("equity", 0) or 0.0)
                
                if balance <= 0 and equity <= 0:
                    logger.error("❌ Both balance and equity are zero or negative")
                    return None
                
                # Determine effective balance
                if equity > 0 and balance > 0:
                    effective_balance = max(equity, balance)
                    logger.debug(f"📊 Using max of equity (${equity:.2f}) and balance (${balance:.2f}): ${effective_balance:.2f}")
                elif equity > 0:
                    effective_balance = equity
                    logger.warning(f"⚠️ Balance is 0, using equity: ${equity:.2f}")
                else:
                    effective_balance = balance
                    logger.warning(f"⚠️ Equity is 0, using balance: ${balance:.2f}")
                
                logger.info(f"💰 Account Status: Balance=${balance:.2f}, Equity=${equity:.2f}, Effective=${effective_balance:.2f}")
                
            except (ValueError, TypeError) as e:
                logger.error(f"❌ Invalid balance/equity values: {e}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error fetching account info: {e}")
            logger.error(traceback.format_exc())
            return None
        
        # ========================================
        # 3. GET ENTRY PRICE - With Fallback and Validation
        # ========================================
        try:
            entry_price = signal.get("entry_price") or signal.get("price")
            
            if not entry_price:
                logger.info(f"⚠️ No entry price in signal - fetching current price for {symbol}")
                try:
                    entry_price = session.get_current_price(symbol)
                    if not entry_price:
                        logger.error(f"❌ Could not fetch current price for {symbol}")
                        return None
                except Exception as e:
                    logger.error(f"❌ Price fetch failed for {symbol}: {e}")
                    return None
            
            entry_price = float(entry_price)
            
            if entry_price <= 0:
                logger.error(f"❌ Invalid entry price for {symbol}: {entry_price}")
                return None
                
            logger.info(f"📈 Entry Price: ${entry_price:.8f}")
            
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Invalid entry price format: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error processing entry price: {e}")
            return None
        
        # ========================================
        # 4. CALCULATE STOP LOSS & TAKE PROFIT - With Configuration Loading
        # ========================================
        try:
            # Load configuration safely
            cfg = loader
            
            if is_momentum:
                # Momentum-specific SL/TP from config
                momentum_sl_pct = cfg.get("momentum_trading", "MOMENTUM_SL_PCT", 7.0, float)  # 7% default for momentum
                momentum_tp_pct = cfg.get("momentum_trading", "MOMENTUM_TP_PCT", 15.0, float)  # 15% default
                
                stop_loss = signal.get("stop_loss", 0)
                take_profit = signal.get("take_profit", 0)
                
                if not stop_loss or not take_profit:
                    logger.info(f"🔧 Calculating momentum stops from config ({momentum_sl_pct}% SL, {momentum_tp_pct}% TP)")
                    if side == "BUY":
                        stop_loss = entry_price * (1 - momentum_sl_pct / 100)
                        take_profit = entry_price * (1 + momentum_tp_pct / 100)
                    else:  # SELL
                        stop_loss = entry_price * (1 + momentum_sl_pct / 100)
                        take_profit = entry_price * (1 - momentum_tp_pct / 100)
            else:
                # Regular trade SL/TP
                stop_loss = signal.get("stop_loss") or signal.get("sl", 0)
                take_profit = signal.get("take_profit") or signal.get("tp", 0)
                
                if not stop_loss:
                    default_sl_pct = cfg.get("trading", "DEFAULT_SL_PCT", 2.0, float)
                    stop_loss = entry_price * (1 - default_sl_pct / 100) if side == "BUY" else entry_price * (1 + default_sl_pct / 100)
                    logger.info(f"🔧 Using default SL: {default_sl_pct}%")
                    
                if not take_profit:
                    default_tp_pct = cfg.get("trading", "DEFAULT_TP_PCT", 5.0, float)
                    take_profit = entry_price * (1 + default_tp_pct / 100) if side == "BUY" else entry_price * (1 - default_tp_pct / 100)
                    logger.info(f"🔧 Using default TP: {default_tp_pct}%")
            
            # Convert to float and validate
            stop_loss = float(stop_loss)
            take_profit = float(take_profit)
            
            # Validate SL/TP values
            if side == "BUY":
                if stop_loss >= entry_price:
                    logger.error(f"❌ Invalid BUY stops: SL ({stop_loss:.8f}) >= Entry ({entry_price:.8f})")
                    return None
                if take_profit <= entry_price:
                    logger.error(f"❌ Invalid BUY stops: TP ({take_profit:.8f}) <= Entry ({entry_price:.8f})")
                    return None
            else:  # SELL
                if stop_loss <= entry_price:
                    logger.error(f"❌ Invalid SELL stops: SL ({stop_loss:.8f}) <= Entry ({entry_price:.8f})")
                    return None
                if take_profit >= entry_price:
                    logger.error(f"❌ Invalid SELL stops: TP ({take_profit:.8f}) >= Entry ({entry_price:.8f})")
                    return None
            
            # Log initial stops
            logger.info(f"🎯 Initial Stops: SL={stop_loss:.8f}, TP={take_profit:.8f}")
            
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Invalid SL/TP values: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error calculating stops: {e}")
            return None
        
        # ========================================
        # 🤖 ML ENHANCEMENT: Get ML predictions with comprehensive error handling
        # ========================================
        traditional_sl = stop_loss
        traditional_tp = take_profit
        
        if ml_manager and hasattr(ml_manager, 'enabled') and ml_manager.enabled and not is_momentum:
            try:
                use_ml = cfg.get("machine_learning", "USE_ML_PREDICTIONS", True, bool)
                
                if use_ml:
                    logger.info("🧠 ML System ACTIVE — applying ML enhancements to stops...")
                    logger.info("🤖 Applying ML enhancements to stops...")
                    
                    try:
                        # Fetch recent candles for ML features
                        kl = session.client.get_kline(
                            category="linear",
                            symbol=symbol,
                            interval="15",
                            limit=100
                        )
                        
                        if kl.get("retCode") != 0:
                            logger.warning(f"⚠️ Failed to fetch kline data for ML: {kl.get('retMsg', 'Unknown error')}")
                        else:
                            lst = kl.get("result", {}).get("list", [])
                            if lst:
                                # Create DataFrame with proper column names
                                df_ml = pd.DataFrame(lst, columns=[
                                    "open_time", "open", "high", "low", 
                                    "close", "volume", "turnover"
                                ])
                                
                                # Convert to numeric
                                numeric_columns = ["open", "high", "low", "close", "volume"]
                                df_ml[numeric_columns] = df_ml[numeric_columns].apply(pd.to_numeric, errors='coerce')
                                
                                # Remove rows with NaN values
                                df_ml = df_ml.dropna(subset=numeric_columns)
                                
                                if len(df_ml) < 20:
                                    logger.warning(f"⚠️  Insufficient data for ML prediction: {len(df_ml)} rows")
                                else:
                                    # Calculate technical indicators
                                    try:
                                        # EMA calculations
                                        df_ml["ema_fast"] = df_ml["close"].ewm(span=12, min_periods=1).mean()
                                        df_ml["ema_slow"] = df_ml["close"].ewm(span=26, min_periods=1).mean()
                                        
                                        # ATR calculation
                                        df_ml["high_low"] = df_ml["high"] - df_ml["low"]
                                        df_ml["high_close"] = abs(df_ml["high"] - df_ml["close"].shift(1))
                                        df_ml["low_close"] = abs(df_ml["low"] - df_ml["close"].shift(1))
                                        df_ml[["high_low", "high_close", "low_close"]] = df_ml[["high_low", "high_close", "low_close"]].fillna(0)
                                        df_ml["true_range"] = df_ml[["high_low", "high_close", "low_close"]].max(axis=1)
                                        df_ml["atr"] = df_ml["true_range"].rolling(14, min_periods=1).mean()
                                        
                                        # Volume ratio
                                        df_ml["volume_ma"] = df_ml["volume"].rolling(20, min_periods=1).mean()
                                        df_ml["volume_ratio"] = df_ml["volume"] / (df_ml["volume_ma"] + 1e-10)
                                        
                                        # RSI calculation
                                        delta = df_ml["close"].diff()
                                        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
                                        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
                                        rs = gain / (loss + 1e-10)
                                        df_ml["rsi"] = 100 - (100 / (1 + rs))
                                        
                                    except Exception as e:
                                        logger.warning(f"⚠️ Error calculating technical indicators for ML: {e}")
                                    
                                    # Get ML prediction
                                    try:
                                        ml_output = ml_manager.predict_stops(signal, df_ml, entry_price)
                                        
                                        if ml_output and isinstance(ml_output, dict):
                                            ml_sl = ml_output.get('sl_price')
                                            ml_tp = ml_output.get('tp_price')
                                            
                                            if ml_sl and ml_tp:
                                                blend_weight = cfg.get("machine_learning", "ML_BLEND_WEIGHT", 0.7, float)
                                                
                                                # Apply blend weight with validation
                                                blend_weight = max(0.0, min(1.0, blend_weight))
                                                
                                                stop_loss = traditional_sl * (1 - blend_weight) + ml_sl * blend_weight
                                                take_profit = traditional_tp * (1 - blend_weight) + ml_tp * blend_weight
                                                
                                                logger.info(
                                                    f"🤖 ML Blend ({blend_weight:.0%}):\n"
                                                    f"   Traditional: SL={traditional_sl:.8f}, TP={traditional_tp:.8f}\n"
                                                    f"   ML Predict:  SL={ml_sl:.8f}, TP={ml_tp:.8f}\n"
                                                    f"   Final:       SL={stop_loss:.8f}, TP={take_profit:.8f}"
                                                )
                                            else:
                                                logger.debug("⚠️ ML prediction returned None values, using traditional")
                                        elif ml_output and isinstance(ml_output, tuple) and len(ml_output) >= 2:
                                            ml_sl, ml_tp = ml_output[:2]
                                            if ml_sl and ml_tp:
                                                blend_weight = cfg.get("machine_learning", "ML_BLEND_WEIGHT", 0.7, float)
                                                blend_weight = max(0.0, min(1.0, blend_weight))
                                                
                                                stop_loss = traditional_sl * (1 - blend_weight) + ml_sl * blend_weight
                                                take_profit = traditional_tp * (1 - blend_weight) + ml_tp * blend_weight
                                                
                                                logger.info(
                                                    f"🤖 ML Blend ({blend_weight:.0%}):\n"
                                                    f"   Traditional: SL={traditional_sl:.8f}, TP={traditional_tp:.8f}\n"
                                                    f"   ML Predict:  SL={ml_sl:.8f}, TP={ml_tp:.8f}\n"
                                                    f"   Final:       SL={stop_loss:.8f}, TP={take_profit:.8f}"
                                                )
                                        else:
                                            logger.debug("⚠️  ML prediction returned invalid format, using traditional")
                                            
                                    except Exception as e:
                                        logger.warning(f"⚠️ ML prediction failed: {e}")
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Error preparing data for ML prediction: {e}")
                        
            except Exception as e:
                logger.warning(f"⚠️ ML enhancement failed: {e}")
        
        # ========================================
        # 5. CALCULATE POSITION SIZE - With Comprehensive Validation
        # ========================================
        try:
            position_usdt = calculate_dynamic_position_size(
                symbol=symbol,
                balance=balance,
                equity=equity,
                entry_price=entry_price,
                signal=signal,
            )
            
            regime_mult = float(signal.get('position_size', 1.0))
            position_usdt *= regime_mult
            
            min_notional = 6.0 
            try:
                limits = session.get_symbol_limits(symbol)
                if isinstance(limits, dict):
                    min_notional = float(limits.get("min_notional", 6.0))
            except:
                pass

            if position_usdt < min_notional:
                logger.warning(f"⚠️ {symbol} size ${position_usdt:.2f} < min ${min_notional:.2f} (after {regime_mult} mult). Adjusting to minimum.")
                position_usdt = min_notional

            # 4. Final Quantity Calculation
            quantity = position_usdt / entry_price
            quantity = round_quantity(quantity, symbol)
            
            if quantity <= 0:
                logger.error(f"❌ Final quantity for {symbol} is 0 after rounding.")
                return None

            # Store final quantity back in signal for order_manager
            signal['quantity'] = quantity
            
        except Exception as e:
            logger.error(f"❌ Position sizing error: {e}")
            return None
        
        # ========================================
        # 6. CALCULATE RISK METRICS - With Comprehensive Validation
        # ========================================
        try:
            if side == "BUY":
                potential_gain = (take_profit - entry_price) * quantity
                potential_loss = (entry_price - stop_loss) * quantity
            else:  # SELL
                potential_gain = (entry_price - take_profit) * quantity
                potential_loss = (stop_loss - entry_price) * quantity
            
            # Validate risk metrics
            if potential_loss <= 0:
                logger.warning(f"⚠️ Invalid risk calculation for {symbol}: potential_loss={potential_loss}")
                return None
                
            rr_ratio = potential_gain / potential_loss if potential_loss > 0 else 0
            expected_gain_pct = (potential_gain / position_usdt) * 100 if position_usdt > 0 else 0
            
            logger.info(
                f"📊 Trade Analysis:\n"
                f"   Entry: ${entry_price:.8f}\n"
                f"   SL: ${stop_loss:.8f}\n"
                f"   TP: ${take_profit:.8f}\n"
                f"   Potential Gain: ${potential_gain:.4f} ({expected_gain_pct:.2f}%)\n"
                f"   Potential Loss: ${potential_loss:.4f}\n"
                f"   Risk/Reward: 1:{rr_ratio:.2f}"
            )
            
        except Exception as e:
            logger.error(f"❌ Error calculating risk metrics: {e}")
            return None
        
        # ========================================
        # 7. VALIDATE TRADE CONDITIONS - Comprehensive Checks
        # ========================================
        
        # Check fees impact
        if risk and hasattr(risk, "should_skip_trade_due_to_fees"):
            try:
                if risk.should_skip_trade_due_to_fees(symbol, position_usdt, expected_gain_pct):
                    logger.warning(f"⚠️ Trade on {symbol} blocked: fees too high relative to expected gain")
                    return None
            except Exception as e:
                logger.warning(f"⚠️ Error checking fees impact: {e}")
        
        # Check minimum RR ratio (lower for momentum)
        try:

            min_rr_ratio = cfg.get("momentum_trading", "MIN_RR_RATIO", 1.2, float) if is_momentum else cfg.get("trading", "MIN_RR_RATIO", 1.5, float)
            
            if round(rr_ratio, 2) < min_rr_ratio:
                logger.warning(
                    f"⚠️ Trade on {symbol} blocked: RR {rr_ratio:.2f} (rounded: {round(rr_ratio, 2)}) < minimum {min_rr_ratio}"
                )
                return None
            
            logger.debug(f"✅ {symbol} RR Check passed: {rr_ratio:.2f} >= {min_rr_ratio}")
            
        except Exception as e:
            logger.warning(f"⚠️ Error validating RR ratio: {e}")

        # Check Risk Manager approval
        if risk and hasattr(risk, "can_open_trade"):
            try:
                try:
                    can_trade = risk.can_open_trade(symbol, position_usdt, side)
                except TypeError:
                    can_trade = risk.can_open_trade(symbol, position_usdt)
                
                if not can_trade:
                    logger.warning(f"🚫 RiskManager blocked trade on {symbol}")
                    return None
            except Exception as e:
                logger.warning(f"⚠️ Error checking risk manager approval: {e}")
        
        # ========================================
        # 8. EXECUTE OR SIMULATE TRADE - With Comprehensive Error Handling
        # ========================================
        try:
            if ENABLE_LIVE_TRADING and order_manager:
                logger.info(f"🚀 Executing LIVE {trade_type} trade for {symbol}...")
                
                result = order_manager.place_order_with_protection(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    signal_strength=strength,
                    equity=equity,
                    is_momentum_trade=is_momentum,
                )
                
                if result and isinstance(result, dict) and result.get("status") == "SUCCESS":
                    entry_exec = float(result.get("price", entry_price))
                    actual_position_usdt = entry_exec * quantity
                    
                    logger.success(f"✅ {trade_type} trade executed: {symbol} {side} @ {entry_exec:.8f}")
                    
                    # Build Telegram message
                    emoji = "🚀" if is_momentum else "✅"
                    msg = (
                        f"{emoji} <b>{trade_type} Trade Executed</b>\n"
                        f"Symbol: {symbol}\n"
                        f"Side: {side}\n"
                        f"Entry: {entry_exec:.8f}\n"
                        f"Size: ${actual_position_usdt:.2f}\n"
                        f"Qty: {quantity:.6f}\n"
                        f"SL: {stop_loss:.8f}\n"
                        f"TP: {take_profit:.8f}\n"
                        f"Strength: {strength:.1f}%\n"
                        f"RR Ratio: 1:{rr_ratio:.2f}"
                    )
                    
                    # Add momentum-specific info
                    if is_momentum:
                        try:
                            momentum_max_time = cfg.get("momentum_trading", "MOMENTUM_MAX_TIME_MINUTES", 45, int)
                            price_change_24h = signal.get("price_change_24h", 0)
                            msg += (
                                f"\n24h Change: {float(price_change_24h):+.1f}%\n"
                                f"Max Hold: {momentum_max_time}min"
                            )
                        except Exception as e:
                            logger.warning(f"⚠️ Error adding momentum info to message: {e}")
                    
                    safe_send(msg, parse_mode="HTML")
                    
                    # Record to database
                    if DB_AVAILABLE:
                        try:
                            meta = {
                                "strength": strength,
                                "rr_ratio": rr_ratio,
                                "expected_gain_pct": expected_gain_pct,
                                "is_momentum": is_momentum,
                                "signal_type": trade_type.lower(),
                                "position_size": position_usdt,
                                "quantity": quantity,
                                "entry_price": entry_price,
                                "stop_loss": stop_loss,
                                "take_profit": take_profit
                            }
                            
                            if is_momentum:
                                meta.update({
                                    "price_change_24h": signal.get("price_change_24h", 0),
                                    "price_change_4h": signal.get("price_change_4h", 0),
                                    "rsi": signal.get("rsi", 50),
                                    "volume_ratio": signal.get("volume_ratio", 1.0),
                                    "momentum_pct": signal.get("momentum_pct", 0)
                                })
                            
                            record_execution(
                                signal_id=signal.get("id"),
                                symbol=symbol,
                                direction=side,
                                entry=entry_exec,
                                size=actual_position_usdt,
                                sl=stop_loss,
                                tp=take_profit,
                                meta=meta,
                            )
                        except Exception as e:
                            logger.warning(f"⚠️ DB record failed: {e}")
                    
                    # Send trade data to ML system for learning
                    try:
                        send_trade_outcome_to_ml({
                            'symbol': symbol,
                            'side': side,
                            'entry_price': entry_price,
                            'exit_price': entry_exec,  # Use executed price
                            'pnl': 0,  # Will be updated when trade closes
                            'sl_used': stop_loss,
                            'tp_used': take_profit,
                            'features': signal.get('features', {}),
                            'signal_strength': strength,
                            'is_momentum': is_momentum,
                            'rr_ratio': rr_ratio,
                            'position_size': position_usdt,
                            'quantity': quantity
                        })
                    except Exception as e:
                        logger.debug(f"⚠️ Failed to send trade data to ML: {e}")
                    
                    return result
                
                else:
                    logger.error(f"❌ {trade_type} trade execution failed: {result}")
                    safe_send(
                        f"❌ <b>{trade_type} Trade Failed</b>\n{symbol} {side}",
                        parse_mode="HTML",
                    )
                    return None
            
            else:
                # SIMULATION MODE
                logger.info(f"📝 SIMULATED {trade_type} trade: {symbol} {side} ${position_usdt:.2f}")
                logger.info(f"   Entry: {entry_price:.8f}")
                logger.info(f"   SL: {stop_loss:.8f}")
                logger.info(f"   TP: {take_profit:.8f}")
                logger.info(f"   Expected Gain: {expected_gain_pct:.2f}%")
                logger.info(f"   Risk/Reward: 1:{rr_ratio:.2f}")
                
                return {
                    "status": "SIMULATED",
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "size": position_usdt,
                    "quantity": quantity,
                    "rr_ratio": rr_ratio,
                    "expected_gain_pct": expected_gain_pct,
                    "is_momentum": is_momentum,
                    "signal_type": trade_type.lower(),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Error executing trade: {e}")
            logger.error(traceback.format_exc())
            safe_send(f"⚠️ <b>Trade Execution Error</b>\n{str(e)}", parse_mode="HTML")
            return None
    
    except Exception as e:
        logger.error(f"❌ Trade execution error: {e}")
        logger.error(traceback.format_exc())
        safe_send(f"⚠️ <b>Trade Execution Error</b>\n{str(e)}", parse_mode="HTML")
        return None

# ============================================================
# 🔄 TRADING CYCLE
# ============================================================
def run_trading_cycle():
    """
    Main trading cycle execution with enhanced sync, safety checks, and comprehensive error handling.
    Monitors positions, evaluates risks, generates signals, and executes trades.
    Includes ML execution filtering, auto-flip logic, and time-based exits.
    """
    global LAST_SYMBOLS_UPDATE, cycle_count, consecutive_failures, ml_manager
    global correlation_manager, time_stop_manager, regime_detector
    global risk, order_manager, session, strategy, smart_safety, emergency_exit

    # 🧠 Circuit Breaker - Global State
    global CIRCUIT_BREAKER_ACTIVE, CIRCUIT_BREAKER_UNTIL

    # 💰 Price Cache for this cycle to minimize API calls
    _price_cache: Dict[str, Tuple[float, float]] = {}  # symbol -> (price, timestamp)

    current_time = time.time()
    cycle_start_time = time.time()

    try:
        # 🔌 Circuit Breaker Check
        if CIRCUIT_BREAKER_ACTIVE and time.time() < CIRCUIT_BREAKER_UNTIL:
            remaining = int(CIRCUIT_BREAKER_UNTIL - time.time())
            logger.warning(f"🛑 Circuit breaker active — skipping cycle ({remaining}s remaining)")
            return

        logger.info("=" * 70)
        logger.info(f"🔄 Starting Trading Cycle #{cycle_count + 1}")
        logger.info("=" * 70)

        # --- Update Symbols List Periodically ---
        if current_time - LAST_SYMBOLS_UPDATE > UPDATE_INTERVAL:
            try:
                logger.info("🕒 Periodic symbols list update starting...")
                run_update_process()
                LAST_SYMBOLS_UPDATE = current_time
                loader.reload_if_needed()
                logger.success("🔄 Symbols list updated and config reloaded")
            except Exception as e:
                logger.error(f"❌ Failed to run periodic symbols update: {e}")

        # --- Reload Configs ---
        try:
            loader.reload_if_needed()
            if strategy and hasattr(strategy, '_load_strategy_config'):
                strategy._load_strategy_config()
                logger.success("✅ Strategy configuration reloaded")
        except Exception as e:
            logger.debug(f"⚠️ Strategy config reload failed: {e}")

        cycle_count += 1

        # Validate critical components
        if not session:
            logger.error("❌ Session is not initialized")
            consecutive_failures += 1
            return
        
        if not order_manager and ENABLE_LIVE_TRADING:
            logger.error("❌ OrderManager is not initialized for live trading")
            consecutive_failures += 1
            return

        # --- 1. Fetch Account Information ---
        try:
            account = session.get_account_info()
            if not account or not isinstance(account, dict):
                logger.error("❌ Failed to fetch valid account information")
                return
            
            balance = float(account.get("balance", 0) or 0)
            equity = float(account.get("equity", 0) or 0)
            logger.info(f"💰 Account Status: Balance=${balance:.2f}, Equity=${equity:.2f}")

            # 💾 Record account metrics to DB for reporting
            try:
                from db_utils import record_metrics
                
                profit = equity - balance
                margin = float(account.get("margin", 0) or 0)
                free_margin = float(account.get("free_margin", 0) or 0)
                margin_level = float(account.get("margin_level", 0) or 0)

                success = record_metrics(
                    balance=balance,
                    equity=equity,
                    profit=profit,
                    margin=margin,
                    free_margin=free_margin,
                    margin_level=margin_level
                )
                
                if success:
                    logger.debug("✅ Account metrics recorded to database")
                else:
                    logger.warning("⚠️ Failed to record account metrics to DB")

            except Exception as e:
                logger.error(f"❌ Exception while recording account metrics: {e}")

        except Exception as e:
            logger.error(f"❌ Failed to fetch account info: {e}")
            return

        # --- 2. Fetch Exchange Positions & Sync ---
        open_positions = []
        try:
            open_positions = session.get_positions()
            logger.info(f"📊 Exchange reports {len(open_positions)} active positions")
            
            # Update Strategy internal tracking
            if strategy:
                strategy.open_positions = {}
                for pos in open_positions:
                    symbol = str(pos.get("symbol", "")).strip()
                    entry_price = float(pos.get("entryPrice") or pos.get("avgPrice", 0))
                    if entry_price > 0:
                        strategy.open_positions[symbol] = {
                            "symbol": symbol,
                            "side": str(pos.get("side", "Buy")).capitalize(),
                            "entry_price": entry_price,
                            "size": float(pos.get("size", 0)),
                            "entry_time": datetime.now(timezone.utc)
                        }

            # Sync OrderManager with Exchange
            if order_manager and ENABLE_LIVE_TRADING:
                order_manager.sync_and_cleanup_positions(open_positions)
                if risk and hasattr(risk, 'update_from_order_manager'):
                    risk.update_from_order_manager(order_manager)
            
        except Exception as e:
            logger.error(f"❌ Position sync failed: {e}")

        # --- 3. Emergency & Smart Safety Checks ---
        # (Keeping your existing logic for Emergency Exit and Smart Safety)
        if emergency_exit and open_positions:
            actions = emergency_exit.evaluate_positions(open_positions, account, getattr(order_manager, "active_orders", {}))
            if actions.get("close_all"):
                emergency_exit.execute_emergency_actions(actions, open_positions)
                return

        # --- 4. Monitor & Monitor Time Stops ---
        if order_manager and ENABLE_LIVE_TRADING:
            logger.success("👀 Monitoring open positions...")
            
            # Inline price helper using the cycle's cache
            def get_cached_price(symbol: str):
                now = time.time()
                if symbol in _price_cache and (now - _price_cache[symbol][1]) < 10:
                    return _price_cache[symbol][0]
                ticker = session.get_ticker(symbol)
                price = float(ticker.get("lastPrice", 0))
                if price > 0:
                    _price_cache[symbol] = (price, now)
                    return price
                return None

            # Time-Stop Logic
            if time_stop_manager:
                for order_id, pos in list(order_manager.active_orders.items()):
                    symbol = pos.get("symbol")
                    current_price = get_cached_price(symbol) or pos.get("entry_price")
                    should_close, reason = time_stop_manager.should_close_by_time(pos, current_price)
                    if should_close:
                        logger.warning(f"⏰ TIME STOP: Closing {symbol} - {reason}")
                        order_manager.close_position_market(symbol, pos.get("side"), reason=f"time_stop_{reason}")

            order_manager.monitor_positions()
            summary = order_manager.get_positions_summary()
            if risk: risk.update_position_count(summary.get("total_positions", 0))

        # --- 5. Signal Generation & Ranking ---
        cfg = loader
        use_enhanced = cfg.get("momentum_trading", "ENABLED", False, bool)
        raw_signals = enhanced_generate_signals() if use_enhanced else fetch_signals()

        if not raw_signals:
            logger.info("⭕ No signals to process")
            return

        signals = strategy.filter_best_signals(raw_signals)
        executed_count = 0
        blocked_count = 0

        # --- 6. Trade Execution Loop ---
        for i, signal in enumerate(signals, 1):
            try:
                sig_symbol = signal.get("symbol")
                side = signal.get("side", "BUY").capitalize()
                
                logger.info(f"📋 Processing Signal {i}/{len(signals)}: {sig_symbol} {side}")

                # 🧠 ML Execution Filter Check
                # If filter is enabled in INI, we check ML recommendation
                if ml_manager and getattr(ml_manager, 'execution_filter_enabled', False):
                    # Passing signal and None for df (assuming manager handles recent data fetch)
                    ml_analysis = ml_manager.analyze_trade(signal, None)
                    if ml_analysis and ml_analysis.get('recommendation') == "SKIP":
                        logger.warning(f"🚫 ML Filter: Blocked {sig_symbol} execution (Low confidence)")
                        blocked_count += 1
                        continue

                # Correlation Check
                if correlation_manager:
                    open_symbols = [p.get("symbol") for p in order_manager.active_orders.values()]
                    can_open, reason = correlation_manager.can_open_position(sig_symbol, open_symbols)
                    if not can_open:
                        logger.warning(f"🚫 {sig_symbol} blocked by correlation: {reason}")
                        continue

                # --- Regime Detection & Adjustments (Fixed for KeyError) ---
                if regime_detector:
                    try:
                        regime = regime_detector.detect_regime(sig_symbol)
                        adjustments = regime_detector.get_regime_adjustments(regime)
                        
                        # FIXED: Use .get() to avoid KeyError if 'position_size' is missing
                        current_size = signal.get("position_size", 1.0)
                        size_mult = adjustments.get("position_size_mult", 1.0)
                        signal["position_size"] = current_size * size_mult
                        
                        # Log adjustment if it changed
                        if size_mult != 1.0:
                            logger.info(f"📐 {sig_symbol} size adjusted by regime: {current_size:.2f} -> {signal['position_size']:.2f}")

                        if adjustments.get("skip_trade", False):
                            logger.warning(f"⏭️ {sig_symbol} skipped due to unfavorable regime: {regime}")
                            blocked_count += 1
                            continue
            
                    except Exception as e:
                        logger.error(f"❌ Regime adjustment failed for {sig_symbol}: {e}")

                # 🛡️ Risk Capacity Check (Using internal attributes)
                max_pos = 2
                if risk:
                    if hasattr(risk, 'max_total_positions'):
                        max_pos = risk.max_total_positions
                    elif hasattr(risk, 'config'):
                        max_pos = risk.config.get("risk_management", "MAX_TOTAL_POSITIONS", 2, int)

                current_pos_count = len(order_manager.active_orders)
                if current_pos_count >= max_pos:
                    logger.info(f"⏭️ Portfolio full ({current_pos_count}/{max_pos}). Skipping {sig_symbol}")
                    blocked_count += 1
                    continue

                # 🚀 Final Execution (Using verified: place_order_with_protection)
                if ENABLE_LIVE_TRADING and order_manager:
                    # 1. Get Entry Price (fallback to current if signal lacks it)
                    entry_price = float(signal.get('price') or signal.get('entry_price') or 0)
                    if entry_price <= 0:
                        entry_price = session.get_current_price(sig_symbol)
                    
                    # 2. Calculate Quantity using your INI settings
                    # We start with your $22.0 from POSITION_SIZE_USDT
                    base_size_usdt = float(cfg.get("risk_management", "POSITION_SIZE_USDT", 22.0))
                    
                    # Apply the Regime Multiplier (the 0.70)
                    regime_mult = float(signal.get("position_size", 1.0))
                    final_usdt = base_size_usdt * regime_mult
                    
                    # Safety Check: Enforce the $6.0 minimum from your INI
                    min_notional = float(cfg.get("risk_management", "MIN_NOTIONAL_USDT", 6.0))
                    if final_usdt < min_notional:
                        final_usdt = min_notional
                    
                    # Convert to actual coin quantity
                    calc_qty = final_usdt / entry_price if entry_price > 0 else 0
                    
                    logger.info(f"💰 Sizing: ${base_size_usdt} * {regime_mult} regime = ${final_usdt:.2f} (Qty: {calc_qty:.4f})")

                    # 3. Execute with calculated values
                    success_order = order_manager.place_order_with_protection(
                        symbol=sig_symbol,
                        side=side,
                        quantity=calc_qty,
                        entry_price=entry_price,
                        stop_loss=signal.get('stop_loss', 0),
                        take_profit=signal.get('take_profit', 0),
                        signal_strength=signal.get('strength', 0),
                        is_momentum_trade=signal.get('is_momentum', False),
                        signal_data=signal 
                    )
                    
                    if success_order:
                        executed_count += 1
                        logger.success(f"✅ Successfully executed trade for {sig_symbol}")
                    else:
                        logger.error(f"❌ Execution failed for {sig_symbol} in place_order_with_protection")

            except Exception as e:
                logger.error(f"❌ Signal processing error for {signal.get('symbol', 'Unknown')}: {e}")
                logger.error(traceback.format_exc())

        logger.info(f"🎯 Cycle Results: Executed {executed_count}, Blocked {blocked_count}")

    except Exception as e:
        logger.error(f"💥 Critical error in trading cycle: {e}")
        logger.error(traceback.format_exc())
        consecutive_failures += 1
    finally:
        elapsed = time.time() - cycle_start_time
        logger.info(f"✅ Trading Cycle #{cycle_count} completed in {elapsed:.2f}s")
        
        # 🔌 Circuit Breaker Trigger
        if consecutive_failures >= 3:
            CIRCUIT_BREAKER_ACTIVE = True
            CIRCUIT_BREAKER_UNTIL = time.time() + 300
            logger.critical("🚨 Circuit breaker TRIGGERED — pausing trading for 5 minutes")
            safe_send("🚨 <b>Circuit Breaker Activated</b>\nTrading paused for 5 minutes.", parse_mode="HTML")
            consecutive_failures = 0

# ============================================================
# 🔁 MAIN LOOP
# ============================================================
def main_loop():
    """
    Main bot execution loop with heartbeat, failure handling, and graceful shutdown.
    Runs until:
    - Signal received (SIGINT/SIGTERM)
    - Maximum consecutive failures reached
    - Critical error occurs
    """
    global running, heartbeat, reporter, last_heartbeat, consecutive_failures

    try:
        logger.info("=" * 70)
        logger.info("🚀 STARTING MAIN TRADING LOOP")
        logger.info("=" * 70)
        
        if not initialize_bot():
            logger.error("❌ Initialization failed - exiting")
            sys.exit(1)

        interval = loader.get("bot_core", "CYCLE_INTERVAL_SECONDS", 5, int)
        logger.info(f"⏱️  Starting main loop with {interval} second intervals")

        while running:
            try:
                # Dynamic interval from config
                try:
                    loader.reload_if_needed()
                    new_interval = loader.get("bot_core", "CYCLE_INTERVAL_SECONDS", 5, int)
                    if new_interval != interval:
                        logger.info(f"🔄 Cycle interval updated from {interval}s to {new_interval}s")
                        interval = new_interval
                except Exception as e:
                    logger.warning(f"⚠️ Failed to reload config or get interval: {e}")

                # Execute trading cycle
                cycle_start_time = time.time()
                run_trading_cycle()
                cycle_end_time = time.time()
                
                # Calculate actual cycle duration
                actual_duration = cycle_end_time - cycle_start_time
                
                # Heartbeat to database
                if time.time() - last_heartbeat >= 30:
                    if heartbeat:
                        try:
                            acct = session.get_account_info()
                            equity = acct.get("equity") if acct else None
                            balance = acct.get("balance") if acct else None
                            
                            open_pos_count = 0
                            if order_manager and hasattr(order_manager, 'active_orders'):
                                open_pos_count = len(order_manager.active_orders)
                            
                            heartbeat.beat(
                                status="running",
                                step="trading_cycle",
                                equity=equity,
                                balance=balance,
                                open_positions=open_pos_count,
                                message=f"Cycle #{cycle_count}"
                            )
                            logger.debug("💓 Heartbeat sent to database")
                        except Exception as e:
                            logger.error(f"❌ Heartbeat failed: {e}")
                    
                    last_heartbeat = time.time()

                # Check consecutive failures
                if consecutive_failures >= MAX_FAILURES:
                    logger.error(f"🚨 Too many consecutive failures ({consecutive_failures}/{MAX_FAILURES})")
                    alert_msg = (
                        f"🚨 <b>BOT STOPPED</b>\n"
                        f"Too many consecutive failures: {consecutive_failures}\n"
                        f"Please check logs and restart manually.\n"
                        f"Last error: {str(e)[:100] if 'e' in locals() else 'Unknown'}"
                    )
                    safe_send(alert_msg, parse_mode="HTML")
                    running = False
                    break

                # Calculate sleep time to maintain consistent intervals
                sleep_time = max(0, interval - actual_duration)
                if sleep_time > 0:
                    logger.debug(f"💤 Sleeping for {sleep_time:.2f} seconds before next cycle")
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"⚠️ Cycle took longer than interval ({actual_duration:.2f}s > {interval}s)")

            except KeyboardInterrupt:
                logger.warning("⌨️ Interrupted by user")
                running = False
                break
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"❌ Cycle failed ({consecutive_failures}/{MAX_FAILURES}): {e}")
                logger.error(traceback.format_exc())
                
                # Send alert for cycle failure
                try:
                    safe_send(
                        f"⚠️ <b>Cycle Failed</b>\n"
                        f"Cycle #{cycle_count}\n"
                        f"Error: {str(e)[:100]}",
                        parse_mode="HTML"
                    )
                except:
                    pass
                
                if consecutive_failures >= MAX_FAILURES:
                    logger.error("🚨 Maximum failures reached - stopping bot")
                    safe_send(
                        "🚨 <b>BOT STOPPED</b>\nToo many failures!",
                        parse_mode="HTML"
                    )
                    running = False
                    break
                
                # Sleep before retrying
                time.sleep(5)

        logger.warning("🛑 Main loop terminated")

        # Periodic ML Report (after loop ends or during)
        if ml_manager and hasattr(ml_manager, 'enabled') and ml_manager.enabled:
            logger.info("🧠 ML System is ENABLED — generating performance report...")
            try:
                stats = ml_manager.get_model_stats()
                
                if isinstance(stats, dict):
                    logger.info(
                        f"🤖 ML Stats:\n"
                        f"   Training samples: {stats['training_samples']}\n"
                        f"   Models trained: {stats['models_trained']}\n"
                        f"   Profitable trades: {stats.get('profit_trades', 0)}\n"
                        f"   Loss trades: {stats.get('loss_trades', 0)}"
                    )
                    
                    if stats['training_samples'] > 0:
                        win_rate = stats.get('profit_trades', 0) / stats['training_samples'] * 100
                        logger.info(f"   ML Win Rate: {win_rate:.1f}%")
                    
                    # Send ML report via Telegram
                    try:
                        ml_report = (
                            f"📊 <b>ML System Report</b>\n"
                            f"Training samples: {stats['training_samples']}\n"
                            f"Models trained: {stats['models_trained']}\n"
                            f"Win Rate: {win_rate:.1f}%\n"
                            f"Profitable: {stats.get('profit_trades', 0)}\n"
                            f"Losses: {stats.get('loss_trades', 0)}"
                        )
                        safe_send(ml_report, parse_mode="HTML")
                    except:
                        pass
                        
            except Exception as e:
                logger.error(f"❌ ML stats failed: {e}")
                logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"❌ CRITICAL: Main loop crashed: {e}")
        logger.error(traceback.format_exc())
        
        # Send critical error alert
        try:
            safe_send(
                f"💥 <b>CRITICAL ERROR</b>\n"
                f"Main loop crashed!\n"
                f"Error: {str(e)[:200]}",
                parse_mode="HTML"
            )
        except:
            pass

    finally:
        logger.info("🧹 Cleaning up resources...")
        
        # Close all positions gracefully
        try:
            if order_manager and ENABLE_LIVE_TRADING:
                logger.info("CloseOperation all positions before shutdown...")
                order_manager.close_all_positions(reason="Bot shutdown")
        except Exception as e:
            logger.error(f"❌ Failed to close positions: {e}")
        
        # Send final shutdown notification
        try:
            safe_send("🛑 <b>Bot Stopped</b>\nTrading system shutdown complete.", parse_mode="HTML")
        except:
            pass
            
        logger.info("👋 Bot shutdown complete - Goodbye!")

# ============================================================
# 🚀 ENTRY POINT
# ============================================================
if __name__ == "__main__":
    """
    Entry point for the trading bot with comprehensive error handling and graceful shutdown.
    """
    try:
        logger.info("=" * 70)
        logger.info("🤖 CRYPTO TRADING BOT STARTING...")
        logger.info("=" * 70)
        
        # Set up signal handlers for graceful shutdown
        def shutdown_handler(signum, frame):
            global running
            logger.warning(f"🛑 Received shutdown signal {signum}")
            running = False

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)
        
        # Start main loop
        main_loop()
        
    except KeyboardInterrupt:
        logger.warning("⌨️ Bot interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error in main execution: {e}")
        logger.error(traceback.format_exc())
        
        # Try to send error notification
        try:
            if 'safe_send' in globals():
                safe_send(
                    f"💥 <b>FATAL ERROR</b>\n"
                    f"Bot crashed during startup!\n"
                    f"Error: {str(e)[:200]}",
                    parse_mode="HTML"
                )
        except:
            pass
            
    finally:
        logger.info("✋ Bot execution terminated")
        sys.exit(0)