# order_manager.py
"""
Professional Order Manager for Trading Bot
Handles order placement, position monitoring, and risk management with comprehensive error handling.
Production-ready for live trading environments.
"""
import time
import traceback
import pandas as pd
from typing import Dict, Optional, List, Any, Tuple, TYPE_CHECKING
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from loguru import logger
from datetime import datetime
from sessions import BybitSession
from time_stop_manager import TimeStopManager

# Lazy load these to avoid circular imports
DB_INTEGRATION = False
record_execution = None
close_trade = None

ML_AVAILABLE = False
MLManager = None

ML_API_AVAILABLE = False
send_trade_outcome_to_ml = None

def _load_db_utils():
    """Lazy load db_utils to avoid circular imports."""
    global DB_INTEGRATION, record_execution, close_trade
    if DB_INTEGRATION:
        return True
    try:
        from db_utils import record_execution as _record_execution, close_trade as _close_trade
        record_execution = _record_execution
        close_trade = _close_trade
        DB_INTEGRATION = True
        return True
    except Exception as e:
        logger.warning(f"⚠️ DB integration not available: {e}")
        return False

def _load_ml_manager():
    """Lazy load ml_manager instance to avoid circular imports."""
    global ML_AVAILABLE, MLManager
    if ML_AVAILABLE and MLManager is not None:
        return MLManager
    try:
        # Add parent directory to path for ml_system import
        import sys
        import os
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from ml_system.ml_manager import ml_manager as _ml_manager
        if _ml_manager is not None and getattr(_ml_manager, 'enabled', False):
            MLManager = _ml_manager
            ML_AVAILABLE = True
            logger.info("🤖 ML Manager instance loaded")
            return MLManager
        else:
            logger.info("ℹ️ ML Manager disabled or not initialized")
            return None
    except ImportError as e:
        logger.warning(f"⚠️ MLManager not available: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ MLManager load error: {e}")
        return None

def _load_config_loader():
    """Lazy load config_loader to avoid circular imports."""
    try:
        from config_loader import get_config_loader
        return get_config_loader()
    except ImportError as e:
        logger.error(f"❌ config_loader not available: {e}")
        return None

class OrderManager:

    def __init__(self, session: BybitSession, risk_manager=None, category: str = "linear"):
        """
        Initialize OrderManager with comprehensive error handling.
        """
        self.session = session
        self.client = session
        self.category = category
        self.risk_manager = risk_manager
        self.risk = risk_manager
        self.active_orders: Dict[str, Dict] = {}
        self.closed_trades_today: List[Dict] = []
        self.closed_trades_date = datetime.utcnow().date()
        
        # Momentum
        self.momentum_positions: Dict[str, Dict] = {}
        self.momentum_mode_enabled = False

        # ML system - use global instance
        self.ml_manager = _load_ml_manager()
        if self.ml_manager:
            logger.info("🤖 ML Manager connected successfully")
        else:
            logger.info("ℹ️ ML Manager not available - running without ML")

        # Load config - lazy load
        cfg = _load_config_loader()
        if cfg is None:
            logger.error("❌ Config loader failed - using defaults")
            self._set_default_config()
            return
        
        section = "risk_management"
        self.tp_method: str = cfg.get(section, "TP_METHOD", "single", str).lower()
        tp_ladder_raw = cfg.get(section, "TP_LADDER_PCTS", "", str)
        self.tp_ladder_pcts: List[float] = []
        if tp_ladder_raw:
            try:
                self.tp_ladder_pcts = [
                    float(p.strip())
                    for p in tp_ladder_raw.split(",")
                    if p.strip()
                ]
            except Exception as e:
                logger.error(f"❌ Failed parsing TP_LADDER_PCTS='{tp_ladder_raw}': {e}")
                self.tp_ladder_pcts = []

        # trailing settings
        self.trailing_enabled: bool = cfg.get(section, "TRAILING_STOP_ENABLED", False, bool)
        self.trailing_activation_pct: float = cfg.get(section, 'TRAILING_STOP_ACTIVATION_PCT', 0.3, float)
        self.trailing_distance_pct: float = cfg.get(section,'TRAILING_STOP_DISTANCE_PCT', 0.1, float)
        
        # Partial Take Profit Settings
        self.tp_partial_at_pct = cfg.get(section, "TAKE_PROFIT_PARTIAL_AT_PCT", 80.0, float)
        self.tp_partial_size_pct = cfg.get(section, "TAKE_PROFIT_PARTIAL_SIZE_PCT", 50.0, float)

        logger.info(
        f"TP_METHOD={self.tp_method}, TP_LADDER_PCTS={self.tp_ladder_pcts}, "
        f"TRAILING_ENABLED={self.trailing_enabled}, "
        f"PARTIAL_TP={self.tp_partial_at_pct}% @ {self.tp_partial_size_pct}%")

        # Momentum settings
        momentum_section = "momentum_trading"
        self.momentum_trading_enabled = cfg.get(momentum_section, "ENABLED", False, bool)
        self.min_momentum_pct = cfg.get(momentum_section, "MIN_MOMENTUM_PCT", 20.0, float)
        self.min_volume_usdt = cfg.get(momentum_section, "MIN_VOLUME_USDT", 2000000, float)
        
        # Momentum position management settings (with defaults if missing)
        self.momentum_tp_pct = cfg.get(momentum_section, "MOMENTUM_TP_PCT", 1.5, float)
        self.momentum_sl_pct = cfg.get(momentum_section, "MOMENTUM_SL_PCT", 1.5, float)
        self.momentum_max_time_minutes = cfg.get(momentum_section, "MOMENTUM_MAX_TIME_MINUTES", 45, int)
        self.momentum_trailing_activation = cfg.get(momentum_section, "MOMENTUM_TRAILING_ACTIVATION", 0.5, float)
        self.momentum_trailing_distance = cfg.get(momentum_section, "MOMENTUM_TRAILING_DISTANCE", 0.3, float)
        self.momentum_exit_on_rsi_drop = cfg.get(momentum_section, "MOMENTUM_EXIT_ON_RSI_DROP", True, bool)
        self.momentum_rsi_exit_threshold = cfg.get(momentum_section, "MOMENTUM_RSI_EXIT_THRESHOLD", 50.0, float)
        self.momentum_min_volume_ratio = cfg.get(momentum_section, "MOMENTUM_MIN_VOLUME_RATIO", 0.7, float)

        logger.success("✅ OrderManager initialized")
        logger.info(
            f"TP_METHOD={self.tp_method}, TP_LADDER_PCTS={self.tp_ladder_pcts}, "
            f"TRAILING_ENABLED={self.trailing_enabled}"
        )
        
        if self.momentum_trading_enabled:
            logger.info(
                f"🚀 Momentum Trading ENABLED: "
                f"TP={self.momentum_tp_pct}%, SL={self.momentum_sl_pct}%, "
                f"Max Time={self.momentum_max_time_minutes}min"
            )

    def _set_default_config(self):
        """Set default configuration values when config loading fails."""
        self.tp_method = "single"
        self.tp_ladder_pcts = []
        self.trailing_enabled = False
        self.trailing_activation_pct = 1.0
        self.trailing_distance_pct = 0.5
        self.momentum_trading_enabled = False
        self.min_momentum_pct = 20.0
        self.min_volume_usdt = 2000000
        self.momentum_tp_pct = 1.5
        self.momentum_sl_pct = 1.5
        self.momentum_max_time_minutes = 45
        self.momentum_trailing_activation = 0.5
        self.momentum_trailing_distance = 0.3
        self.momentum_exit_on_rsi_drop = True
        self.momentum_rsi_exit_threshold = 50.0
        self.momentum_min_volume_ratio = 0.7
        logger.warning("⚠️ Using default OrderManager config")

    def _get_symbol_limits_safe(self, symbol: str) -> Dict[str, float]:
        """Get symbol limits safely with comprehensive error handling."""
        try:
            symbol = symbol.upper()
            target = None

            if hasattr(self.client, "get_symbol_limits"):
                target = self.client
            elif hasattr(self.client, "client") and hasattr(self.client.client, "get_symbol_limits"):
                target = self.client.client

            if target is None:
                logger.debug(f"ℹ️ target object has no get_symbol_limits for {symbol}")
                return {}

            limits = target.get_symbol_limits(symbol) or {}
            if not isinstance(limits, dict):
                logger.debug(f"ℹ️ get_symbol_limits for {symbol} returned non-dict: {limits}")
                return {}
            return limits
        except Exception as e:
            logger.debug(f"ℹ️ get_symbol_limits failed for {symbol}: {e}")
            return {}
    
    def _get_current_price_safe(self, symbol: str) -> Optional[Decimal]:
        """
        Get current price safely with comprehensive error handling.
        Attempts multiple sources: internal wrapper and direct SDK client.
        """
        try:
            symbol = symbol.upper()

            # Method 1: Try the high-level wrapper method if available
            if hasattr(self.client, "get_current_price"):
                try:
                    p = self.client.get_current_price(symbol)
                    if p:
                        price_dec = Decimal(str(p))
                        if price_dec > 0:
                            return price_dec
                except Exception as e:
                    logger.debug(f"ℹ️ get_current_price (wrapper) failed for {symbol}: {e}")

            # Method 2: Fallback to direct HTTP ticker request via Bybit SDK
            http = getattr(self.client, "client", self.client)
            if hasattr(http, "get_tickers"):
                try:
                    # Ensure we use the correct category for derivatives/spot
                    cat = getattr(self.client, "category", self.category)
                    res = http.get_tickers(category=cat, symbol=symbol)
                    
                    if isinstance(res, dict) and res.get("retCode") == 0:
                        result_list = res.get("result", {}).get("list", [])
                        if result_list:
                            raw_price = result_list[0].get("lastPrice")
                            if raw_price and float(raw_price) > 0:
                                return Decimal(str(raw_price))
                                
                    logger.debug(f"ℹ️ Ticker response for {symbol} invalid or empty: {res}")
                except Exception as e:
                    logger.debug(f"ℹ️ get_tickers (direct SDK) failed for {symbol}: {e}")

            logger.warning(f"⚠️ Could not retrieve price for {symbol} from any source")
            return None

        except Exception as e:
            logger.error(f"❌ _get_current_price_safe critical error for {symbol}: {e}")
            return None

    def get_symbol_limits(self, symbol: str) -> Dict[str, Any]:
        """
        Public getter for symbol limits with safe fallback.
        Returns exchange filters like tick_size, lot_size, min_notional.
        """
        limits = self._get_symbol_limits_safe(symbol)
        if not limits:
            logger.warning(f"⚠️ No limits found for {symbol}, returning empty dict")
            return {}
        
        return limits

    def _fetch_position_base_price(
        self, symbol: str, fallback_price: Optional[float] = None, side: Optional[str] = None
    ) -> Tuple[Optional[float], Optional[str]]:
        """Fetch position base price with comprehensive error handling."""
        try:
            candidates = []

            # אפשרויות wrapper
            if hasattr(self.client, "get_position"):
                candidates.append(("wrapper.get_position", self.client.get_position))
            if hasattr(self.client, "get_positions"):
                candidates.append(("wrapper.get_positions", self.client.get_positions))

            target_side = (side or "").upper() if side else None

            for name, func in candidates:
                try:
                    try:
                        res = func(category=self.category, symbol=symbol)
                    except TypeError:
                        res = func(symbol=symbol)
                except Exception as e:
                    logger.debug(f"ℹ️ {name} failed for {symbol}: {e}")
                    continue

                # Parse response
                try:
                    data = res
                    positions = []

                    if isinstance(data, dict):
                        if "result" in data and isinstance(data["result"], dict):
                            data = data["result"]
                        if "list" in data and isinstance(data["list"], list):
                            positions = data["list"]
                        else:
                            positions = [data]
                    else:
                        continue

                    for pos in positions:
                        pos_side = (
                            pos.get("side") 
                            or pos.get("positionSide") 
                            or pos.get("position_side")
                        )
                        norm_pos_side = str(pos_side).upper() if pos_side else None

                        if target_side and norm_pos_side and norm_pos_side != target_side:
                            continue

                        price_str = (
                            pos.get("avgPrice")
                            or pos.get("basePrice")
                            or pos.get("entryPrice")
                            or pos.get("entry_price")
                        )
                        if price_str is None:
                            continue

                        try:
                            base_price = float(price_str)
                        except Exception:
                            continue

                        logger.debug(
                            f"📌 base_price for {symbol}: {base_price} "
                            f"(side={norm_pos_side}) via {name}"
                        )
                        return base_price, (target_side or norm_pos_side)

                except Exception as e:
                    logger.debug(f"ℹ️ failed parsing position response from {name}: {e}")
                    continue

            logger.warning(f"⚠️ Could not fetch base price for {symbol}, fallback used: {fallback_price}")
            return fallback_price, side
        except Exception as e:
            logger.error(f"❌ _fetch_position_base_price failed for {symbol}: {e}")
            return fallback_price, side

    @staticmethod
    def _normalize_stops_for_side(side: str,ref_price: float,sl: Optional[float],tp: Optional[float],tick_size: float,) -> Tuple[Optional[float], Optional[float]]:
        """Normalize stops for side with comprehensive error handling."""
        try:
            if tick_size <= 0:
                tick_size = 0.0001

            side_up = (side or "").upper()

            if sl is None and tp is None:
                return sl, tp

            sl_local = sl
            tp_local = tp

            if side_up == "BUY":
                if sl_local is not None and sl_local >= ref_price:
                    sl_local = ref_price * 0.99
                if tp_local is not None and tp_local <= ref_price:
                    tp_local = ref_price * 1.01

            elif side_up == "SELL":
                if sl_local is not None and sl_local <= ref_price:
                    sl_local = ref_price * 1.01
                if tp_local is not None and tp_local >= ref_price:
                    tp_local = ref_price * 0.99

            if sl_local is not None:
                sl_local = round(sl_local / tick_size) * tick_size
            if tp_local is not None:
                tp_local = round(tp_local / tick_size) * tick_size

            return sl_local, tp_local
        except Exception as e:
            logger.error(f"❌ _normalize_stops_for_side failed: {e}")
            return sl, tp

    def _sanitize_quantity(self, symbol: str, qty: float) -> float:
        """
        Sanitize quantity with comprehensive error handling.
        Fixed: Using ROUND_DOWN to prevent 'Insufficient Margin' and improved limit checks.
        """
        try:
            symbol = symbol.upper()
            if qty <= 0:
                return 0.0

            # 1. שליפת הגבלות מהבורסה
            limits = self._get_symbol_limits_safe(symbol)
            # שימוש ב-get עם ערכי ברירת מחדל כדי למנוע קריסות
            min_notional = Decimal(str(limits.get("min_notional", 0) or 0))
            min_qty = Decimal(str(limits.get("min_qty", 0) or 0))
            qty_step = Decimal(str(limits.get("qty_step", 0) or 0))
            
            qty_dec = Decimal(str(qty))

            if qty_step > 0:
                steps = (qty_dec / qty_step).to_integral_value(rounding=ROUND_DOWN)
                qty_dec = steps * qty_step

            # Minimum Quantity
            if min_qty > 0 and qty_dec < min_qty:
                logger.warning(f"🔢 {symbol} qty {qty_dec} below min_qty {min_qty}. Skipping trade.")
                return 0.0

            price_dec = self._get_current_price_safe(symbol)
            if price_dec is not None and min_notional > 0:
                notional = qty_dec * price_dec

                if notional < min_notional:
                    effective_min_qty = (min_notional * Decimal("1.005") / price_dec)

                    if qty_step > 0:
                        steps = (effective_min_qty / qty_step).to_integral_value(rounding=ROUND_UP)
                        qty_dec = steps * qty_step
                    
                    if qty_dec > Decimal(str(qty)) * Decimal("1.2"):
                        logger.warning(f"🔢 {symbol} bumping to min_notional requires too much extra margin. Skipping.")
                        return 0.0

            if qty_dec <= 0:
                return 0.0

            final = float(qty_dec)
            logger.debug(f"🔢 Sanitized {symbol}: {qty} -> {final} (Step: {qty_step})")
            return final

        except Exception as e:
            logger.error(f"❌ _sanitize_quantity critical failure for {symbol}: {e}")
            return 0.0

    def _calculate_momentum_stops(self, symbol: str, side: str, entry_price: float) -> Tuple[float, float]:
        """Calculate momentum stops with comprehensive error handling."""
        try:
            side_up = side.upper()
            
            tp_pct = self.momentum_tp_pct / 100
            sl_pct = self.momentum_sl_pct / 100
            
            if side_up == "BUY":
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)
            else:
                tp_price = entry_price * (1 - tp_pct)
                sl_price = entry_price * (1 + sl_pct)
            
            logger.info(
                f"🚀 Momentum stops for {symbol} {side}: "
                f"Entry={entry_price:.8f}, "
                f"TP={tp_price:.8f} (+{self.momentum_tp_pct}%), "
                f"SL={sl_price:.8f} (-{self.momentum_sl_pct}%)"
            )
            
            return sl_price, tp_price
        except Exception as e:
            logger.error(f"❌ _calculate_momentum_stops failed for {symbol}: {e}")
            # Return default stops
            if side.upper() == "BUY":
                return entry_price * 0.98, entry_price * 1.05
            else:
                return entry_price * 1.02, entry_price * 0.95

    def _should_exit_momentum_position(self, symbol: str, position_data: Dict, current_price: float, indicators: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Check if should exit momentum position with comprehensive error handling.
        Fixed: Correct trailing stop logic for both Long and Short positions.
        """
        try:
            exit_reasons = []
            side = position_data.get('side', '').upper()
            is_long = side == 'BUY'
            entry_price = float(position_data.get('entry_price', current_price))
            
            # 1. בדיקת זמן (Time-based Exit)
            opened_at = position_data.get('opened_at', time.time())
            age_minutes = (time.time() - opened_at) / 60
            if age_minutes > self.momentum_max_time_minutes:
                exit_reasons.append(f"Time: {age_minutes:.1f}min > {self.momentum_max_time_minutes}min")
            
            # 2. בדיקת אינדיקטורים (RSI & Volume)
            if indicators and self.momentum_exit_on_rsi_drop:
                rsi = indicators.get('rsi')
                if rsi:

                    if is_long and rsi < self.momentum_rsi_exit_threshold:
                        exit_reasons.append(f"RSI Low: {rsi:.1f} < {self.momentum_rsi_exit_threshold}")
                    elif not is_long and rsi > (100 - self.momentum_rsi_exit_threshold):
                        exit_reasons.append(f"RSI High: {rsi:.1f} > {100 - self.momentum_rsi_exit_threshold}")
            
            if indicators and self.momentum_min_volume_ratio > 0:
                volume_ratio = indicators.get('volume_ratio')
                if volume_ratio and volume_ratio < self.momentum_min_volume_ratio:
                    exit_reasons.append(f"Vol Ratio: {volume_ratio:.1f}x < {self.momentum_min_volume_ratio}x")

            extreme_price = position_data.get('extreme_price')
            
            if extreme_price is None:
                extreme_price = current_price
            else:
                extreme_price = float(extreme_price)

            if is_long:
                if current_price > extreme_price:
                    extreme_price = current_price
                    position_data['extreme_price'] = extreme_price
            else: # Short
                if current_price < extreme_price:
                    extreme_price = current_price
                    position_data['extreme_price'] = extreme_price
            
            # 4. חישוב רווח נוכחי באחוזים
            if is_long:
                profit_pct = (current_price / entry_price - 1) * 100
            else:
                profit_pct = (entry_price / current_price - 1) * 100
            
            # 5. בדיקת Trailing Stop
            trail_activation_pct = position_data.get('trailing_activation_pct', self.momentum_trailing_activation)
            trail_distance_pct = position_data.get('trailing_distance_pct', self.momentum_trailing_distance)
            
            if profit_pct >= trail_activation_pct:
                if is_long:
                    # בלונג: הסטופ עולה מתחת לשיא
                    trailing_stop = extreme_price * (1 - trail_distance_pct / 100)
                    if current_price <= trailing_stop:
                        exit_reasons.append(f"Trailing (Long): {current_price:.6f} <= {trailing_stop:.6f}")
                else:
                    # בשורט: הסטופ יורד מעל השפל
                    trailing_stop = extreme_price * (1 + trail_distance_pct / 100)
                    if current_price >= trailing_stop:
                        exit_reasons.append(f"Trailing (Short): {current_price:.6f} >= {trailing_stop:.6f}")
            
            # החזרת תוצאה
            if exit_reasons:
                return True, " | ".join(exit_reasons)
            
            return False, ""

        except Exception as e:
            logger.error(f"❌ _should_exit_momentum_position failed for {symbol}: {e}")
            return False, ""

    def _set_position_stops(self,symbol: str,stop_loss: Optional[float],take_profit: Optional[float],side: Optional[str] = None,
        entry_price: Optional[float] = None,base_price: Optional[float] = None,max_retries: int = 3,) -> bool:
        """Set position stop loss and take profit on exchange with comprehensive error handling."""
        try:
            if stop_loss is None and take_profit is None:
                return False

            symbol = symbol.upper()
            limits = self._get_symbol_limits_safe(symbol)
            tick_size = float(limits.get("tick_size") or 0.0001)

            # Determine reference price
            ref_price = base_price or entry_price
            if ref_price is None:
                price_dec = self._get_current_price_safe(symbol)
                if price_dec is None:
                    logger.warning(f"⚠️ Cannot determine reference price for SL/TP on {symbol}")
                    return False
                ref_price = float(price_dec)

            side_clean = (side or "").capitalize()
            if side_clean not in ("Buy", "Sell"):
                logger.warning(f"⚠️ Invalid side '{side}' for SL/TP, defaulting to Buy")
                side_clean = "Buy"

            def round_to_tick(price: float, tick: float) -> float:
                """Round price to tick size precision."""
                if tick <= 0:
                    tick = 0.0001
                return round(round(price / tick) * tick, 8)

            last_error = None

            for attempt in range(1, max_retries + 1):
                sl_value = stop_loss
                tp_value = take_profit

                if attempt == 1:
                    sl_value, tp_value = self._normalize_stops_for_side(
                        side_clean, ref_price, sl_value, tp_value, tick_size
                    )
                else:
                    logger.warning(
                        f"⚠️ {symbol} SL/TP attempt #{attempt} using fallback around ref_price={ref_price:.6f}"
                    )
                    if side_clean == "Buy":
                        if stop_loss is not None:
                            sl_value = ref_price * (1 - 0.01 * attempt)
                        if take_profit is not None:
                            tp_value = ref_price * (1 + 0.01 * attempt)
                    else:
                        if stop_loss is not None:
                            sl_value = ref_price * (1 + 0.01 * attempt)
                        if take_profit is not None:
                            tp_value = ref_price * (1 - 0.01 * attempt)

                if sl_value is not None:
                    sl_value = round_to_tick(sl_value, tick_size)
                if tp_value is not None:
                    tp_value = round_to_tick(tp_value, tick_size)

                logger.debug(
                    f"[Attempt {attempt}] {symbol} SL={sl_value}, TP={tp_value}, "
                    f"tick_size={tick_size}, side={side_clean}"
                )

                try:
                    res = self.session.set_trading_stop(
                        symbol=symbol,
                        side=side_clean,
                        stop_loss=sl_value,
                        take_profit=tp_value,
                    )

                    status = res.get("status", "").upper()
                    
                    # ═══════════════════════════════════════════════════════════
                    # SUCCESS cases
                    # ═══════════════════════════════════════════════════════════
                    if status == "OK" or status == "SUCCESS":
                        if res.get("already_set"):
                            logger.info(f"ℹ️ SL/TP already set for {symbol} - OK")
                        else:
                            logger.success(f"🛡️  SL/TP set for {symbol}: SL={sl_value}, TP={tp_value}")
                        return True
                    
                    if status == "SKIP":
                        logger.debug(f"ℹ️ SL/TP skipped for {symbol} (no values)")
                        return True

                    # ═══════════════════════════════════════════════════════════
                    # ERROR handling
                    # ═══════════════════════════════════════════════════════════
                    last_error = res
                    error_msg = res.get("error") or res.get("retMsg") or str(res)
                    ret_code = res.get("retCode") or res.get("raw", {}).get("retCode")
                    
                    # Check for "not modified" error (34040) - treat as success
                    if ret_code == 34040 or "not modified" in str(error_msg).lower():
                        logger.info(f"ℹ️ SL/TP already set for {symbol} (error 34040) - OK")
                        return True
                    
                    logger.error(f"❌ SL/TP failed (Attempt {attempt}): {error_msg}")

                    # Retryable errors
                    if ret_code in [10001, 10002, 110017]:
                        if attempt < max_retries:
                            time.sleep(0.5)
                            continue
                    
                    if attempt >= max_retries:
                        break

                except Exception as e:
                    error_str = str(e)
                    
                    # ═══════════════════════════════════════════════════════════
                    # Handle "not modified" exception - treat as success!
                    # ═══════════════════════════════════════════════════════════
                    if "34040" in error_str or "not modified" in error_str.lower():
                        logger.info(f"ℹ️ SL/TP already set for {symbol} (exception 34040) - OK")
                        return True
                    
                    last_error = error_str
                    logger.error(f"❌ SL/TP exception (Attempt {attempt}): {e}")
                    
                    if attempt < max_retries:
                        time.sleep(0.5)
                        continue

            logger.error(f"❌ Giving up on SL/TP for {symbol} after {max_retries} attempts: {last_error}")
            return False
        except Exception as e:
            logger.error(f"❌ _set_position_stops failed for {symbol}: {e}")
            return False

    def _build_tp_ladder(self, side: str, entry_price: float) -> List[Dict]:
        """Build take-profit ladder levels with comprehensive error handling."""
        try:
            # Return empty if ladder is not used
            if self.tp_method != "ladder" or not self.tp_ladder_pcts:
                return []

            levels: List[Dict] = []
            pcts_sorted = sorted(self.tp_ladder_pcts)
            n = len(pcts_sorted)
            if n == 0:
                return []

            base_fraction = 1.0 / n
            remaining = 1.0
            side_up = side.upper()

            for i, pct in enumerate(pcts_sorted):
                # Calculate target price for each ladder level
                target = entry_price * (1.0 + pct / 100.0) if side_up == "BUY" else entry_price * (1.0 - pct / 100.0)
                fraction = base_fraction if i < n - 1 else remaining
                remaining -= fraction
                levels.append({"target": target, "fraction": fraction, "hit": False, "label": f"{pct}%"})

            logger.info(f"📐 TP ladder built (side={side_up}, entry={entry_price}): {[(lvl['label'], lvl['target']) for lvl in levels]}")
            return levels
        except Exception as e:
            logger.error(f"❌ _build_tp_ladder failed: {e}")
            return []

    def place_order_with_protection(self, symbol: str, side: str, quantity: float, entry_price: float, stop_loss: float, take_profit: float,
        signal_strength: float = 0, equity: Optional[float] = None, is_momentum_trade: bool = False, signal_data: Optional[dict] = None) -> Optional[Dict]:
        """Place an order with stop loss and take profit protection with comprehensive error handling."""
        try:
            symbol = symbol.upper()
            quantity = self._sanitize_quantity(symbol, float(quantity))
            if quantity <= 0:
                logger.error(f"🚫 Sanitized qty is 0 for {symbol}")
                return None

            entry_price = float(entry_price)
            stop_loss = float(stop_loss)
            take_profit = float(take_profit)
            
            # 1. Price Deviation Check (Slippage Protection)
            current_market_price = self._get_current_price(symbol)
            if current_market_price:
                deviation = abs(current_market_price - entry_price) / entry_price
                
                try:
                    if hasattr(self, 'risk') and hasattr(self.risk, 'config'):
                        max_dev_pct = float(self.risk.config.get('trading', 'MAX_PRICE_DEVIATION_PCT', 2.5))
                    else:
                        max_dev_pct = 2.5
                except Exception:
                    max_dev_pct = 2.5
                
                max_dev_threshold = max_dev_pct / 100.0
                
                if deviation > max_dev_threshold:
                    logger.warning(f"🚫 {symbol} price moved too far! Signal: {entry_price}, Market: {current_market_price} (Dev: {deviation:.2%}, Max: {max_dev_threshold:.2%})")
                    return None

            # ═══════════════════════════════════════════════════════════
            # ML PREDICTION & FILTERING
            # ═══════════════════════════════════════════════════════════
            ml_adjusted = False
            ml_analysis = None
            
            if self.ml_manager and not is_momentum_trade:
                try:
                    # Get kline data for ML
                    get_kline = getattr(self.client, "get_kline", None)
                    if callable(get_kline):
                        kl = get_kline(
                            category="linear",
                            symbol=symbol,
                            interval="15",
                            limit=100
                        )
                        
                        if kl.get("retCode") == 0:
                            rows = kl.get("result", {}).get("list", [])
                            if rows:
                                df_ml = pd.DataFrame(rows, columns=[
                                    "open_time", "open", "high", "low",
                                    "close", "volume", "turnover"
                                ])
                                
                                for col in ["open", "high", "low", "close", "volume"]:
                                    df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce')
                                
                                ml_signal = signal_data or {
                                    'symbol': symbol,
                                    'side': side,
                                    'price': entry_price,
                                    'stop_loss': stop_loss,
                                    'take_profit': take_profit,
                                    'strength': signal_strength,
                                }
                                
                                # A. Trade Quality Analysis (Gatekeeper)
                                ml_analysis = self.ml_manager.analyze_trade(ml_signal, df_ml)
                                if ml_analysis:
                                    recommendation = ml_analysis.get('recommendation', 'TAKE')
                                    quality = ml_analysis.get('quality', 'UNKNOWN')
                                    confidence = ml_analysis.get('confidence', 0)
                                    
                                    logger.info(f"🤖 ML Analysis for {symbol}: Quality={quality}, Conf={confidence:.1%}, Rec={recommendation}")
                                    
                                    # ML Execution Filter (configurable)
                                    enable_ml_execution_filter = getattr(self.ml_manager, 'enable_execution_filter', True)

                                    if recommendation == 'SKIP' and enable_ml_execution_filter:
                                        logger.warning(f"🛑 ML Filter: Blocking {symbol} trade due to SKIP recommendation (filter enabled)")
                                        return None
                                    elif recommendation == 'SKIP':
                                        logger.info(f"ℹ️ ML recommended SKIP for {symbol}, but execution filter is disabled — proceeding anyway")
                                
                                # B. ML Stops Adjustment
                                ml_stops = self.ml_manager.predict_stops(
                                    signal=ml_signal,
                                    df=df_ml,
                                    entry_price=entry_price
                                )
                                
                                if ml_stops:
                                    ml_sl, ml_tp = ml_stops
                                    logger.info(
                                        f"🤖 ML adjusted stops for {symbol}:\n"
                                        f"   SL: {stop_loss:.8f} → {ml_sl:.8f}\n"
                                        f"   TP: {take_profit:.8f} → {ml_tp:.8f}"
                                    )
                                    stop_loss = ml_sl
                                    take_profit = ml_tp
                                    ml_adjusted = True
                                    
                except Exception as e:
                    logger.debug(f"⚠️ ML prediction/filtering failed for {symbol}: {e}")

            # ═══════════════════════════════════════════════════════════
            # Handle momentum trades
            # ═══════════════════════════════════════════════════════════
            if is_momentum_trade:
                logger.info(f"🚀 Opening MOMENTUM trade for {symbol}")
                stop_loss, take_profit = self._calculate_momentum_stops(symbol, side, entry_price)
                self.momentum_positions[symbol] = {
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "opened_at": time.time(),
                    "highest_price": entry_price,
                    "trailing_activation_pct": self.momentum_trailing_activation,
                    "trailing_distance_pct": self.momentum_trailing_distance,
                    "is_momentum": True,
                }

            # ═══════════════════════════════════════════════════════════
            # ORDER EXECUTION
            # ═══════════════════════════════════════════════════════════
            logger.info(f"📤 Opening {side} {symbol} qty={quantity}")
            order = self.session.place_market_order(symbol=symbol, side=side, qty=quantity)
            if not order or order.get("status") == "ERROR":
                logger.error(f"❌ Order failed: {order}")
                return None

            order_id = order.get("orderId")
            actual_price = float(order.get("price", entry_price))
            logger.success(f"✅ Order Placed: {order_id} @ {actual_price}")
            size_usdt = quantity * actual_price

            # Wait for position to register
            logger.debug("⏳ Waiting for position to register before SL/TP...")
            time.sleep(2.0)
            base_price, position_confirmed = actual_price, False

            for attempt in range(3):
                try:
                    fetched_price, _ = self._fetch_position_base_price(symbol, fallback_price=actual_price, side=side)
                    if fetched_price is not None:
                        base_price = fetched_price
                        position_confirmed = True
                        logger.success(f"✅ Position confirmed: {symbol} @ {base_price} (Attempt {attempt + 1}/3)")
                        break
                except Exception as e:
                    logger.debug(f"⚠️ Position fetch attempt {attempt + 1} failed: {e}")
                time.sleep(1.0)

            if not position_confirmed:
                logger.warning(f"⚠️ Could not confirm position for {symbol}, using actual_price={actual_price} as fallback")

            # ═══════════════════════════════════════════════════════════
            # Recalculate SL/TP based on confirmed base price
            # ═══════════════════════════════════════════════════════════
            side_up = side.upper()
            sl_dist_pct = 0.0
            tp_dist_pct = 0.0

            # Calculate original intended percentage distances from signal entry_price
            if side_up == "BUY":
                sl_dist_pct = (entry_price - stop_loss) / entry_price if stop_loss else 0
                tp_dist_pct = (take_profit - entry_price) / entry_price if take_profit else 0
            else:  # SELL
                sl_dist_pct = (stop_loss - entry_price) / entry_price if stop_loss else 0
                tp_dist_pct = (entry_price - take_profit) / entry_price if take_profit else 0

            # Apply those percentages to the actual execution price (base_price)
            if sl_dist_pct > 0:
                stop_loss = base_price * (1.0 + sl_dist_pct) if side_up == "SELL" else base_price * (1.0 - sl_dist_pct)
            if tp_dist_pct > 0:
                take_profit = base_price * (1.0 - tp_dist_pct) if side_up == "SELL" else base_price * (1.0 + tp_dist_pct)

            logger.info(f"📐 Final stops for {symbol} applied to {base_price:.8f}: SL={stop_loss:.8f}, TP={take_profit:.8f}")

            effective_side = side
            
            # TP ladder handling
            if is_momentum_trade:
                tp_levels = []
                tp_for_exchange = take_profit
            else:
                tp_levels = self._build_tp_ladder(effective_side, base_price)
                
                if take_profit and self.tp_partial_at_pct > 0 and self.tp_partial_size_pct > 0:
                    partial_target_pct = self.tp_partial_at_pct / 100.0
                    partial_size_fraction = self.tp_partial_size_pct / 100.0
                    
                    if side_up == "BUY":
                        partial_price = base_price * (1.0 + (take_profit / base_price - 1.0) * partial_target_pct)
                    else:  # SELL
                        partial_price = base_price * (1.0 - (base_price / take_profit - 1.0) * partial_target_pct)
                    
                    partial_level = {
                        "target": partial_price,
                        "fraction": partial_size_fraction,
                        "hit": False,
                        "label": f"Partial {self.tp_partial_at_pct}%"
                    }
                    
                    if tp_levels:
                        tp_levels.insert(-1, partial_level)
                    else:
                        tp_levels.append(partial_level)
                        tp_levels.append({"target": take_profit, "fraction": 1.0 - partial_size_fraction, "hit": False, "label": "Final TP"})
                    
                    logger.info(f"🎯 Added Partial TP: {partial_price:.8f}")

                tp_for_exchange = tp_levels[-1]["target"] if tp_levels else take_profit

            # Set SL/TP on Exchange
            sltp_ok = self._set_position_stops(
                symbol, stop_loss, tp_for_exchange, 
                side=effective_side, entry_price=actual_price, base_price=base_price
            )

            # Risk tracking
            if self.risk:
                self.risk.track_open(order_id, symbol, quantity, side, actual_price, stop_loss, tp_for_exchange)

            # Active orders tracking
            trail_state = {
                "enabled": self.momentum_trailing_activation if is_momentum_trade else self.trailing_enabled,
                "activation_pct": self.momentum_trailing_activation if is_momentum_trade else self.trailing_activation_pct,
                "distance_pct": self.momentum_trailing_distance if is_momentum_trade else self.trailing_distance_pct,
                "active": False,
                "reference_price": actual_price,
            }

            self.active_orders[order_id] = {
                "symbol": symbol, "side": side, "initial_qty": quantity, "remaining_qty": quantity,
                "entry_price": actual_price, "stop_loss": stop_loss, "take_profit": tp_for_exchange,
                "tp_levels": tp_levels, "trail": trail_state, "opened_at": time.time(),
                "signal_strength": signal_strength, "realized_pnl": 0.0, "trade_id": None,
                "sltp_attempts": 1, "sltp_ok": sltp_ok, "is_momentum": is_momentum_trade,
                "signal_data": signal_data, "ml_adjusted": ml_adjusted, "ml_analysis": ml_analysis,
            }

            ml_tag = " 🤖" if ml_adjusted else ""
            logger.success(
                f"🎉 Order complete: {symbol} {side}{ml_tag} | Entry: {actual_price:.8f} | "
                f"SL: {stop_loss:.8f} | TP: {tp_for_exchange:.8f} | "
                f"{'🚀 MOMENTUM' if is_momentum_trade else '📊 REGULAR'}"
            )

            return {
                "status": "SUCCESS", "order_id": order_id, "price": actual_price,
                "sl": sltp_ok, "tp": sltp_ok, "is_momentum": is_momentum_trade, "ml_adjusted": ml_adjusted,
            }

        except Exception as e:
            logger.error(f"❌ place_order_with_protection failed: {e}")
            logger.error(traceback.format_exc())
            return None
        
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from wrapper or ticker endpoint with comprehensive error handling."""
        try:
            if hasattr(self.client, "get_current_price"):
                return self.client.get_current_price(symbol)
            ticker = self.client.get_tickers(category="linear", symbol=symbol)
            return float(ticker["result"]["list"][0]["lastPrice"])
        except Exception:
            return None

    @staticmethod
    def _calc_pnl(side: str, entry: float, exit_price: float, qty: float) -> float:
        """Calculate PnL with comprehensive error handling."""
        try:
            side_up = side.upper()
            if side_up == "BUY":
                return (exit_price - entry) * qty
            else:
                return (entry - exit_price) * qty
        except Exception as e:
            logger.error(f"❌ _calc_pnl failed: {e}")
            return 0.0

    def _update_trailing(self, order_id: str, pos: Dict, current_price: float) -> None:
        """
        Enhanced trailing stop that tracks BOTH loss protection AND profit locking.
        Activates when position reaches TRAILING_STOP_ACTIVATION_PCT profit.
        Locks in profits by trailing below highest price since activation.
        """
        try:
            symbol = pos.get("symbol", "")
            side = str(pos.get("side", "BUY")).upper()
            entry_price = float(pos.get("entry_price", 0.0))
            
            if entry_price <= 0:
                return
                
            # Calculate PnL percentage
            if side == "BUY":
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # SELL
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Initialize or get trailing data
            if "trailing_high" not in pos:
                pos["trailing_high"] = None  # Highest price since activation for longs
                pos["trailing_low"] = None   # Lowest price since activation for shorts
                pos["trailing_activated"] = False
            
            # Check if we should activate trailing (based on profit)
            if not pos["trailing_activated"] and pnl_pct >= self.trailing_activation_pct:
                pos["trailing_activated"] = True
                if side == "BUY":
                    pos["trailing_high"] = current_price
                else:
                    pos["trailing_low"] = current_price
                logger.info(f"🎯 {symbol} Trailing ACTIVATED at {pnl_pct:+.2f}% profit")

            # Update trailing levels if activated
            if pos["trailing_activated"]:
                if side == "BUY":
                    if pos["trailing_high"] is None or current_price > pos["trailing_high"]:
                        pos["trailing_high"] = current_price
                        logger.debug(f"📈 {symbol} New high: {current_price:.6f}")
                    
                    # Calculate trailing stop (distance below high)
                    trailing_stop = pos["trailing_high"] * (1 - self.trailing_distance_pct / 100)
                    
                    # Check if price hit trailing stop
                    if current_price <= trailing_stop:
                        logger.warning(
                            f"🚨 {symbol} EXIT TRIGGERED by Trailing Take Profit: "
                            f"{current_price:.6f} ≤ {trailing_stop:.6f} (Peak: {pos['trailing_high']:.6f})"
                        )
                        self._handle_trailing_exit(order_id, pos, current_price, "PROFIT_TRAIL_HIT")
                        return
                        
                else:  # SELL position
                    if pos["trailing_low"] is None or current_price < pos["trailing_low"]:
                        pos["trailing_low"] = current_price
                        logger.debug(f"📉 {symbol} New low: {current_price:.6f}")
                    
                    # Calculate trailing stop (distance above low)
                    trailing_stop = pos["trailing_low"] * (1 + self.trailing_distance_pct / 100)
                    
                    # Check if price hit trailing stop
                    if current_price >= trailing_stop:
                        logger.warning(
                            f"🚨 {symbol} EXIT TRIGGERED by Trailing Take Profit: "
                            f"{current_price:.6f} ≥ {trailing_stop:.6f} (Trough: {pos['trailing_low']:.6f})"
                        )
                        self._handle_trailing_exit(order_id, pos, current_price, "PROFIT_TRAIL_HIT")
                        return
                        
        except Exception as e:
            logger.error(f"❌ _update_trailing failed for {symbol}: {e}")
            logger.error(traceback.format_exc())

    def _handle_trailing_exit(self, order_id: str, pos: Dict, exit_price: float, reason: str) -> None:
        """Handle closing position due to trailing stop trigger."""
        try:
            symbol = pos.get("symbol", "")
            remaining_qty = float(pos.get("remaining_qty", pos.get("initial_qty", 0.0)))
            qty = self._sanitize_quantity(symbol, remaining_qty)
            
            if qty <= 0:
                logger.warning(f"⚠️ No quantity to close for {symbol}")
                return
                
            side_str = str(pos.get("side", "BUY")).upper()
            close_side = "Sell" if side_str == "BUY" else "Buy"
            
            try:
                result = self.session.place_market_order(
                    symbol=symbol, side=close_side, qty=qty
                )
                
                if result and result.get("status") != "ERROR":
                    actual_exit = float(result.get("price", exit_price))
                    self._finalize_position(
                        order_id, pos, actual_exit, reason=reason
                    )
                    logger.success(f"✅ {symbol} closed successfully due to {reason}")
                else:
                    logger.error(f"❌ Failed to close {symbol} via trailing: {result}")
                    
            except Exception as e:
                logger.error(f"❌ Exception in _handle_trailing_exit for {symbol}: {e}")
                logger.error(traceback.format_exc())
                
        except Exception as e:
            logger.error(f"❌ _handle_trailing_exit failed: {e}")
            logger.error(traceback.format_exc())

    def _check_and_handle_tp(self, order_id: str, pos: Dict, price: float) -> None:
        """Check and handle take profit levels with comprehensive error handling."""
        try:
            if pos.get("is_momentum"):
                return
                
            tp_levels = pos.get("tp_levels") or []
            if not tp_levels:
                return

            side = pos["side"].upper()
            entry = pos["entry_price"]

            for level in tp_levels:
                if level.get("hit"):
                    continue

                target = level["target"]
                fraction = float(level["fraction"])

                hit = False
                if side == "BUY" and price >= target:
                    hit = True
                elif side == "SELL" and price <= target:
                    hit = True

                if not hit:
                    continue

                remaining_qty = float(pos.get("remaining_qty", pos["initial_qty"]))
                if remaining_qty <= 0:
                    level["hit"] = True
                    continue

                qty_to_close = self._sanitize_quantity(
                    pos["symbol"], remaining_qty * fraction
                )
                if qty_to_close <= 0:
                    logger.warning(
                        f"⚠️ TP level size too small for {pos['symbol']} (fraction={fraction})"
                    )
                    level["hit"] = True
                    continue

                logger.info(
                    f"🎯 TP hit for {pos['symbol']} {side}: target={target:.8f}, closing ~{fraction*100:.1f}% "
                    f"(qty={qty_to_close})"
                )

                close_side = "Sell" if side == "BUY" else "Buy"
                result = self.session.place_market_order(
                    symbol=pos["symbol"], side=close_side, qty=qty_to_close
                )

                if not result or result.get("status") == "ERROR":
                    logger.error(
                        f"❌ Failed to close partial TP for {pos['symbol']}: {result}"
                    )
                    continue

                exit_price = float(result.get("price", price))
                pnl_partial = self._calc_pnl(side, entry, exit_price, qty_to_close)
                pos["realized_pnl"] = pos.get("realized_pnl", 0.0) + pnl_partial
                pos["remaining_qty"] = max(0.0, remaining_qty - qty_to_close)
                pos["last_exit_price"] = exit_price
                level["hit"] = True

                logger.success(
                    f"🧾 Partial TP on {pos['symbol']}: qty={qty_to_close}, "
                    f"exit={exit_price:.8f}, pnl={pnl_partial:.2f}, "
                    f"remaining={pos['remaining_qty']:.6f}"
                )

                if pos["remaining_qty"] <= 0:
                    self._finalize_position(order_id, pos, exit_price, reason="TP ladder")
                    break

            pos["tp_levels"] = tp_levels
        except Exception as e:
            logger.error(f"❌ _check_and_handle_tp failed for {order_id}: {e}")

    def _check_and_handle_stop(self, order_id: str, pos: Dict, price: float) -> bool:
        """
        Check and handle stop loss for a position with comprehensive error handling.
        """
        try:
            # Validate inputs
            if not isinstance(order_id, str) or not order_id:
                logger.error("❌ Invalid order_id for _check_and_handle_stop")
                return False
                
            if not isinstance(pos, dict):
                logger.error("❌ Invalid position data for _check_and_handle_stop")
                return False
                
            if not isinstance(price, (int, float)) or price <= 0:
                logger.error("❌ Invalid price for _check_and_handle_stop")
                return False

            sl = pos.get("stop_loss")
            if sl is None:
                return False

            side = str(pos.get("side", "BUY")).upper()
            if side not in ["BUY", "SELL"]:
                logger.error(f"❌ Invalid side '{side}' for _check_and_handle_stop")
                return False

            # Check if stop loss is hit
            hit = False
            if side == "BUY" and price <= sl:
                hit = True
            elif side == "SELL" and price >= sl:
                hit = True

            if not hit:
                return False

            # Get remaining quantity
            try:
                remaining_qty = float(pos.get("remaining_qty", pos.get("initial_qty", 0.0)))
                if remaining_qty <= 0:
                    logger.warning(f"⚠️ No remaining quantity to close for {pos.get('symbol', 'Unknown')}")
                    self._finalize_position(order_id, pos, price, reason="SL (zero qty)")
                    return True
            except (ValueError, TypeError):
                logger.error(f"❌ Invalid quantity for {pos.get('symbol', 'Unknown')}")
                return False

            # Sanitize quantity
            symbol = pos.get("symbol", "Unknown")
            qty_to_close = self._sanitize_quantity(symbol, remaining_qty)
            
            if qty_to_close <= 0:
                logger.warning(
                    f"⚠️ stop-loss triggered but sanitized qty_to_close=0 for {symbol}"
                )
                self._finalize_position(order_id, pos, price, reason="SL (zero qty after sanitize)")
                return True

            logger.error(
                f"🛑 STOP LOSS hit for {symbol} {side}: "
                f"price={price:.8f}, SL={sl:.8f}, closing full position"
            )

            # Determine close side
            close_side = "Sell" if side == "BUY" else "Buy"

            # Place market order with error handling
            try:
                result = self.session.place_market_order(
                    symbol=symbol, side=close_side, qty=qty_to_close
                )

                if not result or result.get("status") == "ERROR":
                    logger.error(f"❌ Failed to close on SL for {symbol}: {result}")
                    return False

            except Exception as e:
                logger.error(f"❌ Exception placing market order for {symbol}: {e}")
                return False

            # Process execution result
            try:
                exit_price = float(result.get("price", price))
                pnl_partial = self._calc_pnl(side, pos["entry_price"], exit_price, qty_to_close)
                
                # Update position data
                pos["realized_pnl"] = float(pos.get("realized_pnl", 0.0)) + pnl_partial
                pos["remaining_qty"] = max(0.0, remaining_qty - qty_to_close)
                pos["last_exit_price"] = exit_price

                # Finalize position
                self._finalize_position(order_id, pos, exit_price, reason="Stop Loss")
                return True
                
            except Exception as e:
                logger.error(f"❌ Error processing SL execution for {symbol}: {e}")
                return False

        except Exception as e:
            logger.error(f"❌ _check_and_handle_stop failed: {e}")
            logger.error(traceback.format_exc())
            return False

    def calculate_effective_pnl(self, gross_pnl: float, symbol: str, entry_qty: float, entry_price: float, exit_price: Optional[float] = None) -> float:
        """Calculate net PnL after fees and slippage with comprehensive error handling."""
        try:
            # Validate inputs
            if not isinstance(gross_pnl, (int, float)):
                logger.error("❌ Invalid gross_pnl for calculate_effective_pnl")
                return gross_pnl
                
            if not isinstance(symbol, str) or not symbol:
                logger.error("❌ Invalid symbol for calculate_effective_pnl")
                return gross_pnl
                
            if not isinstance(entry_qty, (int, float)) or entry_qty <= 0:
                logger.error("❌ Invalid entry_qty for calculate_effective_pnl")
                return gross_pnl
                
            if not isinstance(entry_price, (int, float)) or entry_price <= 0:
                logger.error("❌ Invalid entry_price for calculate_effective_pnl")
                return gross_pnl

            # Load fees from config
            try:
                cfg = _load_config_loader()
                if cfg:
                    taker_fee = cfg.get("fees", "TAKER_FEE_PCT", 0.06, float) / 100
                    slippage = cfg.get("fees", "EXPECTED_SLIPPAGE_PCT", 0.05, float) / 100
                else:
                    taker_fee = 0.0006  # 0.06% fallback
                    slippage = 0.0005  # 0.05% fallback
            except Exception as e:
                logger.warning(f"⚠️ Failed to load fees config: {e}")
                taker_fee = 0.0006
                slippage = 0.0005

            # Get exit price
            if exit_price is None:
                try:
                    exit_price = self._get_current_price(symbol) or entry_price
                    if exit_price is None or exit_price <= 0:
                        exit_price = entry_price
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get current price for {symbol}: {e}")
                    exit_price = entry_price

            # Calculate values
            try:
                entry_value = float(entry_price) * float(entry_qty)
                exit_value = float(exit_price) * float(entry_qty)
                
                # Entry + Exit fees (both are taker for market orders)
                entry_fee = entry_value * taker_fee
                exit_fee = exit_value * taker_fee
                total_fees = entry_fee + exit_fee
                
                # Slippage cost
                slippage_cost = (entry_value + exit_value) * slippage / 2
                
                # Calculate net PnL
                net_pnl = float(gross_pnl) - total_fees - slippage_cost
                
                # Log calculation
                logger.debug(
                    f"💰 Net PNL calc for {symbol}: "
                    f"gross=${gross_pnl:.4f}, "
                    f"fees=${total_fees:.4f} ({taker_fee*100:.2f}% × 2), "
                    f"slippage=${slippage_cost:.4f}, "
                    f"net=${net_pnl:.4f}"
                )
                
                return net_pnl
                
            except Exception as e:
                logger.error(f"❌ Error calculating effective PnL for {symbol}: {e}")
                return gross_pnl
                
        except Exception as e:
            logger.error(f"❌ calculate_effective_pnl failed: {e}")
            logger.error(traceback.format_exc())
            return gross_pnl
    
    def _finalize_position(self, order_id: str, pos: Dict, exit_price: float, reason: str) -> None:
        """Finalize and close a position, recording all relevant data with comprehensive error handling."""
        global DB_INTEGRATION, close_trade, ML_API_AVAILABLE, send_trade_outcome_to_ml
        
        try:
            # Validate inputs
            if not isinstance(order_id, str) or not order_id:
                logger.error("❌ Invalid order_id for _finalize_position")
                return
                
            if not isinstance(pos, dict):
                logger.error("❌ Invalid position data for _finalize_position")
                return
                
            if not isinstance(exit_price, (int, float)) or exit_price <= 0:
                logger.error("❌ Invalid exit_price for _finalize_position")
                return
                
            if not isinstance(reason, str):
                reason = str(reason)

            # ---------- Safe extraction ----------
            symbol = pos.get("symbol")
            if not symbol or not isinstance(symbol, str):
                logger.error(f"❌ finalize_position: missing or invalid symbol for {order_id}")
                return

            side = str(pos.get("side", "BUY")).upper()
            if side not in ["BUY", "SELL"]:
                logger.error(f"❌ Invalid side '{side}' for {symbol}")
                side = "BUY"

            try:
                entry_price = float(pos.get("entry_price", exit_price))
                initial_qty = float(pos.get("initial_qty", 0.0))
                if initial_qty <= 0:
                    logger.error(f"❌ Invalid initial_qty for {symbol}")
                    return
            except (ValueError, TypeError):
                logger.error(f"❌ Invalid numeric values for {symbol}")
                return

            # ---------- Momentum cleanup ----------
            try:
                if symbol in self.momentum_positions:
                    del self.momentum_positions[symbol]
                    logger.debug(f"✅ Removed {symbol} from momentum positions")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup momentum position for {symbol}: {e}")

            # ---------- Gross PnL ----------
            try:
                if side == "BUY":
                    gross_pnl = (exit_price - entry_price) * initial_qty
                else:
                    gross_pnl = (entry_price - exit_price) * initial_qty

                gross_pnl += float(pos.get("realized_pnl", 0.0))
            except Exception as e:
                logger.error(f"❌ Error calculating gross PnL for {symbol}: {e}")
                gross_pnl = 0.0

            # ---------- Net PnL ----------
            try:
                net_pnl = self.calculate_effective_pnl(
                    gross_pnl=gross_pnl,
                    symbol=symbol,
                    entry_qty=initial_qty,
                    entry_price=entry_price,
                    exit_price=exit_price
                )
            except Exception as e:
                logger.error(f"❌ Error calculating net PnL for {symbol}: {e}")
                net_pnl = gross_pnl

            # ---------- Risk ----------
            try:
                if self.risk:
                    self.risk.track_close(order_id, exit_price, net_pnl)
                    logger.debug(f"📊 Risk manager updated for {symbol}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to update risk manager for {symbol}: {e}")

            # ---------- Daily stats ----------
            try:
                today = datetime.utcnow().date()
                if today != self.closed_trades_date:
                    self.closed_trades_today = []
                    self.closed_trades_date = today

                self.closed_trades_today.append({
                    "symbol": symbol,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "fees": gross_pnl - net_pnl,
                    "reason": reason,
                    "is_momentum": bool(pos.get("is_momentum", False)),
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.warning(f"⚠️ Failed to update daily stats for {symbol}: {e}")

            # ---------- DB (lazy load) ----------
            try:
                trade_id_raw = pos.get("trade_id")
                
                # Convert to int safely
                if trade_id_raw is None:
                    logger.warning(f"⚠️ No trade_id found for position {order_id} ({symbol}) — skipping DB close")
                else:
                    try:
                        trade_id = int(trade_id_raw)
                        if trade_id <= 0:
                            raise ValueError("trade_id must be positive")
                    except (ValueError, TypeError) as e:
                        logger.error(f"❌ Invalid trade_id format: {trade_id_raw} (type: {type(trade_id_raw)}) — cannot close in DB")
                        trade_id = None

                    if trade_id:
                        _load_db_utils()  # Ensure DB is loaded
                        if DB_INTEGRATION and close_trade:
                            try:
                                success = close_trade(
                                    trade_id=trade_id,
                                    exit_price=float(exit_price),
                                    pnl=float(net_pnl),
                                    comment=reason
                                )
                                if success:
                                    logger.success(f"✅ Trade ID {trade_id} for {symbol} successfully closed in database")
                                else:
                                    logger.error(f"❌ close_trade returned False for trade_id={trade_id}, symbol={symbol}")
                            except Exception as e:
                                logger.error(f"💥 Exception calling close_trade for {symbol}: {e}")
                                logger.error(traceback.format_exc())
                        else:
                            logger.warning("⚠️ DB_INTEGRATION disabled or close_trade not available")
            except Exception as e:
                logger.warning(f"⚠️ Failed to update database for {symbol}: {e}")
                logger.error(traceback.format_exc())

            # ---------- Cleanup ----------
            try:
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                    logger.debug(f"🧹 Removed {symbol} from active orders")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup active orders for {symbol}: {e}")

            # ---------- Logging ----------
            try:
                fee_cost = gross_pnl - net_pnl
                fee_pct = (fee_cost / abs(gross_pnl) * 100) if gross_pnl != 0 else 0.0
                momentum_tag = " 🚀" if pos.get("is_momentum", False) else ""

                logger.success(
                    f"🧾 Closed {symbol}{momentum_tag} ({reason}):\n"
                    f"   Gross PnL: ${gross_pnl:.4f}\n"
                    f"   Fees+Slippage: ${fee_cost:.4f} ({fee_pct:.1f}% of PnL)\n"
                    f"   Net PnL: ${net_pnl:.4f}\n"
                    f"   Exit price: {exit_price:.8f}"
                )
            except Exception as e:
                logger.error(f"❌ Error logging finalization for {symbol}: {e}")

            # ---------- ML Recording ----------
            try:
                logger.info(f"🔍 ML Recording for {symbol}: checking...")
                logger.info(f"   ml_manager: {self.ml_manager is not None}")
                logger.info(f"   enabled: {getattr(self.ml_manager, 'enabled', False) if self.ml_manager else 'N/A'}")

                if self.ml_manager and getattr(self.ml_manager, "enabled", False):
                    try:
                        signal_data = pos.get("signal_data")
                        logger.info(f"   signal_data exists: {signal_data is not None}")
                        
                        # Create basic signal_data if missing
                        if not signal_data:
                            logger.info(f"📝 Creating basic signal_data for {symbol}")
                            signal_data = {
                                "symbol": symbol,
                                "side": side,
                                "price": entry_price,
                                "strength": float(pos.get("signal_strength", 50.0)),
                                "stop_loss": pos.get("stop_loss"),
                                "take_profit": pos.get("take_profit"),
                                "timestamp": datetime.utcnow().isoformat(),
                                "order_id": order_id
                            }

                        # Get klines (optional)
                        df_ml = pd.DataFrame()
                        get_kline = getattr(self.session, "get_kline", None)
                        
                        if callable(get_kline):
                            try:
                                kl = get_kline(category="linear", symbol=symbol, interval="15", limit=100)
                                if kl.get("retCode") == 0:
                                    rows = kl.get("result", {}).get("list", [])
                                    if rows:
                                        df_ml = pd.DataFrame(rows, columns=[
                                            "open_time", "open", "high", "low", "close", "volume", "turnover"
                                        ])
                                        for col in ["open", "high", "low", "close", "volume"]:
                                            df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce')
                                        logger.debug(f"   Got {len(df_ml)} candles for ML")
                            except Exception as e:
                                logger.debug(f"   Kline fetch failed: {e}")

                        # Record to ML
                        logger.info(f"📝 Calling ml_manager.record_trade_outcome for {symbol}...")
                        
                        result = self.ml_manager.record_trade_outcome(
                            signal=signal_data,
                            df=df_ml,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            pnl=net_pnl,
                            sl_used=pos.get("stop_loss"),
                            tp_used=pos.get("take_profit")
                        )

                        if result:
                            logger.success(f"✅ ML RECORDED: {symbol} | PnL: ${net_pnl:.2f} | {'WIN' if net_pnl > 0 else 'LOSS'}")
                        else:
                            logger.warning(f"⚠️ ML record_trade_outcome returned False for {symbol}")

                    except Exception as e:
                        logger.error(f"❌ ML recording error for {symbol}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                else:
                    reason_msg = "ml_manager is None" if not self.ml_manager else "ml_manager.enabled is False"
                    logger.warning(f"⚠️ ML NOT recorded for {symbol}: {reason_msg}")

            except Exception as e:
                logger.error(f"❌ Error in ML recording section for {symbol}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        except Exception as e:
            logger.error(f"❌ _finalize_position error for {order_id}: {e}")
            logger.error(traceback.format_exc())

    def monitor_positions(self, indicators_provider=None) -> None:
        """
        Monitor all active positions for SL/TP hits and trailing updates with comprehensive error handling.
        """
        try:
            if not self.active_orders:
                logger.success("ℹ️  No active orders to monitor")
                return

            MAX_SLTP_RETRY_TIME = 300

            # Identify positions without SL/TP
            positions_without_sltp = [
                pos["symbol"] 
                for pos in self.active_orders.values() 
                if not pos.get("sltp_ok", True)
            ]

            if positions_without_sltp:
                logger.debug(f"⚠️ Positions without SL/TP: {positions_without_sltp}")

            # Process each active position
            for order_id, pos in list(self.active_orders.items()):
                try:
                    symbol = str(pos.get("symbol", "")).upper().strip()
                    if not symbol:
                        logger.warning(f"⚠️ Position {order_id} missing symbol - skipping")
                        continue

                    # Get current price
                    price = self._get_current_price(symbol)
                    if price is None:
                        logger.debug(f"⚠️ Could not fetch price for {symbol}, skipping monitoring")
                        continue

                    # Validate price
                    if not isinstance(price, (int, float)) or price <= 0:
                        logger.warning(f"⚠️ Invalid price for {symbol}: {price} - skipping")
                        continue

                    opened_at = pos.get("opened_at", time.time())
                    age_seconds = time.time() - opened_at
                    
                    # Handle SL/TP retry logic
                    if not pos.get("sltp_ok", True):
                        attempts = int(pos.get("sltp_attempts", 0))
                        
                        if age_seconds > MAX_SLTP_RETRY_TIME:
                            logger.error(
                                f"🚨 SL/TP retry timeout for {symbol} after "
                                f"{age_seconds:.0f}s ({attempts} attempts) - CLOSING POSITION"
                            )
                            
                            remaining_qty = float(pos.get("remaining_qty", pos.get("initial_qty", 0.0)))
                            qty = self._sanitize_quantity(symbol, remaining_qty)
                            
                            if qty > 0:
                                side = "Sell" if str(pos.get("side", "BUY")).upper() == "BUY" else "Buy"
                                
                                try:
                                    result = self.session.place_market_order(
                                        symbol=symbol, side=side, qty=qty
                                    )
                                    
                                    if result and result.get("status") != "ERROR":
                                        exit_price = float(result.get("price", price))
                                        logger.warning(
                                            f"⚠️ Emergency close due to SL/TP timeout: "
                                            f"{symbol} @ {exit_price:.8f}"
                                        )
                                        self._finalize_position(
                                            order_id, pos, exit_price, 
                                            reason="SL/TP retry timeout"
                                        )
                                    else:
                                        logger.error(
                                            f"❌ Failed to emergency close {symbol}: {result}"
                                        )
                                except Exception as e:
                                    logger.error(f"❌ Exception closing {symbol}: {e}")
                            
                            continue
                        
                        if attempts < 5:
                            try:
                                attempt_num = attempts + 1
                                logger.info(
                                    f"🔄 Retrying SL/TP for {symbol} (attempt {attempt_num}/5, "
                                    f"age={age_seconds:.0f}s)"
                                )
                                
                                sltp_ok = self._set_position_stops(
                                    symbol,
                                    pos.get("stop_loss"),
                                    pos.get("take_profit"),
                                    side=pos.get("side", "BUY"),
                                    entry_price=pos.get("entry_price", price),
                                )
                                
                                pos["sltp_ok"] = bool(sltp_ok)
                                pos["sltp_attempts"] = attempt_num
                                
                                if sltp_ok:
                                    logger.success(
                                        f"✅ SL/TP retry successful for {symbol} "
                                        f"on attempt {attempt_num}"
                                    )
                                else:
                                    logger.warning(
                                        f"⚠️ SL/TP retry failed for {symbol} "
                                        f"(attempt {attempt_num}/5)"
                                    )
                                    
                            except Exception as e:
                                logger.error(f"❌ SL/TP retry exception for {symbol}: {e}")
                                logger.error(traceback.format_exc())

                    # Handle momentum position exits
                    if pos.get("is_momentum", False) and symbol in self.momentum_positions:
                        try:
                            momentum_data = self.momentum_positions[symbol]
                            
                            indicators = None
                            if indicators_provider and callable(indicators_provider):
                                try:
                                    indicators = indicators_provider(symbol)
                                except Exception as e:
                                    logger.debug(f"⚠️ Failed to get indicators for {symbol}: {e}")
                            
                            should_exit, exit_reason = self._should_exit_momentum_position(
                                symbol, momentum_data, price, indicators
                            )
                            
                            if should_exit:
                                logger.warning(
                                    f"🔄 Momentum exit triggered for {symbol}: {exit_reason}"
                                )
                                
                                remaining_qty = float(pos.get("remaining_qty", pos.get("initial_qty", 0.0)))
                                qty = self._sanitize_quantity(symbol, remaining_qty)
                                
                                if qty > 0:
                                    side = "Sell" if str(pos.get("side", "BUY")).upper() == "BUY" else "Buy"
                                    
                                    try:
                                        result = self.session.place_market_order(
                                            symbol=symbol, side=side, qty=qty
                                        )
                                        
                                        if result and result.get("status") != "ERROR":
                                            exit_price = float(result.get("price", price))
                                            self._finalize_position(
                                                order_id, pos, exit_price, 
                                                reason=f"Momentum exit: {exit_reason}"
                                            )
                                        else:
                                            logger.error(
                                                f"❌ Failed to close momentum position {symbol}: {result}"
                                            )
                                    except Exception as e:
                                        logger.error(f"❌ Exception closing momentum position {symbol}: {e}")
                                
                                continue
                                
                        except Exception as e:
                            logger.error(f"❌ Error handling momentum position for {symbol}: {e}")
                            logger.error(traceback.format_exc())

                    # Update trailing stops
                    try:
                        self._update_trailing(order_id, pos, price)
                    except Exception as e:
                        logger.error(f"❌ trailing update failed for {symbol}: {e}")
                        logger.error(traceback.format_exc())

                    # Check stop loss
                    try:
                        closed = self._check_and_handle_stop(order_id, pos, price)
                        if closed:
                            continue
                    except Exception as e:
                        logger.error(f"❌ SL check failed for {symbol}: {e}")
                        logger.error(traceback.format_exc())

                    # Check take profit
                    try:
                        self._check_and_handle_tp(order_id, pos, price)
                    except Exception as e:
                        logger.error(f"❌ TP ladder check failed for {symbol}: {e}")
                        logger.error(traceback.format_exc())
                        
                except Exception as e:
                    logger.error(f"❌ Error processing position {order_id} ({symbol}): {e}")
                    logger.error(traceback.format_exc())
                    continue

        except Exception as e:
            logger.error(f"❌ monitor_positions failed: {e}")
            logger.error(traceback.format_exc())

    def close_all_positions(self, reason: str = "Manual") -> None:
        """Close all active positions with comprehensive error handling."""
        try:
            logger.warning(f"🚨 Closing ALL positions: {reason}")

            # Clear momentum positions
            try:
                self.momentum_positions.clear()
                logger.debug("✅ Cleared momentum positions")
            except Exception as e:
                logger.warning(f"⚠️ Failed to clear momentum positions: {e}")

            # Close each active position
            for order_id, pos in list(self.active_orders.items()):
                try:
                    symbol = str(pos.get("symbol", "")).upper().strip()
                    if not symbol:
                        logger.warning(f"⚠️ Position {order_id} missing symbol - skipping")
                        continue

                    # Get and validate quantity
                    try:
                        remaining_qty = float(pos.get("remaining_qty", pos.get("initial_qty", 0.0)))
                        qty = self._sanitize_quantity(symbol, remaining_qty)
                        if qty <= 0:
                            logger.warning(f"⚠️ No quantity to close for {symbol}")
                            continue
                    except (ValueError, TypeError):
                        logger.error(f"❌ Invalid quantity for {symbol}")
                        continue

                    # Determine close side
                    try:
                        side_str = str(pos.get("side", "BUY")).upper()
                        close_side = "Sell" if side_str == "BUY" else "Buy"
                    except Exception as e:
                        logger.error(f"❌ Error determining close side for {symbol}: {e}")
                        continue

                    # Place market order
                    try:
                        result = self.session.place_market_order(symbol=symbol, side=close_side, qty=qty)
                        
                        if result and result.get("status") != "ERROR":
                            try:
                                exit_price = float(result.get("price", 0))
                                pnl_partial = self._calc_pnl(
                                    pos.get("side", "BUY"), 
                                    pos.get("entry_price", 0), 
                                    exit_price, 
                                    qty
                                )
                                pos["realized_pnl"] = float(pos.get("realized_pnl", 0.0)) + pnl_partial
                                pos["remaining_qty"] = max(0.0, remaining_qty - qty)
                                pos["last_exit_price"] = exit_price
                                self._finalize_position(
                                    order_id, pos, exit_price, reason=f"{reason} (manual close_all)"
                                )
                            except Exception as e:
                                logger.error(f"❌ Error processing close result for {symbol}: {e}")
                        else:
                            logger.error(f"❌ Failed to close {symbol}: {result}")
                            
                    except Exception as e:
                        logger.error(f"❌ Exception closing {symbol}: {e}")
                        logger.error(traceback.format_exc())
                        
                except Exception as e:
                    logger.error(f"❌ Error closing position {order_id}: {e}")
                    logger.error(traceback.format_exc())
                    continue

        except Exception as e:
            logger.error(f"❌ close_all_positions failed: {e}")
            logger.error(traceback.format_exc())

    def close_position_market(self, symbol: str, side: str, reason: str = "Manual") -> Optional[Dict]:
        """Close a specific position by symbol with comprehensive error handling."""
        try:
            symbol = str(symbol).upper().strip()
            if not symbol:
                logger.error("❌ Invalid symbol for close_position_market")
                return {"status": "ERROR", "error": "Invalid symbol"}

            # Find the order_id for this symbol
            order_id = None
            pos = None
            for oid, p in self.active_orders.items():
                if str(p.get("symbol", "")).upper().strip() == symbol:
                    order_id = oid
                    pos = p
                    break
            
            if not pos:
                logger.warning(f"⚠️ No active position found for {symbol}")
                return {"status": "ERROR", "error": f"No active position found for {symbol}"}
            
            # Validate and sanitize quantity
            try:
                remaining_qty = float(pos.get("remaining_qty", pos.get("initial_qty", 0.0)))
                qty = self._sanitize_quantity(symbol, remaining_qty)
                
                if qty <= 0:
                    logger.warning(f"⚠️ No quantity to close for {symbol}")
                    return {"status": "ERROR", "error": f"No quantity to close for {symbol}"}
            except (ValueError, TypeError) as e:
                logger.error(f"❌ Invalid quantity for {symbol}: {e}")
                return {"status": "ERROR", "error": f"Invalid quantity: {e}"}

            # Determine close side
            try:
                pos_side = str(pos.get("side", "BUY")).upper()
                close_side = "Sell" if pos_side == "BUY" else "Buy"
            except Exception as e:
                logger.error(f"❌ Error determining close side for {symbol}: {e}")
                return {"status": "ERROR", "error": f"Error determining close side: {e}"}

            # Place market order
            try:
                result = self.session.place_market_order(symbol=symbol, side=close_side, qty=qty)
                
                if result and result.get("status") != "ERROR":
                    try:
                        exit_price = float(result.get("price", 0))
                        self._finalize_position(order_id, pos, exit_price, reason=reason)
                        return {"status": "SUCCESS", "exit_price": exit_price}
                    except Exception as e:
                        logger.error(f"❌ Error finalizing position for {symbol}: {e}")
                        return {"status": "ERROR", "error": f"Error finalizing position: {e}"}
                else:
                    logger.error(f"❌ Failed to close {symbol}: {result}")
                    return {"status": "ERROR", "error": f"Failed to close position: {result}"}
                    
            except Exception as e:
                logger.error(f"❌ Exception closing {symbol}: {e}")
                logger.error(traceback.format_exc())
                return {"status": "ERROR", "error": str(e)}
                
        except Exception as e:
            logger.error(f"❌ close_position_market failed: {e}")
            logger.error(traceback.format_exc())
            return {"status": "ERROR", "error": str(e)}

    def close_position_by_symbol(self, symbol: str, reason: str = "Signal Flip") -> bool:
        """
        Close the entire open position for a given symbol.
        This is what you need for 'flipping' logic.
        """
        try:
            symbol = symbol.upper().strip()
            if not symbol:
                logger.error("❌ Invalid symbol for close_position_by_symbol")
                return False

            # Get current position info from exchange
            position_info = self.session.get_position(symbol)
            if not position_info:
                logger.warning(f"⚠️ No position found for {symbol}")
                return True  # already flat

            size = float(position_info.get("size", 0))
            side = position_info.get("side", "").upper()

            if size <= 0:
                logger.debug(f"✅ {symbol} already flat")
                return True

            # Determine closing side
            close_side = "Sell" if side == "Buy" else "Buy"
            qty = self._sanitize_quantity(symbol, size)

            logger.info(f"CloseOperation: Closing {symbol} {side} position of {qty} (reason: {reason})")

            # Place reduce-only market order
            result = self.session.place_market_order(
                symbol=symbol,
                side=close_side,
                qty=qty,
                reduce_only=True
            )

            if result and result.get("status") != "ERROR":
                exit_price = float(result.get("price", 0)) or self._get_current_price(symbol)
                logger.success(f"✅ Closed {symbol} @ {exit_price:.8f} | Reason: {reason}")
                return True
            else:
                logger.error(f"❌ Failed to close {symbol}: {result}")
                return False

        except Exception as e:
            logger.exception(f"💥 Exception in close_position_by_symbol({symbol}): {e}")
            return False
    
    def sync_closed_trades_to_ml(self) -> int:
        """
        Sync closed trades from exchange to ML with comprehensive error handling.
        Checks for positions that were closed externally (TP/SL hit on exchange).
        Returns number of trades synced.
        """
        try:
            if not self.ml_manager or not getattr(self.ml_manager, "enabled", False):
                logger.debug("ℹ️ ML manager not available or disabled")
                return 0
            
            synced = 0
            
            try:
                # Get closed PnL from exchange
                try:
                    closed_pnl = self.session.get_closed_pnl(limit=50)
                    if not closed_pnl:
                        logger.debug("ℹ️ No closed PnL data available")
                        return 0
                except Exception as e:
                    logger.error(f"❌ Failed to get closed PnL: {e}")
                    return 0
                
                # Get already recorded trade IDs
                existing_trades = set()
                try:
                    if hasattr(self.ml_manager, 'storage'):
                        for trade in self.ml_manager.storage.load_trades():
                            # Use orderId or symbol+timestamp as key
                            key = trade.get('order_id') or f"{trade.get('symbol')}_{trade.get('timestamp', '')[:16]}"
                            existing_trades.add(key)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load existing trades: {e}")

                # Process each closed trade
                for trade in closed_pnl:
                    try:
                        order_id = trade.get('orderId')
                        symbol = trade.get('symbol')
                        
                        # Skip if already recorded
                        if order_id in existing_trades:
                            continue
                        
                        # Extract and validate trade data
                        try:
                            side = str(trade.get('side', 'Buy'))
                            entry_price = float(trade.get('avgEntryPrice', 0))
                            exit_price = float(trade.get('avgExitPrice', 0))
                            pnl = float(trade.get('closedPnl', 0))
                            qty = float(trade.get('qty', 0))
                            
                            if entry_price == 0 or exit_price == 0:
                                logger.warning(f"⚠️ Invalid prices for {symbol}: entry={entry_price}, exit={exit_price}")
                                continue
                        except (ValueError, TypeError) as e:
                            logger.error(f"❌ Invalid trade data for {symbol}: {e}")
                            continue
                        
                        # Create signal data
                        signal_data = {
                            "symbol": symbol,
                            "side": side,
                            "price": entry_price,
                            "strength": 50,  # Unknown strength
                            "stop_loss": None,
                            "take_profit": None,
                            "timestamp": datetime.utcnow().isoformat(),
                            "source": "exchange_sync"
                        }
                        
                        # Get klines for features
                        df_ml = pd.DataFrame()
                        try:
                            kl = self.session.get_kline(category="linear", symbol=symbol, interval="15", limit=100)
                            if kl.get("retCode") == 0:
                                rows = kl.get("result", {}).get("list", [])
                                if rows:
                                    df_ml = pd.DataFrame(rows, columns=[
                                        "open_time", "open", "high", "low", "close", "volume", "turnover"
                                    ])
                                    for col in ["open", "high", "low", "close", "volume"]:
                                        df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce')
                        except Exception as e:
                            logger.debug(f"⚠️ Kline fetch failed for {symbol}: {e}")
                        
                        # Record to ML
                        try:
                            result = self.ml_manager.record_trade_outcome(
                                signal=signal_data,
                                df=df_ml,
                                entry_price=entry_price,
                                exit_price=exit_price,
                                pnl=pnl,
                                sl_used=None,
                                tp_used=None
                            )
                            
                            if result:
                                logger.info(f"📝 Synced closed trade to ML: {symbol} ${pnl:.2f}")
                                synced += 1
                            else:
                                logger.warning(f"⚠️ ML record_trade_outcome returned False for {symbol}")
                                
                        except Exception as e:
                            logger.error(f"❌ ML recording error for {symbol}: {e}")
                            logger.error(traceback.format_exc())
                            
                    except Exception as e:
                        logger.error(f"❌ Error processing trade {trade.get('orderId', 'Unknown')}: {e}")
                        logger.error(traceback.format_exc())
                        continue
                
                if synced > 0:
                    logger.success(f"✅ Synced {synced} closed trades to ML")
                
                return synced
                
            except Exception as e:
                logger.error(f"❌ sync_closed_trades_to_ml error: {e}")
                logger.error(traceback.format_exc())
                return 0
                
        except Exception as e:
            logger.error(f"❌ sync_closed_trades_to_ml failed: {e}")
            logger.error(traceback.format_exc())
            return 0
    
    def get_positions_summary(self) -> Dict:
        """Get summary of all active and closed positions with comprehensive error handling."""
        try:
            summary = []
            total_unrealized = 0.0

            # Process active positions
            for order_id, pos in self.active_orders.items():
                try:
                    symbol = str(pos.get("symbol", "")).upper().strip()
                    if not symbol:
                        continue

                    price = self._get_current_price(symbol)
                    if not price or not isinstance(price, (int, float)) or price <= 0:
                        continue

                    try:
                        entry = float(pos.get("entry_price", 0))
                        qty = float(pos.get("remaining_qty", pos.get("initial_qty", 0.0)))
                        if qty <= 0:
                            continue

                        side = str(pos.get("side", "BUY")).upper()
                        if side == "BUY":
                            pnl = (price - entry) * qty
                        else:
                            pnl = (entry - price) * qty

                        total_unrealized += pnl

                        summary.append(
                            {
                                "symbol": symbol,
                                "side": pos.get("side"),
                                "entry": entry,
                                "current": price,
                                "qty": qty,
                                "unrealized_pnl": pnl,
                                "sl": pos.get("stop_loss"),
                                "tp": pos.get("take_profit"),
                                "is_momentum": bool(pos.get("is_momentum", False)),
                                "order_id": order_id,
                                "age_minutes": round((time.time() - pos.get("opened_at", time.time())) / 60, 2)
                            }
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(f"⚠️ Invalid position data for {symbol}: {e}")
                        continue
                        
                except Exception as e:
                    logger.error(f"❌ Error processing position {order_id}: {e}")
                    logger.error(traceback.format_exc())
                    continue

            # Calculate daily statistics
            try:
                realized_today = sum(t.get("net_pnl", 0.0) for t in self.closed_trades_today)
                wins_today = sum(1 for t in self.closed_trades_today if t.get("net_pnl", 0) > 0)
                losses_today = sum(1 for t in self.closed_trades_today if t.get("net_pnl", 0) < 0)
                
                momentum_trades = [t for t in self.closed_trades_today if t.get("is_momentum", False)]
                regular_trades = [t for t in self.closed_trades_today if not t.get("is_momentum", False)]

                win_rate = (wins_today / len(self.closed_trades_today) * 100) if len(self.closed_trades_today) > 0 else 0.0

                result = {
                    "positions": summary,
                    "total_unrealized_pnl": total_unrealized,
                    "total_positions": len(summary),
                    "total_value": sum(p.get("current", 0) * p.get("qty", 0) for p in summary),

                    "realized_pnl_today": realized_today,
                    "trades_today": len(self.closed_trades_today),
                    "wins_today": wins_today,
                    "losses_today": losses_today,
                    "win_rate_today": win_rate,
                    
                    "momentum_trades_count": len(momentum_trades),
                    "momentum_wins": sum(1 for t in momentum_trades if t.get("net_pnl", 0) > 0),
                    "momentum_losses": sum(1 for t in momentum_trades if t.get("net_pnl", 0) < 0),
                    "momentum_pnl": sum(t.get("net_pnl", 0.0) for t in momentum_trades),
                    "regular_trades_count": len(regular_trades),
                    "regular_pnl": sum(t.get("net_pnl", 0.0) for t in regular_trades),
                    
                    "timestamp": datetime.utcnow().isoformat(),
                    "date": datetime.utcnow().strftime("%Y-%m-%d")
                }
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Error calculating daily statistics: {e}")
                logger.error(traceback.format_exc())
                
                # Return basic summary even if daily stats fail
                return {
                    "positions": summary,
                    "total_unrealized_pnl": total_unrealized,
                    "total_positions": len(summary),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ get_positions_summary failed: {e}")
            logger.error(traceback.format_exc())
            
            # Return empty structure
            return {
                "positions": [],
                "total_unrealized_pnl": 0.0,
                "total_positions": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def sync_and_cleanup_positions(self, exchange_positions: List[Dict]) -> int:
        """Sync internal state with exchange positions and cleanup stale entries with comprehensive error handling."""
        try:
            if not exchange_positions:
                exchange_open_symbols = set()
            else:
                try:
                    exchange_open_symbols = {
                        str(pos.get("symbol", "")).upper().strip()
                        for pos in exchange_positions
                        if float(pos.get("size") or 0.0) > 0
                    }
                except Exception as e:
                    logger.error(f"❌ Error processing exchange positions: {e}")
                    exchange_open_symbols = set()

            orders_to_remove = []
            
            # Identify positions to remove
            for order_id, pos_data in list(self.active_orders.items()):
                try:
                    symbol = str(pos_data.get("symbol", "")).upper().strip()
                    if not symbol:
                        continue
                        
                    if symbol and symbol not in exchange_open_symbols:
                        orders_to_remove.append((order_id, symbol, pos_data))
                        
                except Exception as e:
                    logger.error(f"❌ Error checking position {order_id}: {e}")
                    logger.error(traceback.format_exc())
                    continue

            closed_count = 0
            
            # Process positions to remove
            for order_id, symbol, pos_data in orders_to_remove:
                try:
                    logger.warning(
                        f"🧹 Position {symbol} closed externally (TP/SL hit on exchange)"
                    )
                    
                    exit_price = self._get_current_price_safe(symbol)
                    
                    if exit_price is not None and isinstance(exit_price, (int, float)) and exit_price > 0:
                        try:
                            self._finalize_position(
                                order_id, 
                                pos_data, 
                                float(exit_price), 
                                reason="External close (TP/SL)"
                            )
                            closed_count += 1
                        except Exception as e:
                            logger.error(f"❌ Error finalizing position {symbol}: {e}")
                            logger.error(traceback.format_exc())
                    else:
                        logger.error(f"❌ Could not get valid exit price for {symbol}")
                        
                        # Cleanup without finalization
                        try:
                            if symbol in self.momentum_positions:
                                del self.momentum_positions[symbol]
                                
                            if self.risk:
                                self.risk.track_close(order_id, 0.0, 0.0)
                                
                            if order_id in self.active_orders:
                                del self.active_orders[order_id]
                        except Exception as e:
                            logger.error(f"❌ Error cleaning up position {symbol}: {e}")
                            logger.error(traceback.format_exc())
                            
                except Exception as e:
                    logger.error(f"❌ Error processing external close for {symbol}: {e}")
                    logger.error(traceback.format_exc())
                    continue

            if closed_count > 0:
                logger.success(f"✅ Synced {closed_count} externally closed trades to ML")

            return len(orders_to_remove)
            
        except Exception as e:
            logger.error(f"❌ sync_and_cleanup_positions failed: {e}")
            logger.error(traceback.format_exc())
            return 0