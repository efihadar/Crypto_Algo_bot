# emergency_exit_manager.py
"""
🚨 Emergency Exit Manager - Advanced Protection System

Features:
1. Percentage-based loss exit (e.g., close if -2% from entry)
2. Liquidation distance monitoring (close if too close)
3. Missing SL/TP detection & recovery
4. Time-based trailing stop-loss tightening
5. Correlation with account equity drawdown
6. Funding cost monitoring
7. Maximum position age exit
8. Momentum-specific exit conditions
"""

import time
import traceback
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from loguru import logger
from config_loader import get_config_loader


class EmergencyExitManager:
    """
    Advanced Emergency Exit Manager - Protects the bot from catastrophic losses.
    Implements multiple layers of protection with configurable thresholds.
    """

    def __init__(self, session=None, order_manager=None):
        """
        Initialize the Emergency Exit Manager.
        
        :param session: BybitSession instance for fetching positions and prices
        :param order_manager: OrderManager instance for executing closes
        """
        self.session = session
        self.order_manager = order_manager

        # Load config
        cfg = get_config_loader()
        section = "emergency_exit"

        # ============================================
        # 📉 Percentage Loss Exit
        # ============================================
        self.enable_pct_loss_exit: bool = cfg.get(
            section, "ENABLE_PCT_LOSS_EXIT", True, bool
        )
        self.max_loss_pct_per_trade: float = cfg.get(
            section, "MAX_LOSS_PCT_PER_TRADE", 2.0, float
        )

        # ============================================
        # 💀 Liquidation Distance Protection
        # ============================================
        self.enable_liq_protection: bool = cfg.get(
            section, "ENABLE_LIQ_PROTECTION", True, bool
        )
        self.liq_distance_threshold_pct: float = cfg.get(
            section, "LIQ_DISTANCE_THRESHOLD_PCT", 15.0, float
        )

        # ============================================
        # 🛡️ Missing SL/TP Recovery
        # ============================================
        self.enable_missing_sltp_exit: bool = cfg.get(
            section, "ENABLE_MISSING_SLTP_EXIT", True, bool
        )
        self.missing_sltp_timeout_minutes: int = cfg.get(
            section, "MISSING_SLTP_TIMEOUT_MINUTES", 10, int
        )

        # ============================================
        # ⏰ Time-Based Trailing (tighten SL over time)
        # ============================================
        self.enable_time_trailing: bool = cfg.get(
            section, "ENABLE_TIME_TRAILING", False, bool
        )
        self.time_trailing_intervals: List[Dict[str, float]] = self._parse_time_trailing(
            cfg.get(section, "TIME_TRAILING_INTERVALS", "30:1.5,60:1.0,120:0.5", str)
        )

        # ============================================
        # 🌊 Account Equity Drawdown Protection
        # ============================================
        self.enable_equity_drawdown_exit: bool = cfg.get(
            section, "ENABLE_EQUITY_DRAWDOWN_EXIT", True, bool
        )
        self.max_equity_drawdown_pct: float = cfg.get(
            section, "MAX_EQUITY_DRAWDOWN_PCT", 5.0, float
        )

        # ============================================
        # 💸 Funding Cost Protection
        # ============================================
        self.enable_funding_protection: bool = cfg.get(
            section, "ENABLE_FUNDING_PROTECTION", True, bool
        )
        self.max_funding_cost_pct: float = cfg.get(
            section, "MAX_FUNDING_COST_PCT", 0.15, float
        )

        # ============================================
        # ⏱️ Maximum Position Age
        # ============================================
        self.enable_max_age_exit: bool = cfg.get(
            section, "ENABLE_MAX_AGE_EXIT", False, bool
        )
        self.max_position_age_hours: float = cfg.get(
            section, "MAX_POSITION_AGE_HOURS", 24.0, float
        )

        # ============================================
        # 🚀 Momentum-Specific Settings
        # ============================================
        self.momentum_max_loss_pct: float = cfg.get(
            section, "MOMENTUM_MAX_LOSS_PCT", 1.5, float
        )
        self.momentum_max_age_minutes: int = cfg.get(
            section, "MOMENTUM_MAX_AGE_MINUTES", 45, int
        )

        # ============================================
        # 📊 Internal State
        # ============================================
        self.last_equity_peak: Optional[float] = None
        self.exit_cooldown: Dict[str, float] = {}  # symbol -> last exit timestamp
        self.exit_cooldown_seconds: int = 60  # Prevent rapid re-entries

        logger.success(
            f"✅ EmergencyExitManager initialized:\n"
            f"   - Pct Loss Exit: {self.enable_pct_loss_exit} (max={self.max_loss_pct_per_trade}%)\n"
            f"   - Liq Protection: {self.enable_liq_protection} (threshold={self.liq_distance_threshold_pct}%)\n"
            f"   - Missing SL/TP Recovery: {self.enable_missing_sltp_exit} (timeout={self.missing_sltp_timeout_minutes}m)\n"
            f"   - Time Trailing: {self.enable_time_trailing}\n"
            f"   - Equity Drawdown Exit: {self.enable_equity_drawdown_exit} (max={self.max_equity_drawdown_pct}%)\n"
            f"   - Funding Protection: {self.enable_funding_protection} (max={self.max_funding_cost_pct}%)\n"
            f"   - Max Age Exit: {self.enable_max_age_exit} ({self.max_position_age_hours}h)"
        )

    # ================================================================
    # 🔧 Helper: Parse time trailing config
    # ================================================================
    def _parse_time_trailing(self, raw: str) -> List[Dict[str, float]]:
        """
        Parse format: "30:1.5,60:1.0,120:0.5"
        → [{"minutes": 30, "sl_multiplier": 1.5}, ...]
        
        Each interval defines: after X minutes, SL should be Y * ATR from entry
        """
        try:
            if not raw or not isinstance(raw, str):
                return []
                
            intervals = []
            for part in raw.split(","):
                part = part.strip()
                if ":" not in part:
                    continue
                mins_str, mult_str = part.split(":", 1)
                intervals.append({
                    "minutes": float(mins_str.strip()),
                    "sl_multiplier": float(mult_str.strip()),
                })
            return sorted(intervals, key=lambda x: x["minutes"])
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse TIME_TRAILING_INTERVALS: {e}")
            return []

    # ================================================================
    # 🔍 Helper: Get current price for symbol
    # ================================================================
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Safely get current price for a symbol."""
        if not self.session:
            return None
            
        try:
            if hasattr(self.session, "get_current_price"):
                return self.session.get_current_price(symbol)
        except Exception as e:
            logger.debug(f"⚠️ Could not fetch price for {symbol}: {e}")
        
        return None

    # ================================================================
    # 📉 Check percentage loss from entry
    # ================================================================
    def _check_pct_loss(
        self,
        position: Dict[str, Any],
        current_price: float,
        is_momentum: bool = False
    ) -> Optional[str]:
        """
        Returns close reason if loss exceeds threshold, else None.
        Uses different threshold for momentum trades.
        """
        if not self.enable_pct_loss_exit:
            return None

        entry_price = float(position.get("avgPrice") or position.get("entryPrice") or 0)
        side = str(position.get("side", "")).capitalize()

        if entry_price <= 0 or current_price <= 0:
            return None

        # Calculate current P&L %
        if side == "Buy":
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        elif side == "Sell":
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        else:
            return None

        # Use appropriate threshold
        threshold = self.momentum_max_loss_pct if is_momentum else self.max_loss_pct_per_trade

        # Check if loss exceeds threshold
        if pnl_pct < -threshold:
            trade_type = "MOMENTUM" if is_momentum else "REGULAR"
            return f"PCT_LOSS_EXIT ({trade_type}): {pnl_pct:.2f}% < -{threshold}%"

        return None

    # ================================================================
    # 💀 Check distance to liquidation price
    # ================================================================
    def _check_liquidation_distance(
        self,
        position: Dict[str, Any],
        current_price: float
    ) -> Optional[str]:
        """
        Returns close reason if too close to liquidation, else None.
        """
        if not self.enable_liq_protection:
            return None

        liq_price = float(position.get("liqPrice", 0) or 0)
        if liq_price <= 0 or current_price <= 0:
            return None

        side = str(position.get("side", "")).capitalize()

        # Calculate distance to liquidation
        if side == "Buy":
            # For long: liq price is below current
            if liq_price >= current_price:
                return f"LIQ_PROTECTION: Liq price {liq_price} >= current {current_price}"
            distance_pct = ((current_price - liq_price) / current_price) * 100
        elif side == "Sell":
            # For short: liq price is above current
            if liq_price <= current_price:
                return f"LIQ_PROTECTION: Liq price {liq_price} <= current {current_price}"
            distance_pct = ((liq_price - current_price) / current_price) * 100
        else:
            return None

        if distance_pct < self.liq_distance_threshold_pct:
            return f"LIQ_PROTECTION: {distance_pct:.2f}% distance < {self.liq_distance_threshold_pct}% threshold"

        return None

    # ================================================================
    # 🛡️ Check for missing SL/TP (timeout-based)
    # ================================================================
    def _check_missing_sltp(
        self,
        position: Dict[str, Any],
        active_orders: Dict[str, Dict]
    ) -> Optional[str]:
        """
        Returns close reason if position has no SL/TP after timeout.
        Checks both internal tracking and exchange-level SL/TP.
        """
        if not self.enable_missing_sltp_exit:
            return None

        symbol = position.get("symbol")
        if not symbol:
            return None

        # First check if exchange reports SL/TP
        exchange_sl = position.get("stopLoss", "")
        exchange_tp = position.get("takeProfit", "")
        
        # If exchange has SL/TP set, we're good
        if exchange_sl or exchange_tp:
            try:
                if float(exchange_sl or 0) > 0 or float(exchange_tp or 0) > 0:
                    return None
            except (ValueError, TypeError):
                pass

        # Check internal tracking
        for order_id, order_data in active_orders.items():
            if order_data.get("symbol") != symbol:
                continue
                
            sl = order_data.get("stop_loss")
            tp = order_data.get("take_profit")
            sltp_ok = order_data.get("sltp_ok", True)

            # If we have protection and it's confirmed, we're good
            if (sl or tp) and sltp_ok:
                return None

            # Check position age for timeout
            opened_at = order_data.get("opened_at")
            if opened_at:
                age_minutes = (time.time() - opened_at) / 60
                
                # If SL/TP is not set and timeout exceeded
                if not sltp_ok and age_minutes > self.missing_sltp_timeout_minutes:
                    return (
                        f"MISSING_SLTP_TIMEOUT: Position {age_minutes:.1f}m old, "
                        f"SL/TP not confirmed (timeout={self.missing_sltp_timeout_minutes}m)"
                    )

        # Position not in active_orders but on exchange without SL/TP
        # This could be a manually opened position
        return None

    # ================================================================
    # 💸 Check funding cost accumulation
    # ================================================================
    def _check_funding_cost(
        self,
        position: Dict[str, Any],
        active_orders: Dict[str, Dict]
    ) -> Optional[str]:
        """
        Close position if cumulative funding fees eat into profit.
        Estimates funding based on position age.
        """
        if not self.enable_funding_protection:
            return None
            
        symbol = position.get("symbol")
        if not symbol:
            return None

        # Get position age
        order_data = None
        for order_id, data in active_orders.items():
            if data.get("symbol") == symbol:
                order_data = data
                break

        if not order_data:
            return None

        opened_at = order_data.get("opened_at")
        if not opened_at:
            return None

        # Calculate hours held
        age_hours = (time.time() - opened_at) / 3600

        # Estimate funding fees (0.01% per 8 hours is typical, can be positive or negative)
        # For safety, we assume worst case (paying funding)
        funding_intervals = age_hours / 8
        estimated_funding_pct = funding_intervals * 0.01  # 0.01% per interval

        if estimated_funding_pct >= self.max_funding_cost_pct:
            return (
                f"FUNDING_COST: Estimated {estimated_funding_pct:.3f}% funding over "
                f"{age_hours:.1f}h exceeds {self.max_funding_cost_pct}% threshold"
            )

        return None

    # ================================================================
    # ⏱️ Check maximum position age
    # ================================================================
    def _check_max_age(
        self,
        position: Dict[str, Any],
        active_orders: Dict[str, Dict],
        is_momentum: bool = False
    ) -> Optional[str]:
        """
        Close position if it's been open too long without hitting TP.
        Uses different threshold for momentum trades.
        """
        if not self.enable_max_age_exit and not is_momentum:
            return None

        symbol = position.get("symbol")
        if not symbol:
            return None

        # Get position age
        order_data = None
        for order_id, data in active_orders.items():
            if data.get("symbol") == symbol:
                order_data = data
                break

        if not order_data:
            return None

        opened_at = order_data.get("opened_at")
        if not opened_at:
            return None

        age_seconds = time.time() - opened_at

        if is_momentum:
            # Momentum trades have stricter time limits (in minutes)
            age_minutes = age_seconds / 60
            if age_minutes > self.momentum_max_age_minutes:
                return (
                    f"MOMENTUM_AGE_EXIT: {age_minutes:.1f}m > "
                    f"{self.momentum_max_age_minutes}m limit"
                )
        else:
            # Regular trades (in hours)
            if self.enable_max_age_exit:
                age_hours = age_seconds / 3600
                if age_hours > self.max_position_age_hours:
                    return (
                        f"MAX_AGE_EXIT: {age_hours:.1f}h > "
                        f"{self.max_position_age_hours}h limit"
                    )

        return None

    # ================================================================
    # ⏰ Time-based trailing (tighten SL based on position age)
    # ================================================================
    def _check_time_trailing(
        self,
        position: Dict[str, Any],
        active_orders: Dict[str, Dict],
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """
        Returns new SL if time-based tightening is needed, else None.
        
        Returns:
            {"symbol": str, "new_sl": float, "reason": str} or None
        """
        if not self.enable_time_trailing or not self.time_trailing_intervals:
            return None

        symbol = position.get("symbol")
        if not symbol:
            return None

        # Find matching order
        order_data = None
        order_id = None
        for oid, data in active_orders.items():
            if data.get("symbol") == symbol:
                order_data = data
                order_id = oid
                break

        if not order_data:
            return None

        opened_at = order_data.get("opened_at")
        if not opened_at:
            return None

        age_minutes = (time.time() - opened_at) / 60
        entry_price = float(order_data.get("entry_price", 0))
        side = str(order_data.get("side", "")).upper()
        current_sl = float(order_data.get("stop_loss", 0) or 0)

        if entry_price <= 0:
            return None

        # Find applicable interval (highest minutes that we've passed)
        applicable_mult = None
        applicable_minutes = None
        for interval in reversed(self.time_trailing_intervals):
            if age_minutes >= interval["minutes"]:
                applicable_mult = interval["sl_multiplier"]
                applicable_minutes = interval["minutes"]
                break

        if applicable_mult is None:
            return None

        # Calculate ATR estimate (use stored ATR or estimate from entry price)
        atr = order_data.get("atr")
        if not atr or atr <= 0:
            # Fallback: estimate ATR as 2% of entry price
            atr = entry_price * 0.02
        
        new_sl_dist = atr * applicable_mult

        # Calculate new SL
        if side == "BUY":
            new_sl = entry_price - new_sl_dist
            # Only tighten (move SL up), never loosen
            if current_sl > 0 and new_sl > current_sl:
                # Also check that new SL is still below current price
                if new_sl < current_price * 0.995:  # Leave 0.5% buffer
                    return {
                        "symbol": symbol,
                        "new_sl": new_sl,
                        "side": "Buy",
                        "reason": f"TIME_TRAILING: {age_minutes:.1f}m > {applicable_minutes}m → mult={applicable_mult}x ATR"
                    }
        elif side == "SELL":
            new_sl = entry_price + new_sl_dist
            # Only tighten (move SL down), never loosen
            if current_sl > 0 and new_sl < current_sl:
                # Also check that new SL is still above current price
                if new_sl > current_price * 1.005:  # Leave 0.5% buffer
                    return {
                        "symbol": symbol,
                        "new_sl": new_sl,
                        "side": "Sell",
                        "reason": f"TIME_TRAILING: {age_minutes:.1f}m > {applicable_minutes}m → mult={applicable_mult}x ATR"
                    }

        return None

    # ================================================================
    # 🌊 Check account equity drawdown
    # ================================================================
    def _check_equity_drawdown(self, account: Dict[str, Any]) -> Optional[str]:
        """
        Returns close reason if equity has dropped too much from peak.
        """
        if not self.enable_equity_drawdown_exit:
            return None

        if not account:
            return None

        current_equity = float(account.get("equity", 0) or 0)
        if current_equity <= 0:
            return None

        # Track peak equity
        if self.last_equity_peak is None or current_equity > self.last_equity_peak:
            self.last_equity_peak = current_equity
            return None

        # Calculate drawdown from peak
        drawdown_pct = ((self.last_equity_peak - current_equity) / self.last_equity_peak) * 100

        if drawdown_pct > self.max_equity_drawdown_pct:
            return (
                f"EQUITY_DRAWDOWN: {drawdown_pct:.2f}% drawdown from peak "
                f"${self.last_equity_peak:.2f} > {self.max_equity_drawdown_pct}% threshold"
            )

        return None

    # ================================================================
    # 🔒 Check exit cooldown (prevent rapid re-entry)
    # ================================================================
    def _is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in exit cooldown period."""
        last_exit = self.exit_cooldown.get(symbol)
        if last_exit:
            elapsed = time.time() - last_exit
            if elapsed < self.exit_cooldown_seconds:
                return True
        return False

    def _set_cooldown(self, symbol: str) -> None:
        """Set exit cooldown for symbol."""
        self.exit_cooldown[symbol] = time.time()

    # ================================================================
    # 🚨 MAIN EVALUATION - Check all emergency conditions
    # ================================================================
    def evaluate_positions(
        self,
        positions: List[Dict[str, Any]],
        account: Dict[str, Any],
        active_orders: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate all positions for emergency conditions.

        Args:
            positions: List of position dicts from exchange
            account: Account info dict with equity, balance, etc.
            active_orders: Dict of active orders from OrderManager

        Returns:
            {
                "close_all": bool,
                "close_symbols": [symbol1, symbol2, ...],
                "tighten_sl": {symbol: {"new_sl": price, "side": str}, ...},
                "reasons": {symbol: reason_str, ...}
            }
        """
        if active_orders is None:
            active_orders = {}

        result: Dict[str, Any] = {
            "close_all": False,
            "close_symbols": [],
            "tighten_sl": {},
            "reasons": {},
        }

        # 1) Check global equity drawdown
        try:
            equity_reason = self._check_equity_drawdown(account)
            if equity_reason:
                logger.error(f"🚨 {equity_reason} - CLOSING ALL POSITIONS")
                result["close_all"] = True
                result["reasons"]["GLOBAL"] = equity_reason
                return result
        except Exception as e:
            logger.error(f"❌ Error checking equity drawdown: {e}")

        # 2) Check each position individually
        for pos in positions:
            try:
                symbol = pos.get("symbol")
                if not symbol:
                    continue

                # Skip if in cooldown
                if self._is_in_cooldown(symbol):
                    logger.debug(f"⏳ {symbol} in exit cooldown, skipping")
                    continue

                # Get current price
                current_price = self._get_current_price(symbol)
                if not current_price:
                    logger.debug(f"⚠️ Could not get price for {symbol}")
                    continue

                # Check if this is a momentum trade
                is_momentum = False
                for order_id, order_data in active_orders.items():
                    if order_data.get("symbol") == symbol:
                        is_momentum = order_data.get("is_momentum", False)
                        break

                # Collect all reasons for potential exit
                reasons: List[str] = []

                # A) Percentage loss
                pct_reason = self._check_pct_loss(pos, current_price, is_momentum)
                if pct_reason:
                    reasons.append(pct_reason)

                # B) Liquidation distance
                liq_reason = self._check_liquidation_distance(pos, current_price)
                if liq_reason:
                    reasons.append(liq_reason)

                # C) Missing SL/TP timeout
                sltp_reason = self._check_missing_sltp(pos, active_orders)
                if sltp_reason:
                    reasons.append(sltp_reason)

                # D) Funding cost accumulation
                funding_reason = self._check_funding_cost(pos, active_orders)
                if funding_reason:
                    reasons.append(funding_reason)

                # E) Maximum position age
                age_reason = self._check_max_age(pos, active_orders, is_momentum)
                if age_reason:
                    reasons.append(age_reason)

                # F) Time-based trailing (this updates SL, doesn't close)
                trailing_result = self._check_time_trailing(pos, active_orders, current_price)
                if trailing_result:
                    result["tighten_sl"][symbol] = {
                        "new_sl": trailing_result["new_sl"],
                        "side": trailing_result.get("side", pos.get("side", "Buy")),
                    }
                    result["reasons"][f"{symbol}_TIGHTEN"] = trailing_result["reason"]
                    logger.info(
                        f"⏰ Time trailing for {symbol}: new SL = {trailing_result['new_sl']:.8f}"
                    )

                # If any critical reason found → close this position
                if reasons:
                    result["close_symbols"].append(symbol)
                    result["reasons"][symbol] = " | ".join(reasons)
                    logger.warning(f"🚨 Emergency condition on {symbol}: {reasons}")

            except Exception as e:
                logger.error(f"❌ Error evaluating position {pos.get('symbol', 'unknown')}: {e}")
                logger.debug(traceback.format_exc())

        return result

    # ================================================================
    # 🔧 Execute emergency actions (called by bot_main)
    # ================================================================
    def execute_emergency_actions(
        self,
        actions: Dict[str, Any],
        positions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute the emergency actions determined by evaluate_positions.

        Args:
            actions: Result dict from evaluate_positions
            positions: List of position dicts from exchange

        Returns:
            Summary of actions taken
        """
        summary = {
            "closed_all": False,
            "closed_symbols": [],
            "tightened_sl": [],
            "errors": [],
        }

        if not self.order_manager:
            logger.error("❌ No order_manager configured for emergency actions")
            summary["errors"].append("No order_manager available")
            return summary

        try:
            # 1) Close all if needed
            if actions.get("close_all"):
                logger.error("🚨 EMERGENCY: Closing ALL positions")
                reason = actions["reasons"].get("GLOBAL", "Emergency exit")
                
                try:
                    self.order_manager.close_all_positions(reason=reason)
                    summary["closed_all"] = True
                    
                    # Set cooldown for all symbols
                    for pos in positions:
                        symbol = pos.get("symbol")
                        if symbol:
                            self._set_cooldown(symbol)
                            
                except Exception as e:
                    logger.error(f"❌ Failed to close all positions: {e}")
                    summary["errors"].append(f"close_all failed: {e}")
                
                return summary

            # 2) Close specific symbols
            for symbol in actions.get("close_symbols", []):
                reason = actions["reasons"].get(symbol, "Emergency exit")
                logger.warning(f"🚨 Emergency closing {symbol}: {reason}")

                # Find position details
                pos = next((p for p in positions if p.get("symbol") == symbol), None)
                if not pos:
                    logger.warning(f"⚠️ Position not found for {symbol}")
                    continue

                side = str(pos.get("side", "")).capitalize()
                qty = float(pos.get("size", 0) or pos.get("qty", 0))

                if qty <= 0:
                    logger.warning(f"⚠️ No quantity to close for {symbol}")
                    continue

                # Close via order manager or session
                close_side = "Sell" if side == "Buy" else "Buy"
                
                try:
                    # Try using order_manager's close_position_market if available
                    if hasattr(self.order_manager, "close_position_market"):
                        result = self.order_manager.close_position_market(
                            symbol=symbol,
                            side=side,
                            reason=reason
                        )
                    else:
                        # Fallback to direct market order
                        result = self.order_manager.client.place_market_order(
                            symbol=symbol,
                            side=close_side,
                            qty=qty
                        )

                    if result and result.get("status") != "ERROR":
                        logger.success(f"✅ Emergency closed {symbol}")
                        summary["closed_symbols"].append(symbol)
                        self._set_cooldown(symbol)
                    else:
                        error_msg = result.get("error", "Unknown error") if result else "No result"
                        logger.error(f"❌ Failed to emergency close {symbol}: {error_msg}")
                        summary["errors"].append(f"{symbol}: {error_msg}")
                        
                except Exception as e:
                    logger.error(f"❌ Exception closing {symbol}: {e}")
                    summary["errors"].append(f"{symbol}: {e}")

            # 3) Tighten SL where needed
            for symbol, sl_data in actions.get("tighten_sl", {}).items():
                new_sl = sl_data.get("new_sl") if isinstance(sl_data, dict) else sl_data
                side = sl_data.get("side", "Buy") if isinstance(sl_data, dict) else "Buy"
                reason = actions["reasons"].get(f"{symbol}_TIGHTEN", "Time-based tightening")
                
                logger.info(f"🔧 Tightening SL for {symbol} to {new_sl:.8f}: {reason}")

                try:
                    # Use order_manager's SL setter
                    if hasattr(self.order_manager, "_set_position_stops"):
                        success = self.order_manager._set_position_stops(
                            symbol=symbol,
                            stop_loss=new_sl,
                            take_profit=None,  # Keep existing TP
                            side=side
                        )
                    elif self.session and hasattr(self.session, "set_trading_stop"):
                        result = self.session.set_trading_stop(
                            symbol=symbol,
                            side=side,
                            stop_loss=new_sl,
                        )
                        success = result.get("status") == "OK"
                    else:
                        logger.warning(f"⚠️ No method available to set SL for {symbol}")
                        continue

                    if success:
                        logger.success(f"✅ SL tightened for {symbol}")
                        summary["tightened_sl"].append(symbol)
                    else:
                        logger.warning(f"⚠️ Failed to tighten SL for {symbol}")
                        
                except Exception as e:
                    logger.error(f"❌ Failed to tighten SL for {symbol}: {e}")
                    summary["errors"].append(f"{symbol} SL: {e}")

        except Exception as e:
            logger.error(f"❌ Error executing emergency actions: {e}")
            logger.error(traceback.format_exc())
            summary["errors"].append(f"General error: {e}")

        return summary

    # ================================================================
    # 📊 Get current status/stats
    # ================================================================
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the emergency exit manager."""
        return {
            "enabled_checks": {
                "pct_loss_exit": self.enable_pct_loss_exit,
                "liq_protection": self.enable_liq_protection,
                "missing_sltp_exit": self.enable_missing_sltp_exit,
                "time_trailing": self.enable_time_trailing,
                "equity_drawdown_exit": self.enable_equity_drawdown_exit,
                "funding_protection": self.enable_funding_protection,
                "max_age_exit": self.enable_max_age_exit,
            },
            "thresholds": {
                "max_loss_pct": self.max_loss_pct_per_trade,
                "liq_distance_pct": self.liq_distance_threshold_pct,
                "sltp_timeout_min": self.missing_sltp_timeout_minutes,
                "equity_drawdown_pct": self.max_equity_drawdown_pct,
                "max_funding_pct": self.max_funding_cost_pct,
                "max_age_hours": self.max_position_age_hours,
                "momentum_max_loss_pct": self.momentum_max_loss_pct,
                "momentum_max_age_min": self.momentum_max_age_minutes,
            },
            "state": {
                "last_equity_peak": self.last_equity_peak,
                "symbols_in_cooldown": list(self.exit_cooldown.keys()),
            }
        }

    def reset_equity_peak(self, new_peak: Optional[float] = None) -> None:
        """Reset the equity peak tracker."""
        self.last_equity_peak = new_peak
        logger.info(f"📊 Equity peak reset to: {new_peak}")

    def clear_cooldowns(self) -> None:
        """Clear all exit cooldowns."""
        self.exit_cooldown.clear()
        logger.info("🔓 All exit cooldowns cleared")