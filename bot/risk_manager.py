# risk_manager.py
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Callable
from collections import defaultdict
from loguru import logger
from bot.sessions import BybitSession
from config_loader import get_config_loader


class EnhancedRiskManager:
    """
    Enhanced Risk Manager for Bybit - WINNING BOT VERSION
    Only allows high-quality trades with proper risk management.
    Supports auto-flip, position enhancement, and dynamic risk controls.
    """
    def __init__(self, config=None):
        # Load config - either passed in or load from global loader
        if config is not None:
            self.config = config
            logger.debug("✅ Config passed to RiskManager")
        else:
            try:
                from config_loader import get_config_loader
                self.config = get_config_loader()
                logger.debug("✅ Config loaded in RiskManager")
            except Exception as e:
                logger.error(f"❌ Failed to load config in RiskManager: {e}")
                self.config = None

        section = "risk_management"

        # -------------------------------------------------------------
        # POSITION LIMITS
        # -------------------------------------------------------------
        self.max_positions_per_symbol: int = self._get_config(section, "MAX_POSITIONS_PER_SYMBOL", 1, int)
        self.max_total_positions: int = self._get_config(section, "MAX_TOTAL_POSITIONS", 2, int)
        self.allow_sell_positions: bool = self._get_config(section, "ALLOW_SELL_POSITIONS", True, bool)
        self.allow_buy_positions: bool = self._get_config(section, "ALLOW_BUY_POSITIONS", True, bool)
        self.position_cooldown_minutes: int = self._get_config(section, "POSITION_COOLDOWN_MINUTES", 5, int)

        # -------------------------------------------------------------
        # RISK LIMITS
        # -------------------------------------------------------------
        self.max_daily_loss_pct: float = self._get_config(section, "MAX_DAILY_LOSS_PCT", 3.0, float)
        self.max_drawdown_pct: float = self._get_config(section, "MAX_DRAWDOWN_PCT", 5.0, float)
        self.risk_per_trade_pct: float = self._get_config(section, "RISK_PER_TRADE_PCT", 0.5, float)

        # -------------------------------------------------------------
        # POSITION SIZE
        # -------------------------------------------------------------
        self.position_size_usdt: float = self._get_config(section, "POSITION_SIZE_USDT", 10.0, float)
        self.min_position_usdt: float = self._get_config(section, "MIN_POSITION_USDT", 1.0, float)
        self.max_position_usdt: float = self._get_config(section, "MAX_POSITION_USDT", 50.0, float)
        self.use_dynamic_position_limits: bool = self._get_config(section, "USE_DYNAMIC_POSITION_LIMITS", True, bool)
        
        # -------------------------------------------------------------
        # AUTO FLIP CONFIGURATION
        # -------------------------------------------------------------
        self.auto_flip_enabled: bool = self._get_config(section, "AUTO_FLIP_ENABLED", True, bool)
        self.auto_flip_min_loss_pct: float = self._get_config(section, "AUTO_FLIP_MIN_LOSS_PCT", 0.3, float)
        self.auto_flip_min_signal_strength: int = self._get_config(section, "AUTO_FLIP_MIN_SIGNAL_STRENGTH", 85, int)
        self.auto_flip_cooldown_minutes: int = self._get_config(section, "AUTO_FLIP_COOLDOWN_MINUTES", 1, int)
        self.auto_flip_min_movement_pct: float = self._get_config(section, "AUTO_FLIP_MIN_MOVEMENT_PCT", 0.1, float)
        self.auto_enhance_threshold: float = self._get_config(section, "AUTO_ENHANCE_THRESHOLD", 5.0, float)

        if self.auto_flip_enabled:
            logger.info(
                f"🔁 Auto-flip ENABLED (min_loss={self.auto_flip_min_loss_pct}%, "
                f"min_strength={self.auto_flip_min_signal_strength}, "
                f"min_movement={self.auto_flip_min_movement_pct}%)"
            )
        else:
            logger.info("🔁 Auto-flip DISABLED")

        # -------------------------------------------------------------
        # KELLY CRITERION
        # -------------------------------------------------------------
        self.use_kelly_criterion: bool = self._get_config(section, "USE_KELLY_CRITERION", True, bool)
        self.max_kelly_fraction: float = self._get_config(section, "MAX_KELLY_FRACTION", 0.25, float)
        self.min_trades_for_kelly: int = self._get_config(section, "MIN_TRADES_FOR_KELLY", 20, int)

        # -------------------------------------------------------------
        # SIGNAL QUALITY REQUIREMENTS
        # -------------------------------------------------------------
        self.min_signal_strength: int = self._get_config('enhanced_strategy', "MIN_SIGNAL_STRENGTH", 70, int)
        self.min_rr_ratio: float = self._get_config('trading', "MIN_RR_RATIO", 1.5, float)
        self.require_trend_alignment: bool = self._get_config('enhanced_strategy', "REQUIRE_TREND_ALIGNMENT", True, bool)
        self.require_volume_confirmation: bool = self._get_config('enhanced_strategy', "REQUIRE_VOLUME_CONFIRMATION", True, bool)
        self.min_volume_ratio: float = self._get_config('enhanced_strategy', "MIN_VOLUME_RATIO", 1.2, float)

        # -------------------------------------------------------------
        # MARKET CONDITIONS FILTERS
        # -------------------------------------------------------------
        self.max_spread_pct: float = self._get_config(section, "MAX_SPREAD_PCT", 0.1, float)
        self.skip_high_volatility: bool = self._get_config('enhanced_strategy', "SKIP_HIGH_VOLATILITY", True, bool)
        self.max_atr_pct: float = self._get_config('enhanced_strategy', "MAX_ATR_PCT", 5.0, float)
        self.skip_ranging_markets: bool = self._get_config('trend', "SKIP_RANGING_MARKETS", True, bool)

        # -------------------------------------------------------------
        # INTERNAL STATE
        # -------------------------------------------------------------
        self.open_positions: Dict[str, Dict] = {}
        self.positions_by_symbol: Dict[str, int] = defaultdict(int)
        self.last_trade_time: Dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
        self.trade_history: List[Dict[str, Any]] = []
        self.win_streak: int = 0
        self.loss_streak: int = 0
        self.peak_balance: float = 0.0
        self.current_drawdown_pct: float = 0.0

        self.daily_stats: Dict[str, Any] = {
            "start_balance": 0.0,
            "current_balance": 0.0,
            "daily_pnl": 0.0,
            "trades_today": 0,
            "wins_today": 0,
            "losses_today": 0,
            "last_reset": datetime.now(timezone.utc).date(),
        }

        # Track rejected signals for analysis
        self.rejected_signals: List[Dict[str, Any]] = []

        # -------------------------------------------------------------
        # AUTO SAFETY CONFIGURATION
        # -------------------------------------------------------------
        self.enable_auto_safety: bool = self._get_config("risk", "ENABLE_AUTO_SAFETY", True, bool)
        self.max_margin_usage_pct: float = self._get_config("risk", "MAX_MARGIN_USAGE_PCT", 85.0, float)
        self.hard_stop_margin_usage_pct: float = self._get_config("risk", "HARD_STOP_MARGIN_USAGE_PCT", 95.0, float)
        self.emergency_loss_pct: float = self._get_config(section, "EMERGENCY_LOSS_PCT", 5.0, float)
        self.max_consecutive_losses: int = self._get_config(section, "MAX_CONSECUTIVE_LOSSES", 5, int)

        # -------------------------------------------------------------
        # SESSION
        # -------------------------------------------------------------
        self.session = BybitSession()
        self._balance_callback: Optional[Callable[[], Optional[float]]] = None

        logger.success("✅ Enhanced RiskManager initialized (WINNING BOT MODE)")
        
    # =================================================================

    def _get_config(self, section: str, key: str, default: Any, type_func: type = str) -> Any:
        """Helper method to safely get config values."""
        if self.config is None:
            return default
        try:
            return self.config.get(section, key, default, type_func)
        except Exception as e:
            logger.debug(f"⚠️ Failed to get config {section}.{key}: {e}")
            return default
        
    # =================================================================
    # SIGNAL QUALITY VALIDATION - THE KEY TO WINNING!
    # =================================================================
    def validate_signal_quality(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        symbol = signal.get("symbol", "Unknown")
        strength = signal.get("strength", 0)
        side = signal.get("side", "").upper()
        rr_ratio = signal.get("risk_reward", 0)
        
        reasons = []
        warnings = []

        # ---------------------------------------------------------
        # 1. Signal Strength Check (מבוסס INI)
        # ---------------------------------------------------------
        if strength < self.min_signal_strength:
            reasons.append(f"Strength {strength} < {self.min_signal_strength}")

        # ---------------------------------------------------------
        # 2. Risk/Reward Ratio Check
        # ---------------------------------------------------------
        if rr_ratio < self.min_rr_ratio:
            reasons.append(f"R:R ratio too low: {rr_ratio:.2f} < {self.min_rr_ratio}")

        # ---------------------------------------------------------
        # 3. EMA Cross Requirement (REAL cross check)
        # ---------------------------------------------------------
        signal_reasons = signal.get("reasons", [])
        has_ema_cross = any("EMA cross" in str(r) for r in signal_reasons)
        has_weak_ema = any("Weak EMA" in str(r) for r in signal_reasons)

        if has_weak_ema and not has_ema_cross:
            reasons.append("No real EMA cross (only weak EMA convergence)")

        # ---------------------------------------------------------
        # 4. Trend Alignment Check
        # ---------------------------------------------------------
        if self.require_trend_alignment:
            trend = signal.get("trend", "UNKNOWN")
            
            if trend == "RANGING":
                if self.skip_ranging_markets:
                    reasons.append("Market is RANGING - no clear trend")
                else:
                    warnings.append("Trading in RANGING market (higher risk)")
            
            elif side == "BUY" and trend == "DOWNTREND":
                reasons.append(f"BUY signal against DOWNTREND")
            
            elif side == "SELL" and trend == "UPTREND":
                reasons.append(f"SELL signal against UPTREND")

        # ---------------------------------------------------------
        # 5. Volume Confirmation Check
        # ---------------------------------------------------------
        if self.require_volume_confirmation:
            volume_ratio = signal.get("volume_ratio", 0)
            if volume_ratio < self.min_volume_ratio:
                reasons.append(f"Volume too low: {volume_ratio:.2f}x < {self.min_volume_ratio}x")

        # ---------------------------------------------------------
        # 6. RSI Extremes Check
        # ---------------------------------------------------------
        rsi = signal.get("rsi", 50)
        if side == "BUY" and rsi > 65:
            warnings.append(f"RSI high for BUY: {rsi:.1f} (potential overbought)")
        if side == "SELL" and rsi < 35:
            warnings.append(f"RSI low for SELL: {rsi:.1f} (potential oversold)")

        # ---------------------------------------------------------
        # 7. ATR / Volatility Check
        # ---------------------------------------------------------
        if self.skip_high_volatility:
            atr = signal.get("atr", 0)
            price = signal.get("price", 1)
            if price > 0:
                atr_pct = (atr / price) * 100
                if atr_pct > self.max_atr_pct:
                    reasons.append(f"Volatility too high: ATR {atr_pct:.2f}% > {self.max_atr_pct}%")

        # ---------------------------------------------------------
        # 8. Momentum Signal Special Handling
        # ---------------------------------------------------------
        if signal.get("is_momentum", False):
            if strength < 75:
                reasons.append(f"Momentum signal too weak: {strength}")

        # ---------------------------------------------------------
        # Decision Making
        # ---------------------------------------------------------
        approved = len(reasons) == 0

        if not approved:
            # תיעוד דחיות למעקב
            self.rejected_signals.append({
                "symbol": symbol,
                "side": side,
                "strength": strength,
                "reasons": reasons,
                "timestamp": datetime.now(timezone.utc),
            })
            if len(self.rejected_signals) > 50:
                self.rejected_signals.pop(0)

            logger.warning(f"❌ {symbol} signal REJECTED: {', '.join(reasons)}")
        else:
            if warnings:
                logger.info(f"⚠️ {symbol} signal APPROVED with warnings: {', '.join(warnings)}")
            else:
                logger.success(f"✅ {symbol} signal APPROVED (strength={strength}, R:R={rr_ratio:.2f})")

        return {
            "approved": approved,
            "reasons": reasons,
            "warnings": warnings,
            "signal_strength": strength,
            "rr_ratio": rr_ratio,
        }
    
    # =================================================================
    # CAN OPEN TRADE - ENHANCED WITH SIGNAL QUALITY
    # =================================================================
    def can_open_trade(self,symbol: str,size_usdt: float,side: str = "BUY",signal: Optional[Dict[str, Any]] = None,) -> bool:
        """
        Check all conditions before opening a trade.
        Now includes signal quality validation and strict position counting.
        Enforces MAX_TOTAL_POSITIONS even when dynamic limits are disabled.
        """
        try:
            self._reset_daily_if_needed()
            self._update_drawdown()

            side_up = side.upper()

            # ---------------------------------------------------------
            # 1. SIGNAL QUALITY CHECK
            # ---------------------------------------------------------
            if signal:
                quality_check = self.validate_signal_quality(signal)
                if not quality_check["approved"]:
                    logger.debug(f"❌ {symbol} blocked by signal quality filter")
                    return False

            # ---------------------------------------------------------
            # 2. SIZE LIMITS
            # ---------------------------------------------------------
            if size_usdt < self.min_position_usdt:
                logger.warning(
                    f"🚫 {symbol} size too small: ${size_usdt:.2f} < ${self.min_position_usdt:.2f}"
                )
                return False

            if size_usdt > self.max_position_usdt:
                logger.warning(
                    f"🚫 {symbol} size too large: ${size_usdt:.2f} > ${self.max_position_usdt:.2f}"
                )
                return False

            # ---------------------------------------------------------
            # 3. EMERGENCY CONDITIONS
            # ---------------------------------------------------------
            if self.should_close_all_positions():
                logger.warning(f"🚫 {symbol} trade blocked: Emergency conditions active")
                return False

            # ---------------------------------------------------------
            # 4. BUY/SELL PERMISSIONS
            # ---------------------------------------------------------
            if side_up == "SELL" and not self.allow_sell_positions:
                logger.warning(f"🚫 {symbol} SELL blocked (ALLOW_SELL_POSITIONS=false)")
                return False

            if side_up == "BUY" and not self.allow_buy_positions:
                logger.warning(f"🚫 {symbol} BUY blocked (ALLOW_BUY_POSITIONS=false)")
                return False

            # ---------------------------------------------------------
            # 5. PER-SYMBOL LIMIT
            # ---------------------------------------------------------
            current_positions = self.positions_by_symbol.get(symbol, 0)
            if current_positions >= self.max_positions_per_symbol:
                if signal and self.auto_flip_enabled:
                    current_price = signal.get("price", 0)
                    signal_strength = signal.get("strength", 0)
                    if current_price > 0:
                        order_to_flip = self.should_flip_position(symbol, side, signal_strength, current_price)
                        if order_to_flip:
                            logger.warning(
                                f"⚠️ {symbol}: Conflicting position exists — flip required before opening new trade"
                            )
                            return False
                logger.info(
                    f"ℹ️ {symbol} already has {current_positions}/{self.max_positions_per_symbol} positions (no valid flip found)"
                )
                return False

            # ---------------------------------------------------------
            # 6. GLOBAL POSITION LIMIT — FIXED LOGIC
            # ---------------------------------------------------------
            # Calculate actual active positions (only those with remaining_qty > 0)
            active_positions = [
                p for p in self.open_positions.values() 
                if p.get("remaining_qty", 0) > 0
            ]
            actual_position_count = len(active_positions)

            # Validate integrity between open_positions and positions_by_symbol
            counted_by_symbol = sum(self.positions_by_symbol.values())
            if actual_position_count != counted_by_symbol:
                logger.critical(
                    f"🚨 Position count mismatch! "
                    f"Active Orders: {actual_position_count}, "
                    f"Symbol Count: {counted_by_symbol} — "
                    f"TRUSTING ACTIVE ORDERS COUNT."
                )

            # Determine limit to use
            limit_to_use = self.max_total_positions
            current_equity = self._get_balance()

            if self.use_dynamic_position_limits and current_equity > 0:
                if current_equity < 50:
                    limit_to_use = min(self.max_total_positions, 6)
                elif current_equity < 100:
                    limit_to_use = min(self.max_total_positions, 8)
                elif current_equity < 250:
                    limit_to_use = min(self.max_total_positions, 10)
                elif current_equity < 500:
                    limit_to_use = min(self.max_total_positions, 12)
                logger.debug(f"📊 Dynamic position limit applied: {limit_to_use} (equity=${current_equity:.2f})")

            # Enforce limit
            if actual_position_count >= limit_to_use:
                reason = "[Dynamic]" if self.use_dynamic_position_limits else "[Static]"
                logger.info(
                    f"⏭️ Portfolio full ({actual_position_count}/{limit_to_use}) {reason} — blocking {symbol} {side_up}"
                )
                return False

            # ---------------------------------------------------------
            # 7. COOLDOWN CHECK
            # ---------------------------------------------------------
            last_time = self.last_trade_time.get(symbol, datetime.min.replace(tzinfo=timezone.utc))
            delta = datetime.now(timezone.utc) - last_time
            cooldown_seconds = self.position_cooldown_minutes * 60
            remaining = cooldown_seconds - delta.total_seconds()
            if remaining > 0:
                logger.debug(f"⏳ {symbol} in cooldown: {remaining:.0f}s remaining")
                return False

            # ---------------------------------------------------------
            # 8. DAILY LOSS LIMIT
            # ---------------------------------------------------------
            if self.daily_stats["start_balance"] > 0:
                daily_loss_pct = (
                    abs(min(0, self.daily_stats["daily_pnl"]))
                    / self.daily_stats["start_balance"]
                ) * 100
                if daily_loss_pct >= self.max_daily_loss_pct:
                    logger.warning(
                        f"🚫 Daily loss limit reached: {daily_loss_pct:.2f}% >= {self.max_daily_loss_pct:.2f}%"
                    )
                    return False

            # ---------------------------------------------------------
            # 9. LOSS STREAK PROTECTION
            # ---------------------------------------------------------
            if self.loss_streak >= 3:
                logger.warning(
                    f"⚠️ {symbol} on loss streak ({self.loss_streak}) - extra caution applied"
                )

            logger.debug(f"✅ {symbol} {side} trade approved by RiskManager")
            return True

        except Exception as e:
            logger.error(f"❌ Unexpected error in can_open_trade for {symbol}: {e}")
            return False
    # =================================================================
    # FLIP POSITION
    # =================================================================
    def should_flip_position(self, symbol: str, new_side: str, signal_strength: float, current_price: float) -> Optional[str]:
        """
        Enhanced flip/enhance logic.
        Triggers if:
        1. Existing position is opposite AND loss >= AUTO_FLIP_MIN_LOSS_PCT AND strength >= AUTO_FLIP_MIN_SIGNAL_STRENGTH (Classic Flip)
        2. Existing position is same direction BUT new signal is significantly stronger AND price moved enough (Position Enhancement)
        
        Returns order_id of position to close, or None if no action needed.
        """
        if not self.auto_flip_enabled:
            return None

        # Load enhanced config values
        min_loss_pct = getattr(self, 'auto_flip_min_loss_pct', 0.5)
        min_strength = getattr(self, 'auto_flip_min_signal_strength', 85.0)
        min_movement_pct = self.config.get("risk_management", "AUTO_FLIP_MIN_MOVEMENT_PCT", 0.1, float)
        enhance_threshold = self.config.get("risk_management", "AUTO_ENHANCE_THRESHOLD", 10.0, float)  # Strength improvement required

        # Check cooldown
        last_flip_time = self.last_trade_time.get(f"{symbol}_FLIP", datetime.min.replace(tzinfo=timezone.utc))
        delta = datetime.now(timezone.utc) - last_flip_time
        cooldown_seconds = getattr(self, 'auto_flip_cooldown_minutes', 3) * 60
        if delta.total_seconds() < cooldown_seconds:
            remaining = cooldown_seconds - delta.total_seconds()
            logger.debug(f"⏳ {symbol} flip in cooldown: {remaining:.0f}s remaining")
            return None

        # Find existing position
        for order_id, pos in self.open_positions.items():
            if pos.get("symbol") == symbol and pos.get("remaining_qty", 0) > 0:
                existing_side = pos.get("side", "").upper()
                entry_price = pos.get("entry_price", 0)
                if entry_price <= 0:
                    continue

                # Calculate PnL percentage
                if existing_side == "BUY":
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                else:  # SELL
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100

                # Calculate price movement percentage
                movement_pct = abs((current_price - entry_price) / entry_price) * 100

                # Get existing signal strength
                existing_strength = pos.get("signal_strength", 0)

                # ===========================================================
                # CASE 1: Classic FLIP (opposite direction + loss + strong signal)
                # ===========================================================
                if existing_side != new_side.upper():
                    if pnl_pct <= -min_loss_pct and signal_strength >= min_strength:
                        logger.info(
                            f"🔁 CLASSIC FLIP: {symbol} from {existing_side} to {new_side} "
                            f"(PnL={pnl_pct:.2f}%, Strength={signal_strength})"
                        )
                        return order_id

                # ===========================================================
                # CASE 2: POSITION ENHANCEMENT (same direction + better setup)
                # ===========================================================
                elif (
                    signal_strength >= min_strength
                    and (signal_strength - existing_strength) >= enhance_threshold
                    and movement_pct >= min_movement_pct
                    and abs(pnl_pct) < 5.0  # Don't enhance if already winning big
                ):
                    logger.info(
                        f"✨ ENHANCE POSITION: {symbol} {existing_side} @ {entry_price:.6f} → "
                        f"{new_side} @ {current_price:.6f} | "
                        f"Strength: {existing_strength} → {signal_strength} (+{signal_strength - existing_strength:.1f}) | "
                        f"Movement: {movement_pct:.2f}% | PnL: {pnl_pct:+.2f}%"
                    )
                    return order_id

        return None

    def mark_flip_executed(self, symbol: str) -> None:
        """Mark that a flip/enhancement was executed for cooldown tracking."""
        self.last_trade_time[f"{symbol}_FLIP"] = datetime.now(timezone.utc)

    # =================================================================
    # FEE CHECK
    # =================================================================
    def should_skip_trade_due_to_fees(self, symbol: str, entry_price: float, tp_price: float) -> bool:
        """
        Enhanced fee check: require profit margin to be at least 3x total fees.
        """
        try:
            config = get_config_loader()
            
            # Get fees from config
            taker_fee_pct = config.get("fees", "TAKER_FEE_PCT", 0.06, float) / 100.0
            slippage_pct = config.get("fees", "EXPECTED_SLIPPAGE_PCT", 0.05, float) / 100.0
            min_profit_margin_pct = config.get("fees", "MIN_PROFIT_MARGIN_PCT", 0.50, float) / 100.0

            # Total cost per trade (entry + exit)
            total_fee_pct = (taker_fee_pct * 2) + slippage_pct  # Entry + Exit + Slippage
            required_min_gain_pct = total_fee_pct * 3  # Require 3x coverage

            # Calculate actual potential gain %
            if entry_price == 0:
                return True

            potential_gain_pct = abs(tp_price - entry_price) / entry_price

            # Log for transparency
            logger.debug(
                f"📊 {symbol} Fee Analysis → "
                f"Total Fees: {total_fee_pct*100:.2f}%, "
                f"Required Min Gain: {required_min_gain_pct*100:.2f}%, "
                f"Actual Potential Gain: {potential_gain_pct*100:.2f}%"
            )

            # Block if gain doesn't cover 3x fees
            if potential_gain_pct < required_min_gain_pct:
                logger.warning(
                    f"🚫 {symbol} blocked by fees: gain={potential_gain_pct*100:.2f}% < 3x fees ({required_min_gain_pct*100:.2f}%)"
                )
                return True

            # Also enforce absolute minimum profit margin (e.g., 0.5%)
            if potential_gain_pct < min_profit_margin_pct:
                logger.warning(
                    f"🚫 {symbol} blocked by min profit margin: gain={potential_gain_pct*100:.2f}% < {min_profit_margin_pct*100:.2f}%"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"❌ Fee check failed for {symbol}: {e}")
            return True  # Fail-safe: block trade if error

    # =================================================================
    # BALANCE CALLBACK
    # =================================================================
    def set_balance_callback(self, cb: Callable[[], Optional[float]]) -> None:
        self._balance_callback = cb

    def _get_balance(self) -> float:
        if self._balance_callback:
            try:
                b = self._balance_callback()
                if b is not None:
                    return float(b)
            except Exception as e:
                logger.debug(f"⚠️ Balance callback failed: {e}")
        return 0.0

    # =================================================================
    # DAILY RESET
    # =================================================================
    def _reset_daily_if_needed(self) -> None:
        now = datetime.now(timezone.utc).date()
        if now != self.daily_stats["last_reset"]:
            bal = self._get_balance()
            self.daily_stats = {
                "start_balance": bal,
                "current_balance": bal,
                "daily_pnl": 0.0,
                "trades_today": 0,
                "wins_today": 0,
                "losses_today": 0,
                "last_reset": now,
            }
            self.win_streak = 0
            self.loss_streak = 0
            self.peak_balance = bal
            self.current_drawdown_pct = 0.0
            self.rejected_signals = []  # Reset rejected signals
            logger.info("🔄 Daily stats reset")

    # =================================================================
    # KELLY CRITERION
    # =================================================================
    def _calculate_kelly_fraction(self) -> float:
        if not self.use_kelly_criterion:
            return 1.0

        if len(self.trade_history) < self.min_trades_for_kelly:
            return 1.0

        recent_trades = self.trade_history[-50:]
        wins = [t for t in recent_trades if t.get("pnl", 0) > 0]
        losses = [t for t in recent_trades if t.get("pnl", 0) < 0]

        if not wins or not losses:
            return 1.0

        p = len(wins) / len(recent_trades)
        q = 1 - p

        avg_win = sum(t.get("pnl", 0) for t in wins) / len(wins)
        avg_loss = abs(sum(t.get("pnl", 0) for t in losses) / len(losses))

        if avg_loss == 0:
            return 1.0

        b = avg_win / avg_loss
        kelly = (p * b - q) / b
        kelly = kelly * 0.5  # Half-Kelly for safety
        kelly = max(0.1, min(kelly, self.max_kelly_fraction))

        return kelly

    # =================================================================
    # ADAPTIVE POSITION SIZE
    # =================================================================
    def get_adaptive_position_size(self, symbol: str, signal: Optional[Dict] = None) -> float:
        """Calculate adaptive position size based on conditions and signal quality."""
        self._reset_daily_if_needed()
        self._update_drawdown()

        # 🔥 NEW: Scale base size based on account equity for better capital utilization
        current_equity = self._get_balance()
        if current_equity > 0:
            # Reduce size if equity is very low (under $50)
            if current_equity < 50:
                base_multiplier = 0.8
            elif current_equity < 100:
                base_multiplier = 0.9
            else:
                base_multiplier = 1.0
        else:
            base_multiplier = 1.0

        base_size = self.position_size_usdt * base_multiplier

        # Apply Kelly Criterion
        kelly_factor = self._calculate_kelly_fraction()
        size = base_size * kelly_factor

        # Loss streak reduction (exponential)
        if self.loss_streak >= 3:
            streak_factor = 0.85 ** (self.loss_streak - 2)
            size *= streak_factor
            logger.debug(f"📉 Loss streak {self.loss_streak}: size factor={streak_factor:.2%}")

        # Win streak bonus (capped)
        if self.win_streak >= 3:
            streak_factor = min(1.2, 1.0 + (self.win_streak - 2) * 0.05)
            size *= streak_factor

        # Drawdown reduction
        if self.current_drawdown_pct > 1.0:
            dd_factor = max(0.3, 0.9 ** self.current_drawdown_pct)
            size *= dd_factor

        # 🆕 Signal quality adjustment
        if signal:
            strength = signal.get("strength", 50)
            if strength >= 90:
                size *= 1.15  # Boost for excellent signals
            elif strength >= 80:
                size *= 1.05
            elif strength < 60:
                size *= 0.8  # Reduce for weaker signals

            # R:R adjustment
            rr = signal.get("risk_reward", 1.5)
            if rr >= 3.0:
                size *= 1.1
            elif rr < 1.5:
                size *= 0.85

        # Enforce limits
        size = max(self.min_position_usdt, min(size, self.max_position_usdt))

        logger.debug(
            f"🎯 Adaptive size for {symbol}: ${size:.2f} (base=${base_size:.2f}, kelly={kelly_factor:.2%})"
        )

        return size

    # =================================================================
    # AUTO SAFETY
    # =================================================================
    def _compute_margin_usage_pct(self, account: Dict[str, Any]) -> Optional[float]:
        if not account:
            return None

        equity = account.get("equity") or account.get("total_equity") or account.get("wallet_balance")
        used_margin = account.get("margin") or account.get("used_margin") or account.get("position_margin")

        try:
            equity = float(equity or 0)
            used_margin = float(used_margin or 0)
        except (ValueError, TypeError):
            return None

        if equity <= 0:
            return None

        return (used_margin / equity) * 100.0

    def check_global_safety(self, account: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enable_auto_safety:
            return {
                "allow_new_trades": True,
                "force_reduce": False,
                "margin_usage_pct": None,
                "reason": "AutoSafety disabled",
            }

        mu = self._compute_margin_usage_pct(account)
        if mu is None:
            return {
                "allow_new_trades": True,
                "force_reduce": False,
                "margin_usage_pct": None,
                "reason": "No margin usage info",
            }

        if mu >= self.hard_stop_margin_usage_pct:
            logger.error(
                f"🛑 AutoSafety HARD STOP – margin usage={mu:.1f}% (limit={self.hard_stop_margin_usage_pct:.1f}%)"
            )
            return {
                "allow_new_trades": False,
                "force_reduce": True,
                "margin_usage_pct": mu,
                "reason": "HARD_STOP_MARGIN_LIMIT",
            }

        if mu >= self.max_margin_usage_pct:
            logger.warning(
                f"⚠️ AutoSafety SOFT BLOCK – margin usage={mu:.1f}% (limit={self.max_margin_usage_pct:.1f}%)"
            )
            return {
                "allow_new_trades": False,
                "force_reduce": False,
                "margin_usage_pct": mu,
                "reason": "SOFT_MARGIN_LIMIT",
            }

        return {
            "allow_new_trades": True,
            "force_reduce": False,
            "margin_usage_pct": mu,
            "reason": "OK",
        }

    # =================================================================
    # EMERGENCY CONDITIONS
    # =================================================================
    def should_close_all_positions(self) -> bool:
        self._reset_daily_if_needed()
        
        current_pnl = self.daily_stats["daily_pnl"]
        
        if current_pnl <= -5.0:
            logger.error(f"🚨 EMERGENCY: Daily loss limit reached (${current_pnl})")
            return True

        if self.loss_streak >= self.max_consecutive_losses:
            return True

        return False

    # =================================================================
    # POSITION TRACKING
    # =================================================================
    def track_open(
        self,
        order_id: str,
        symbol: str,
        quantity: float,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        self.open_positions[order_id] = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "initial_qty": quantity,
            "remaining_qty": quantity,
            "realized_pnl": 0.0,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "opened_at": datetime.now(timezone.utc),
        }
        self.positions_by_symbol[symbol] = self.positions_by_symbol.get(symbol, 0) + 1
        self.last_trade_time[symbol] = datetime.now(timezone.utc)
        self.daily_stats["trades_today"] += 1

        logger.success(f"📘 Tracking OPEN {symbol} ({side}) qty={quantity}, entry={entry_price}")

    def track_close(self, order_id: str, exit_price: float, pnl: float) -> None:
        pos = self.open_positions.get(order_id)

        if pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
            self.daily_stats["wins_today"] += 1
        elif pnl < 0:
            self.loss_streak += 1
            self.win_streak = 0
            self.daily_stats["losses_today"] += 1

        self.daily_stats["daily_pnl"] += pnl
        self.daily_stats["current_balance"] = self._get_balance()

        self.trade_history.append({
            "pnl": pnl,
            "exit_price": exit_price,
            "closed_at": datetime.now(timezone.utc),
            "order_id": order_id,
        })

        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]

        if pos:
            pos["realized_pnl"] += pnl
            pos["remaining_qty"] = 0
            symbol = pos["symbol"]

            if symbol in self.positions_by_symbol:
                self.positions_by_symbol[symbol] = max(0, self.positions_by_symbol[symbol] - 1)

            logger.success(f"📕 Tracking CLOSE {symbol} pnl={pnl:.2f}, exit={exit_price}")
            del self.open_positions[order_id]

        self._update_drawdown()

    def _update_drawdown(self) -> None:
        current_balance = self._get_balance()
        if current_balance <= 0:
            return

        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
            self.current_drawdown_pct = 0.0
        elif self.peak_balance > 0:
            self.current_drawdown_pct = ((self.peak_balance - current_balance) / self.peak_balance) * 100

    def sync_positions_from_exchange(self, positions: List[Dict[str, Any]]) -> None:
        self.positions_by_symbol = defaultdict(int)
        if not positions:
            return
        for p in positions:
            symbol = p.get("symbol")
            if not symbol:
                continue
            qty = float(p.get("size") or p.get("qty") or 0)
            if qty > 0:
                self.positions_by_symbol[symbol] += 1

    def update_from_order_manager(self, order_manager) -> None:
        if not order_manager:
            return

        active_orders = getattr(order_manager, "active_orders", {})

        new_positions: Dict[str, int] = defaultdict(int)
        for order_id, pos_data in active_orders.items():
            symbol = pos_data.get("symbol")
            if symbol:
                remaining_qty = float(pos_data.get("remaining_qty", 0))
                if remaining_qty > 0:
                    new_positions[symbol] += 1

        self.positions_by_symbol = new_positions

    def update_position_count(self, count: int) -> None:
        logger.success(f"📊 Position count updated: {count}")

    # =================================================================
    # EXCHANGE MINIMUM CHECK
    # =================================================================
    def is_above_exchange_minimum(self, symbol: str, quote_value_usdt: float) -> bool:
        try:
            limits = self.session.get_symbol_limits(symbol)
            if not limits:
                return True
            min_notional = float(limits.get("min_notional") or 0)
            if quote_value_usdt < min_notional:
                logger.error(
                    f"❌ {symbol} below minNotional {quote_value_usdt} < {min_notional}"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"❌ minNotional check failed for {symbol}: {e}")
            return True

    # =================================================================
    # POSITION SIZE (LEGACY)
    # =================================================================
    def get_position_size_usdt(self, signal_strength: float) -> float:
        size = self.position_size_usdt
        if signal_strength >= 80:
            size *= 1.25
        elif signal_strength <= 40:
            size *= 0.8
        size = max(self.min_position_usdt, min(size, self.max_position_usdt))
        return size

    # =================================================================
    # REPORTS
    # =================================================================
    def get_risk_report(self) -> Dict[str, Any]:
        self._reset_daily_if_needed()

        total_positions = sum(self.positions_by_symbol.values())
        total_trades = self.daily_stats["wins_today"] + self.daily_stats["losses_today"]
        win_rate = 0.0
        if total_trades > 0:
            win_rate = (self.daily_stats["wins_today"] / total_trades) * 100

        daily_pnl_pct = 0.0
        if self.daily_stats["start_balance"] > 0:
            daily_pnl_pct = (self.daily_stats["daily_pnl"] / self.daily_stats["start_balance"]) * 100

        current_equity = self._get_balance()
        dynamic_max_positions = self.max_total_positions

        if self.use_dynamic_position_limits and current_equity > 0:
            if current_equity < 50:
                dynamic_max_positions = min(self.max_total_positions, 6)
            elif current_equity < 100:
                dynamic_max_positions = min(self.max_total_positions, 8)
            elif current_equity < 250:
                dynamic_max_positions = min(self.max_total_positions, 10)
            elif current_equity < 500:
                dynamic_max_positions = min(self.max_total_positions, 12)

        return {
            "total_positions": total_positions,
            "positions_by_symbol": dict(self.positions_by_symbol),
            "max_positions": self.max_total_positions,
            "dynamic_max_positions": dynamic_max_positions,
            "daily_pnl": self.daily_stats["daily_pnl"],
            "daily_pnl_pct": daily_pnl_pct,
            "trades_today": self.daily_stats["trades_today"],
            "wins_today": self.daily_stats["wins_today"],
            "losses_today": self.daily_stats["losses_today"],
            "win_rate": win_rate,
            "win_streak": self.win_streak,
            "loss_streak": self.loss_streak,
            "current_drawdown_pct": self.current_drawdown_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "peak_balance": self.peak_balance,
            "kelly_fraction": self._calculate_kelly_fraction() if self.use_kelly_criterion else None,
            "emergency_active": self.should_close_all_positions(),
            "rejected_signals_today": len(self.rejected_signals),
        }

    def get_risk_status(self) -> Dict[str, Any]:
        total_positions = sum(self.positions_by_symbol.values())
        return {
            "total_positions": total_positions,
            "positions_by_symbol": dict(self.positions_by_symbol),
            "win_streak": self.win_streak,
            "loss_streak": self.loss_streak,
            "daily_stats": self.daily_stats.copy(),
            "current_drawdown_pct": self.current_drawdown_pct,
        }

    def close_position_for_symbol(self, symbol: str) -> Optional[str]:
        for order_id, pos in self.open_positions.items():
            if pos.get("symbol") == symbol and pos.get("remaining_qty", 0) > 0:
                return order_id
        return None

# Alias for backwards compatibility
RiskManager = EnhancedRiskManager