# trade_strategy.py
"""
EnhancedBybitStrategy — Institutional Grade Trading Engine
Combines EMA/RSI/Trend/Volume with Momentum Detection & Dynamic Risk Management.
Fully configurable via strategy_config.ini.
"""
import time
import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional, Dict, List, Any
from config_loader import get_config_loader


class EnhancedBybitStrategy:
    """
    Professional Multi-Layer Trading Strategy with:
    - EMA Cross + RSI Filtering
    - Dynamic Trend Detection
    - Momentum Burst Capture
    - Volume-Weighted Confirmation
    - Adaptive Stop Loss / Take Profit
    """

    def __init__(self, client):
        self.client = client

        # Initialize ALL attributes to safe defaults first
        self._initialize_attributes()

        # Load configuration from INI
        self._load_strategy_config()

        # Calculate dynamic candles per day based on interval
        self.candles_per_day = self._calculate_candles_per_day()

        logger.success("✅ EnhancedBybitStrategy initialized successfully")

    # =====================================================================
    # ATTRIBUTE INITIALIZATION (Safe Defaults)
    # =====================================================================
    def _initialize_attributes(self):
        """Initialize all attributes to prevent AttributeError during loading."""
        # Core
        self.symbols = []
        self.momentum_symbols = []

        # Technicals
        self.ema_fast = 9
        self.ema_slow = 21
        self.ema_long = 50
        self.rsi_period = 14
        self.atr_period = 14
        self.volume_ma_period = 20
        self.klines_interval = "15"
        self.klines_limit = 200
        self.min_candles_required = 100

        # Filters
        self.detect_ranging = True
        self.skip_ranging_markets = True
        self.ranging_atr_threshold = 0.15
        self.min_trend_strength = 0.25
        self.allow_buy_signals = True
        self.allow_sell_signals = True
        self.enable_range_scalping = False
        self.enable_breakout_filter = True
        self.breakout_lookback_bars = 20
        self.breakout_min_pct = 0.1

        # Scoring
        self.min_signal_strength = 70.0
        self.strictness_mode = "MEDIUM_STRICT"
        self.buy_strictness = 0.20
        self.sell_strictness = 0.28
        self.ema_weight = 30
        self.rsi_weight = 20
        self.trend_weight = 25
        self.volume_weight = 15
        self.trend_strength_weight = 10

        # Entry Rules
        self.require_ema_cross_for_entry = False
        self.enable_weak_ema_entry = True
        self.weak_ema_tolerance = 0.05
        self.enable_rsi_filter = True
        self.enable_volume_filter = True
        self.min_volume_ratio = 1.5
        self.rsi_overbought = 70
        self.rsi_oversold = 30

        # Strategy Style
        self.strategy_style = "HYBRID"

        # Risk
        self.sl_atr_multiplier = 2.0
        self.tp_atr_multiplier = 4.0

        # Momentum
        self.momentum_enabled = False
        self.min_momentum_pct = 20.0
        self.max_momentum_pct = 50.0
        self.min_volume_usdt = 2_000_000
        self.momentum_scan_interval = 5
        self.momentum_rsi_threshold = 65
        self.momentum_rsi_cap = 80
        self.momentum_volume_ratio = 2.0

        # Runtime
        self.last_momentum_scan: Dict[str, float] = {}

        # ML Settings (defaults)
        self.ml_blend_weight = 0.7
        self.confidence_threshold = 0.6

        self.open_positions = {}

    # =====================================================================
    # CONFIG LOADING
    # =====================================================================
    def _load_strategy_config(self):
        """Load and validate all strategy parameters from config."""
        try:
            config = get_config_loader()

            # Core Symbols
            symbols_raw = config.get("trading", "SYMBOLS", "BTCUSDT,ETHUSDT", str)
            self.symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]

            momentum_raw = config.get("momentum_trading", "MOMENTUM_SYMBOLS", "", str)
            self.momentum_symbols = (
                [s.strip().upper() for s in momentum_raw.split(",") if s.strip()]
                if momentum_raw
                else self.symbols.copy()
            )

            # Technical Parameters
            self.ema_fast = config.get("enhanced_strategy", "EMA_FAST", 9, int)
            self.ema_slow = config.get("enhanced_strategy", "EMA_SLOW", 21, int)
            self.ema_long = config.get("enhanced_strategy", "EMA_LONG", 50, int)
            self.rsi_period = config.get("enhanced_strategy", "RSI_PERIOD", 14, int)
            self.atr_period = config.get("enhanced_strategy", "ATR_PERIOD", 14, int)
            self.volume_ma_period = config.get("enhanced_strategy", "VOLUME_MA_PERIOD", 20, int)
            self.klines_interval = str(config.get("enhanced_strategy", "KLINES_INTERVAL", "15"))
            self.klines_limit = config.get("enhanced_strategy", "KLINES_LIMIT", 200, int)
            self.min_candles_required = config.get("enhanced_strategy", "MIN_CANDLES_REQUIRED", 100, int)

            # Filters
            self.detect_ranging = config.get("trend", "DETECT_RANGING", True, bool)
            self.skip_ranging_markets = config.get("trend", "SKIP_RANGING_MARKETS", True, bool)
            self.ranging_atr_threshold = config.get("trend", "RANGING_ATR_THRESHOLD", 0.15, float)
            self.min_trend_strength = config.get("trend", "MIN_TREND_STRENGTH", 0.25, float)
            self.allow_buy_signals = config.get("strategy_filters", "ALLOW_BUY_SIGNALS", True, bool)
            self.allow_sell_signals = config.get("strategy_filters", "ALLOW_SELL_SIGNALS", True, bool)
            self.enable_range_scalping = config.get("strategy_filters", "ENABLE_RANGE_SCALPING", False, bool)
            self.enable_breakout_filter = config.get("enhanced_strategy", "ENABLE_BREAKOUT_FILTER", True, bool)
            self.breakout_lookback_bars = config.get("enhanced_strategy", "BREAKOUT_LOOKBACK_BARS", 20, int)
            self.breakout_min_pct = config.get("enhanced_strategy", "BREAKOUT_MIN_PCT", 0.1, float)

            # Scoring & Strictness
            self.min_signal_strength = config.get("enhanced_strategy", "MIN_SIGNAL_STRENGTH", 70.0, float)
            self.strictness_mode = config.get("enhanced_strategy", "STRICTNESS_MODE", "MEDIUM_STRICT", str).upper()
            self.buy_strictness = config.get("enhanced_strategy", "BUY_STRICTNESS", 0.20, float)
            self.sell_strictness = config.get("enhanced_strategy", "SELL_STRICTNESS", 0.28, float)
            self.ema_weight = config.get("enhanced_strategy", "EMA_WEIGHT", 30, int)
            self.rsi_weight = config.get("enhanced_strategy", "RSI_WEIGHT", 20, int)
            self.trend_weight = config.get("enhanced_strategy", "TREND_WEIGHT", 25, int)
            self.volume_weight = config.get("enhanced_strategy", "VOLUME_WEIGHT", 15, int)
            self.trend_strength_weight = config.get("enhanced_strategy", "TREND_STRENGTH_WEIGHT", 10, int)

            # Entry Rules
            self.require_ema_cross_for_entry = config.get("enhanced_strategy", "REQUIRE_EMA_CROSS_FOR_ENTRY", False, bool)
            self.enable_weak_ema_entry = config.get("enhanced_strategy", "ENABLE_WEAK_EMA_ENTRY", True, bool)
            self.weak_ema_tolerance = config.get("enhanced_strategy", "WEAK_EMA_TOLERANCE", 0.05, float)
            self.enable_rsi_filter = config.get("enhanced_strategy", "ENABLE_RSI_FILTER", True, bool)
            self.enable_volume_filter = config.get("enhanced_strategy", "ENABLE_VOLUME_FILTER", True, bool)
            self.min_volume_ratio = config.get("enhanced_strategy", "MIN_VOLUME_RATIO", 1.5, float)
            self.rsi_overbought = config.get("enhanced_strategy", "RSI_OVERBOUGHT", 70, int)
            self.rsi_oversold = config.get("enhanced_strategy", "RSI_OVERSOLD", 30, int)

            # Strategy Style
            self.strategy_style = config.get("enhanced_strategy", "STRATEGY_STYLE", "HYBRID", str).upper()

            # Risk Management
            self.sl_atr_multiplier = config.get("risk_management", "SL_ATR_MULTIPLIER", 2.0, float)
            self.tp_atr_multiplier = config.get("risk_management", "TP_ATR_MULTIPLIER", 4.0, float)

            # Momentum Trading
            self.momentum_enabled = config.get("momentum_trading", "ENABLED", False, bool)
            self.min_momentum_pct = config.get("momentum_trading", "MIN_MOMENTUM_PCT", 20.0, float)
            self.max_momentum_pct = config.get("enhanced_strategy", "MAX_MOMENTUM_PCT", 50.0, float)
            self.min_volume_usdt = config.get("momentum_trading", "MIN_VOLUME_USDT", 2_000_000, float)
            self.momentum_scan_interval = config.get("momentum_trading", "MOMENTUM_SCAN_INTERVAL", 5, int)
            self.momentum_rsi_threshold = config.get("momentum_trading", "MOMENTUM_RSI_THRESHOLD", 65, float)
            self.momentum_rsi_cap = config.get("enhanced_strategy", "MOMENTUM_RSI_CAP", 80, float)
            self.momentum_volume_ratio = config.get("momentum_trading", "MOMENTUM_VOLUME_RATIO", 2.0, float)

            logger.info(f"📈 Loaded {len(self.symbols)} trading symbols")
        
            if self.momentum_enabled:
                logger.info(f"⚡ Momentum scanning {len(self.momentum_symbols)} symbols")

            try:
                self.ml_blend_weight = config.getfloat("machine_learning", "ML_BLEND_WEIGHT", 0.7)
                self.confidence_threshold = config.getfloat("machine_learning", "CONFIDENCE_THRESHOLD", 0.6)
                logger.debug(f"🧠 ML Settings loaded: blend_weight={self.ml_blend_weight}, confidence_threshold={self.confidence_threshold}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load ML settings: {e}")
                self.ml_blend_weight = 0.7
                self.confidence_threshold = 0.6
                
        except Exception as e:
            logger.error(f"❌ Failed to load strategy config: {e}")

    # =====================================================================
    # UTILITIES
    # =====================================================================
    def _calculate_candles_per_day(self) -> int:
        """Calculate candles per 24h based on kline interval."""
        interval_map = {
            "1": 1440, "3": 480, "5": 288, "15": 96,
            "30": 48, "60": 24, "120": 12, "240": 6,
            "360": 4, "720": 2, "D": 1, "W": 1, "M": 1
        }
        candles = interval_map.get(self.klines_interval, 96)
        logger.debug(f"📊 Interval {self.klines_interval} → {candles} candles/day")
        return candles

    def _effective_min_strength(self, side: str) -> float:
        """Apply side-specific strictness adjustment."""
        base = self.min_signal_strength
        if side == "BUY":
            return base * (1 + self.buy_strictness)
        elif side == "SELL":
            return base * (1 + self.sell_strictness)
        return base

    # =====================================================================
    # INDICATORS
    # =====================================================================
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators with professional-grade formulas."""
        try:
            df = df.copy()

            # EMAs
            df["ema_fast"] = df["close"].ewm(span=self.ema_fast, adjust=False).mean()
            df["ema_slow"] = df["close"].ewm(span=self.ema_slow, adjust=False).mean()
            df["ema_long"] = df["close"].ewm(span=self.ema_long, adjust=False).mean()

            # RSI (Exponential Smoothing)
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.ewm(alpha=1/self.rsi_period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/self.rsi_period, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            df["rsi"] = 100 - (100 / (1 + rs))

            # ATR
            prev_close = df["close"].shift(1)
            tr1 = df["high"] - df["low"]
            tr2 = (df["high"] - prev_close).abs()
            tr3 = (df["low"] - prev_close).abs()
            df["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df["atr"] = df["true_range"].rolling(window=self.atr_period, min_periods=1).mean()

            # Volume
            df["volume_ma"] = df["volume"].rolling(window=self.volume_ma_period, min_periods=1).mean()
            df["volume_ratio"] = df["volume"] / (df["volume_ma"] + 1e-10)

            # Trend Strength
            df["trend_strength"] = abs(df["ema_fast"] - df["ema_slow"]) / df["close"] * 100

            return df.dropna()

        except Exception as e:
            logger.error(f"📉 Indicator calculation failed: {e}")
            return df

    def detect_trend(self, df: pd.DataFrame) -> str:
        """Detect market trend: UPTREND, DOWNTREND, RANGING."""
        last = df.iloc[-1]
        if last["ema_fast"] > last["ema_slow"] > last["ema_long"]:
            return "UPTREND"
        elif last["ema_fast"] < last["ema_slow"] < last["ema_long"]:
            return "DOWNTREND"
        else:
            return "RANGING"

    # =====================================================================
    # SIGNAL SCORING ENGINE
    # =====================================================================
    def calculate_signal_strength(self, df: pd.DataFrame, signal_type: str, symbol: str) -> float:
        """
        Institutional Scoring Model (0-100):
        Layer 1: Trend Alignment (40%)
        Layer 2: Momentum Quality (30%)
        Layer 3: Volume Confirmation (20%)
        Layer 4: Volatility Health (10%)
        """
        last = df.iloc[-1]
        trend = self.detect_trend(df)

        # Layer 1: Trend (Max 40)
        trend_score = 0
        if (signal_type == "BUY" and trend == "UPTREND") or (signal_type == "SELL" and trend == "DOWNTREND"):
            trend_score = 40
        elif trend == "RANGING":
            trend_score = 15

        # Layer 2: Momentum/RSI (Max 30)
        mom_score = 0
        rsi = last["rsi"]
        if signal_type == "BUY":
            if 30 <= rsi <= 55:   mom_score = 30
            elif rsi < 30:        mom_score = 20
            elif 55 < rsi <= 65:  mom_score = 10
            else:                 mom_score = -20  # Penalty
        else:  # SELL
            if 45 <= rsi <= 70:   mom_score = 30
            elif rsi > 70:        mom_score = 20
            elif 35 <= rsi < 45:  mom_score = 10
            else:                 mom_score = -20

        # Layer 3: Volume (Max 20)
        vol_score = 0
        v_ratio = last["volume_ratio"]
        if v_ratio >= 2.5:        vol_score = 20
        elif v_ratio >= 1.5:      vol_score = 15
        elif v_ratio >= 1.0:      vol_score = 5
        else:                     vol_score = -10

        # Layer 4: Volatility (Max 10)
        vola_score = 0
        ema_dist = abs(last["close"] - last["ema_slow"]) / last["ema_slow"]
        if ema_dist < 0.05:       vola_score = 10

        total = trend_score + mom_score + vol_score + vola_score
        final = max(0, min(100, total))

        logger.debug(
            f"[{symbol}] SCORE: {final} | "
            f"Trend:{trend_score}, Mom:{mom_score}, Vol:{vol_score}, Vola:{vola_score}"
        )
        return float(final)

    # =====================================================================
    # MOMENTUM TRADING
    # =====================================================================
    def detect_momentum(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Detect high-momentum breakout opportunities."""
        if not self.momentum_enabled or symbol not in self.momentum_symbols:
            return None

        if len(df) < max(100, self.candles_per_day):
            return None

        try:
            close = df["close"].astype(float)
            volume = df["volume"].astype(float)
            turnover = df["turnover"].astype(float) if "turnover" in df.columns else None

            lookback = min(len(close) - 1, self.candles_per_day)
            price_24h_ago = close.iloc[-lookback]
            current_price = close.iloc[-1]
            price_change_pct = ((current_price - price_24h_ago) / price_24h_ago) * 100

            # Reject extremes
            if abs(price_change_pct) > self.max_momentum_pct:
                logger.debug(f"🚫 {symbol} momentum rejected: {price_change_pct:+.1f}% > {self.max_momentum_pct}%")
                return None

            # Volume check
            volume_24h_usdt = turnover.iloc[-lookback:].sum() if turnover is not None else 0
            if volume_24h_usdt < self.min_volume_usdt:
                return None

            # RSI Check
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)
            alpha = 1.0 / self.rsi_period
            avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
            avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1])

            if current_rsi > self.momentum_rsi_cap:
                logger.debug(f"🚫 {symbol} RSI too high: {current_rsi:.1f} > {self.momentum_rsi_cap}")
                return None

            # Volume ratio
            avg_vol = volume.iloc[-20:].mean() if len(volume) >= 20 else volume.mean()
            volume_ratio = volume.iloc[-1] / avg_vol if avg_vol > 0 else 1
            if volume_ratio < self.momentum_volume_ratio:
                return None

            # Final decision
            if abs(price_change_pct) >= self.min_momentum_pct:
                direction = "BUY" if price_change_pct > 0 else "SELL"
                atr = df["atr"].iloc[-1] if "atr" in df.columns else current_price * 0.02

                stops = self.calculate_dynamic_stops(df, direction, current_price)

                logger.info(f"🚀 MOMENTUM: {symbol} {direction} ({price_change_pct:+.1f}%)")

                return {
                    "symbol": symbol,
                    "side": direction,
                    "price": float(current_price),
                    "stop_loss": stops["stop_loss"],
                    "take_profit": stops["take_profit"],
                    "strength": min(100, 70 + abs(price_change_pct)),
                    "reasons": [
                        f"Momentum {price_change_pct:+.1f}%",
                        f"Volume {volume_ratio:.1f}x",
                        f"RSI {current_rsi:.1f}"
                    ],
                    "rsi": current_rsi,
                    "volume_ratio": float(volume_ratio),
                    "is_momentum": True,
                    "signal_type": "momentum"
                }

            return None

        except Exception as e:
            logger.error(f"💥 Momentum detection failed for {symbol}: {e}")
            return None

    def scan_for_momentum(self, force_scan: bool = False) -> List[Dict]:
        """Scan momentum symbols for breakout opportunities."""
        if not self.momentum_enabled:
            return []

        signals = []
        now = time.time()

        for symbol in self.momentum_symbols:
            last_scan = self.last_momentum_scan.get(symbol, 0)
            if not force_scan and (now - last_scan) < (self.momentum_scan_interval * 60):
                continue

            try:
                kl = self.client.client.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=self.klines_interval,
                    limit=150,
                )
                if kl.get("retCode") != 0:
                    continue

                lst = kl.get("result", {}).get("list", [])
                if len(lst) < 80:
                    continue

                df = pd.DataFrame(lst, columns=["open_time", "open", "high", "low", "close", "volume", "turnover"])
                df[["open", "high", "low", "close", "volume", "turnover"]] = df[
                    ["open", "high", "low", "close", "volume", "turnover"]
                ].apply(pd.to_numeric)

                df = self.calculate_indicators(df)
                sig = self.detect_momentum(symbol, df)
                if sig:
                    signals.append(sig)

                self.last_momentum_scan[symbol] = now
                time.sleep(0.05)  # Rate limiting

            except Exception as e:
                logger.error(f"💥 Momentum scan error for {symbol}: {e}")

        if signals:
            logger.success(f"🎯 Found {len(signals)} momentum signals")
        return signals

    # =====================================================================
    # SIGNAL GENERATION & FILTERING
    # =====================================================================
    def generate_signals(self) -> List[Dict]:
        """Generate and rank all available signals."""
        signals: List[Dict] = []

        # Step 1: Momentum Signals
        if self.momentum_enabled:
            momentum_signals = self.scan_for_momentum()
            signals.extend(momentum_signals)

        # Step 2: Regular Strategy Signals (skip if momentum exists)
        momentum_symbols = {s["symbol"] for s in signals if s.get("is_momentum")}
        
        for symbol in self.symbols:
            if symbol in momentum_symbols:
                continue

            try:
                kl = self.client.client.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=self.klines_interval,
                    limit=self.klines_limit,
                )
                if kl.get("retCode") != 0:
                    logger.warning(f"⚠️ {symbol} kline fetch failed: {kl.get('retMsg', 'Unknown')}")
                    continue

                lst = kl.get("result", {}).get("list", [])
                if not lst:
                    continue

                df = pd.DataFrame(lst, columns=["open_time", "open", "high", "low", "close", "volume", "turnover"])
                df[["open", "high", "low", "close", "volume", "turnover"]] = df[
                    ["open", "high", "low", "close", "volume", "turnover"]
                ].apply(pd.to_numeric)

                result = self.analyze_symbol(symbol, df)
                if result:
                    signals.append(result)

            except Exception as e:
                logger.error(f"💥 Signal generation failed for {symbol}: {e}")

        # Step 3: Rank and Filter
        if signals:
            signals = self.filter_best_signals(signals)

        # Summary
        momentum_count = len([s for s in signals if s.get("is_momentum")])
        regular_count = len([s for s in signals if not s.get("is_momentum")])
        logger.info(f"📊 Total signals: {len(signals)} (Momentum: {momentum_count}, Regular: {regular_count})")

        return signals

    def filter_best_signals(self, signals: List[Dict]) -> List[Dict]:
        """Rank signals by strength and limit by MAX_TOTAL_POSITIONS."""
        if not signals:
            return []

        try:
            config = get_config_loader()
            max_positions = config.get("risk_management", "MAX_TOTAL_POSITIONS", 3, int)

            sorted_signals = sorted(
                signals,
                key=lambda x: x.get("strength", 0),
                reverse=True
            )

            selected = sorted_signals[:max_positions]

            if selected:
                top = selected[0]
                logger.info(f"🏆 Top Signal: {top['symbol']} {top['side']} (Strength: {top['strength']:.1f})")

            return selected

        except Exception as e:
            logger.error(f"❌ Signal filtering failed: {e}")
            return signals[:1] if signals else []

    # =====================================================================
    # RISK MANAGEMENT — ENHANCED
    # =====================================================================
    def calculate_dynamic_stops(self, df: pd.DataFrame, signal_type: str, entry: float) -> Dict[str, Any]:
        """
        🛡️ Professional ATR-Based Stop Loss & Take Profit
        Uses volatility-adaptive sizing with fallbacks and sanity checks.
        """
        try:
            last = df.iloc[-1]
            atr = float(last.get("atr", 0))

            # Fallback to 1% of price if ATR missing or invalid
            if atr <= 0 or pd.isna(atr):
                atr = entry * 0.01
                logger.debug(f"⚠️ ATR fallback used: {atr:.4f} (1% of entry)")

            # Get multipliers from config (with defaults)
            sl_mult = getattr(self, "sl_atr_multiplier", 2.0)
            tp_mult = getattr(self, "tp_atr_multiplier", 4.0)

            sl_dist = atr * sl_mult
            tp_dist = atr * tp_mult

            # Calculate stops based on direction
            if signal_type == "BUY":
                stop_loss = entry - sl_dist
                take_profit = entry + tp_dist
            else:  # SELL
                stop_loss = entry + sl_dist
                take_profit = entry - tp_dist

            # Risk-Reward Ratio
            risk_reward = tp_dist / sl_dist if sl_dist > 0 else 0

            logger.debug(
                f"🎯 Stops calculated → SL={stop_loss:.4f}, TP={take_profit:.4f}, "
                f"ATR={atr:.4f}, RR={risk_reward:.2f}"
            )

            return {
                "stop_loss": float(stop_loss),
                "take_profit": float(take_profit),
                "atr": float(atr),
                "risk_reward": float(risk_reward),
                "sl_distance": float(sl_dist),
                "tp_distance": float(tp_dist)
            }

        except Exception as e:
            logger.error(f"❌ Failed to calculate stops: {e}")
            # Return conservative fallback
            fallback_pct = 0.02  # 2%
            if signal_type == "BUY":
                return {
                    "stop_loss": entry * (1 - fallback_pct),
                    "take_profit": entry * (1 + fallback_pct * 2),
                    "atr": 0.0,
                    "risk_reward": 2.0,
                    "sl_distance": entry * fallback_pct,
                    "tp_distance": entry * fallback_pct * 2
                }
            else:
                return {
                    "stop_loss": entry * (1 + fallback_pct),
                    "take_profit": entry * (1 - fallback_pct * 2),
                    "atr": 0.0,
                    "risk_reward": 2.0,
                    "sl_distance": entry * fallback_pct,
                    "tp_distance": entry * fallback_pct * 2
                }

    # =====================================================================
    # MOMENTUM TRADING ENGINE
    # =====================================================================
    def detect_momentum(self, symbol: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        🚀 Detect High-Momentum Breakouts with Multi-Factor Confirmation
        Uses dynamic 24h lookback, volume surge, and RSI cap for quality control.
        """
        if not self.momentum_enabled or symbol not in self.momentum_symbols:
            return None

        if len(df) < max(50, self.candles_per_day // 2):  # Minimum data required
            logger.debug(f"📉 {symbol}: Insufficient data for momentum ({len(df)} candles)")
            return None

        try:
            close = df["close"].astype(float)
            volume = df["volume"].astype(float)

            # Dynamic 24h lookback
            lookback_candles = min(len(close) - 1, self.candles_per_day)
            if lookback_candles < 10:
                return None

            price_24h_ago = close.iloc[-lookback_candles]
            current_price = close.iloc[-1]
            price_change_pct = ((current_price - price_24h_ago) / price_24h_ago) * 100

            # Extreme move filter
            if abs(price_change_pct) > self.max_momentum_pct:
                logger.debug(f"🚫 {symbol}: Momentum too extreme ({price_change_pct:+.1f}%)")
                return None

            # Volume confirmation (24h turnover)
            turnover = df["turnover"].astype(float) if "turnover" in df.columns else None
            volume_24h_usdt = (
                turnover.iloc[-lookback_candles:].sum()
                if turnover is not None and len(turnover) >= lookback_candles
                else float(volume.iloc[-lookback_candles:].sum() * current_price)
            )

            if volume_24h_usdt < self.min_volume_usdt:
                logger.debug(f"🚫 {symbol}: Volume too low (${volume_24h_usdt:,.0f} < ${self.min_volume_usdt:,.0f})")
                return None

            # RSI Calculation (EMA-based)
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.ewm(alpha=1/self.rsi_period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/self.rsi_period, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi_series = 100 - (100 / (1 + rs))
            current_rsi = float(rsi_series.iloc[-1])

            # RSI Cap Filter
            if current_rsi > self.momentum_rsi_cap:
                logger.debug(f"🚫 {symbol}: RSI too high ({current_rsi:.1f} > {self.momentum_rsi_cap})")
                return None

            # Volume Ratio (vs 20-period MA)
            avg_vol = volume.iloc[-20:].mean() if len(volume) >= 20 else volume.mean()
            volume_ratio = float(volume.iloc[-1] / avg_vol) if avg_vol > 0 else 1.0

            if volume_ratio < self.momentum_volume_ratio:
                logger.debug(f"🚫 {symbol}: Volume ratio too low ({volume_ratio:.2f}x)")
                return None

            # Final Decision
            if abs(price_change_pct) >= self.min_momentum_pct:
                direction = "BUY" if price_change_pct > 0 else "SELL"

                # Use ATR for stops (fallback to 2% if unavailable)
                atr = df["atr"].iloc[-1] if "atr" in df.columns and not pd.isna(df["atr"].iloc[-1]) else current_price * 0.02
                stops = self.calculate_dynamic_stops(df, direction, current_price)

                strength_score = min(100.0, 70.0 + abs(price_change_pct) * 0.5)

                logger.info(
                    f"🚀 MOMENTUM SIGNAL: {symbol} {direction} "
                    f"({price_change_pct:+.1f}%) | RSI: {current_rsi:.1f} | Vol: {volume_ratio:.1f}x"
                )

                return {
                    "symbol": symbol,
                    "side": direction,
                    "price": float(current_price),
                    "stop_loss": stops["stop_loss"],
                    "take_profit": stops["take_profit"],
                    "strength": strength_score,
                    "reasons": [
                        f"Momentum: {price_change_pct:+.1f}%",
                        f"Volume: {volume_ratio:.1f}x above average",
                        f"RSI: {current_rsi:.1f}",
                        f"24h Volume: ${volume_24h_usdt:,.0f}"
                    ],
                    "rsi": current_rsi,
                    "volume_ratio": volume_ratio,
                    "volume_24h_usdt": volume_24h_usdt,
                    "price_change_pct": price_change_pct,
                    "is_momentum": True,
                    "signal_type": "momentum",
                    "atr_used": float(atr)
                }

            return None

        except Exception as e:
            logger.error(f"💥 Momentum detection failed for {symbol}: {e}")
            return None

    def scan_for_momentum(self, force_scan: bool = False) -> List[Dict[str, Any]]:
        """
        🔍 Scan all momentum symbols for breakout opportunities.
        Respects rate limits and scan intervals.
        """
        if not self.momentum_enabled:
            return []

        signals = []
        now = time.time()

        for symbol in self.momentum_symbols:
            # Respect scan interval
            last_scan = self.last_momentum_scan.get(symbol, 0)
            if not force_scan and (now - last_scan) < (self.momentum_scan_interval * 60):
                continue

            try:
                # Fetch klines
                kl = self.client.client.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=self.klines_interval,
                    limit=max(100, self.candles_per_day + 20),  # Ensure enough data
                )

                if kl.get("retCode") != 0:
                    logger.warning(f"⚠️ {symbol}: Kline fetch failed - {kl.get('retMsg', 'Unknown error')}")
                    continue

                lst = kl.get("result", {}).get("list", [])
                if len(lst) < 50:
                    logger.debug(f"⚠️ {symbol}: ({len(lst)} candles)")
                    continue

                # Create DataFrame
                df = pd.DataFrame(lst, columns=["open_time", "open", "high", "low", "close", "volume", "turnover"])
                df[["open", "high", "low", "close", "volume", "turnover"]] = df[
                    ["open", "high", "low", "close", "volume", "turnover"]
                ].apply(pd.to_numeric)

                # Calculate indicators
                df = self.calculate_indicators(df)
                if len(df) < 20:
                    continue

                # Detect momentum
                sig = self.detect_momentum(symbol, df)
                if sig:
                    signals.append(sig)
                    logger.success(f"✅ Momentum signal detected: {symbol} {sig['side']}")

                # Update scan timestamp
                self.last_momentum_scan[symbol] = now

                # Rate limiting
                time.sleep(0.05)

            except Exception as e:
                logger.error(f"💥 Error scanning {symbol} for momentum: {e}")
                continue

        if signals:
            logger.info(f"📊 Found {len(signals)} momentum signals")
        else:
            logger.debug("🔍 No momentum signals found in this scan")

        return signals

    # =====================================================================
    # CORE ANALYSIS ENGINE
    # =====================================================================
    def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        🧠 Comprehensive Symbol Analysis Engine
        Combines trend, momentum, volume, and breakout filters with institutional-grade scoring.
        NOW WITH ML INTEGRATION + EMERGENCY EXIT LOGIC!
        """
        try:
            if len(df) < self.min_candles_required:
                logger.debug(f"📉 {symbol}: Insufficient data ({len(df)} < {self.min_candles_required})")
                return None

            # Calculate indicators
            df = self.calculate_indicators(df)
            if len(df) < 10:
                return None

            last = df.iloc[-1]
            prev = df.iloc[-2]
            reasons = []
            trend = self.detect_trend(df)

            logger.debug(f"📈 {symbol} → Trend detected: {trend}")

            # Handle ranging markets
            if trend == "RANGING":
                if self.skip_ranging_markets and not self.enable_range_scalping:
                    logger.debug(
                        f"⏩ {symbol} ❌ Skipping RANGING market "
                        f"(SKIP_RANGING_MARKETS={self.skip_ranging_markets}, ENABLE_RANGE_SCALPING={self.enable_range_scalping})"
                    )
                    return None
                else:
                    logger.debug(f"🔄 {symbol}: Processing ranging market (range scalping enabled)")

                        # Add forced exit logic for open positions
            if hasattr(self, 'open_positions') and symbol in self.open_positions:
                position = self.open_positions[symbol]
                entry_price = float(position["entry_price"])

                # Use existing get_current_price method from BybitSession
                try:
                    if hasattr(self, 'client') and self.client and hasattr(self.client, 'get_current_price'):
                        current_price = self.client.get_current_price(symbol)
                        if current_price is None:
                            raise ValueError("get_current_price returned None")
                        logger.debug(f"✅ {symbol} Real-time price from exchange: {current_price}")
                    else:
                        current_price = float(last["close"])
                        logger.warning(f"⚠️ Using chart price for {symbol} — no live session or get_current_price unavailable")
                except Exception as e:
                    logger.error(f"❌ Failed to fetch real-time price for {symbol}: {e}")
                    current_price = float(last["close"])  # fallback

                # Calculate PnL percentage
                if position["side"] == "BUY":
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                else:  # SELL
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100

                logger.info(f"📊 {symbol} Position PnL: {pnl_pct:+.2f}% | Entry: {entry_price} | Current: {current_price}")

                # Check for emergency exit conditions
                should_exit = False
                exit_reasons = []

                # 1. RSI-based exit
                if position["side"] == "BUY" and last["rsi"] > self.rsi_overbought:
                    should_exit = True
                    exit_reasons.append(f"RSI overbought ({last['rsi']:.1f} > {self.rsi_overbought})")
                    logger.info(f"⚠️ FORCED EXIT: {symbol} BUY - RSI overbought ({last['rsi']:.1f})")

                elif position["side"] == "SELL" and last["rsi"] < self.rsi_oversold:
                    should_exit = True
                    exit_reasons.append(f"RSI oversold ({last['rsi']:.1f} < {self.rsi_oversold})")
                    logger.info(f"⚠️ FORCED EXIT: {symbol} SELL - RSI oversold ({last['rsi']:.1f})")

                # 2. Loss-based exit (if configured)
                try:
                    # Load from config if not already loaded
                    if not hasattr(self, 'max_loss_pct_per_trade'):
                        config_loader = get_config_loader()
                        self.max_loss_pct_per_trade = config_loader.getfloat("emergency_exit", "MAX_LOSS_PCT_PER_TRADE", 1.5)
                        logger.debug(f"🛡️ Loaded max_loss_pct_per_trade={self.max_loss_pct_per_trade} from INI")

                    max_loss_pct = self.max_loss_pct_per_trade
                    if pnl_pct < -max_loss_pct:
                        should_exit = True
                        exit_reasons.append(f"Loss exceeded limit ({pnl_pct:.2f}% < -{max_loss_pct}%)")
                        logger.warning(f"🚨 CLOSING {symbol} {position['side']} - Loss too high ({pnl_pct:.2f}%)")
                        
                except Exception as e:
                    logger.error(f"❌ Error loading loss exit threshold: {e}")
                    max_loss_pct = 1.5  # fallback
                    if pnl_pct < -max_loss_pct:
                        should_exit = True
                        exit_reasons.append(f"Loss exceeded limit ({pnl_pct:.2f}% < -{max_loss_pct}%)")
                        logger.warning(f"🚨 CLOSING {symbol} {position['side']} - Loss too high ({pnl_pct:.2f}%)")

                # 3. Time-based exit (if configured)
                max_age_hours = getattr(self, 'max_position_age_hours', 24)
                if hasattr(position, 'entry_time'):
                    from datetime import datetime, timezone
                    now = datetime.now(timezone.utc)
                    age_hours = (now - position["entry_time"]).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        should_exit = True
                        exit_reasons.append(f"Position too old ({age_hours:.1f}h > {max_age_hours}h)")
                        logger.info(f"⚠️ FORCED EXIT: {symbol} {position['side']} - Age exceeded ({age_hours:.1f}h)")

                if should_exit:
                    # Determine exit side (opposite of entry)
                    exit_side = "SELL" if position["side"] == "BUY" else "BUY"
                    
                    return {
                        "symbol": symbol,
                        "side": exit_side,
                        "price": current_price,
                        "stop_loss": 0.0,
                        "take_profit": 0.0,
                        "strength": 100.0,  # Force high priority for exits
                        "reasons": exit_reasons,
                        "is_momentum": False,
                        "signal_type": "emergency_exit",
                        "trend": trend,
                        "rsi": float(last["rsi"]),
                        "volume_ratio": float(last.get("volume_ratio", 1.0)),
                        "emergency_exit": True
                    }

            # BUY Conditions
            ema_cross_buy = prev["ema_fast"] < prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
            ema_weak_buy = (
                self.enable_weak_ema_entry and
                abs(last["ema_fast"] - last["ema_slow"]) / last["ema_slow"] <= self.weak_ema_tolerance
            )
            rsi_ok_buy = last["rsi"] < self.rsi_overbought
            trend_ok_buy = last["ema_fast"] > last["ema_long"]

            # Hybrid Mode Override for Ranging Markets
            if trend == "RANGING" and self.strategy_style == "HYBRID":
                trend_ok_buy = True
                logger.debug(f"🔄 {symbol}: HYBRID mode override - trend_ok_buy set to True for ranging market")

            if self.require_ema_cross_for_entry:
                buy_condition = ema_cross_buy and rsi_ok_buy and trend_ok_buy
            else:
                buy_condition = (ema_cross_buy or ema_weak_buy) and rsi_ok_buy and trend_ok_buy

            # SELL Conditions
            ema_cross_sell = prev["ema_fast"] > prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]
            ema_weak_sell = (
                self.enable_weak_ema_entry and
                abs(last["ema_fast"] - last["ema_slow"]) / last["ema_slow"] <= self.weak_ema_tolerance
            )
            rsi_ok_sell = last["rsi"] > self.rsi_oversold
            trend_ok_sell = last["ema_fast"] < last["ema_long"]

            # Hybrid Mode Override for Ranging Markets
            if trend == "RANGING" and self.strategy_style == "HYBRID":
                trend_ok_sell = True
                logger.debug(f"🔄 {symbol}: HYBRID mode override - trend_ok_sell set to True for ranging market")

            if self.require_ema_cross_for_entry:
                sell_condition = ema_cross_sell and rsi_ok_sell and trend_ok_sell
            else:
                sell_condition = (ema_cross_sell or ema_weak_sell) and rsi_ok_sell and trend_ok_sell

            # Log detailed BUY/SELL checks
            logger.debug(
                f"{symbol} BUY CHECKS → "
                f"ema_cross={ema_cross_buy}, ema_weak={ema_weak_buy}, "
                f"rsi_ok={rsi_ok_buy}, trend_ok={trend_ok_buy}"
            )
            logger.debug(
                f"{symbol} SELL CHECKS → "
                f"ema_cross={ema_cross_sell}, ema_weak={ema_weak_sell}, "
                f"rsi_ok={rsi_ok_sell}, trend_ok={trend_ok_sell}"
            )

            # Breakout Filter
            breakout_buy = breakout_sell = True
            if self.enable_breakout_filter and len(df) > self.breakout_lookback_bars + 2:
                lookback_start = max(0, len(df) - self.breakout_lookback_bars - 2)
                lookback_end = len(df) - 2
                lookback_df = df.iloc[lookback_start:lookback_end]
                
                if len(lookback_df) > 0:
                    recent_high = lookback_df["high"].max()
                    recent_low = lookback_df["low"].min()
                    breakout_pct = self.breakout_min_pct / 100.0

                    breakout_buy = last["close"] >= recent_high * (1 + breakout_pct)
                    breakout_sell = last["close"] <= recent_low * (1 - breakout_pct)

                    if breakout_buy:
                        reasons.append(f"Broke out above resistance (+{self.breakout_min_pct}%)")
                        logger.debug(f"🚀 {symbol}: Breakout BUY condition met (price ≥ high + {self.breakout_min_pct}%)")
                    if breakout_sell:
                        reasons.append(f"Broke down below support (-{self.breakout_min_pct}%)")
                        logger.debug(f"🚀 {symbol}: Breakout SELL condition met (price ≤ low - {self.breakout_min_pct}%)")

            # Apply Strategy Style
            if self.strategy_style == "BREAKOUT":
                final_buy = buy_condition and breakout_buy
                final_sell = sell_condition and breakout_sell
                logger.debug(f"🎯 {symbol}: BREAKOUT mode - requiring both EMA/RSI and breakout conditions")
            elif self.strategy_style == "HYBRID":
                final_buy = buy_condition  # Breakout is optional confidence booster
                final_sell = sell_condition
                logger.debug(f"🎯 {symbol}: HYBRID mode - EMA/RSI signals primary, breakout adds confidence")
            else:  # "TREND" or other
                final_buy = buy_condition
                final_sell = sell_condition
                logger.debug(f"🎯 {symbol}: TREND mode - using EMA/RSI signals without breakout requirement")

            # Global Signal Filters
            if final_buy and not self.allow_buy_signals:
                logger.debug(f"🚫 {symbol} ❌ BUY blocked by config (ALLOW_BUY_SIGNALS={self.allow_buy_signals})")
                final_buy = False

            if final_sell and not self.allow_sell_signals:
                logger.debug(f"🚫 {symbol} ❌ SELL blocked by config (ALLOW_SELL_SIGNALS={self.allow_sell_signals})")
                final_sell = False

            if not (final_buy or final_sell):
                logger.debug(f"❌ {symbol} ❌ No signal (buy={final_buy}, sell={final_sell})")
                return None

            # Determine signal type
            signal_type = "BUY" if final_buy else "SELL"
            reasons.append(f"Direction: {signal_type}")

            # Calculate Technical Signal Strength
            tech_score = self.calculate_signal_strength(df, signal_type, symbol)
            min_required = self._effective_min_strength(signal_type)

            logger.debug(
                f"📊 {symbol} Technical Score → "
                f"raw={tech_score:.1f}, required={min_required:.1f}, "
                f"base={self.min_signal_strength}, side_strictness={getattr(self, 'buy_strictness' if signal_type == 'BUY' else 'sell_strictness', 0.0)}"
            )

            final_score = tech_score
            ml_used = False

            # Check if ML should be used
            if hasattr(self, 'ml_manager') and self.ml_manager and getattr(self.ml_manager, 'enabled', False):
                try:
                    # Prepare features for ML prediction
                    ml_features = {
                        "price": float(last["close"]),
                        "rsi": float(last["rsi"]),
                        "volume": float(last["volume"]),
                        "volatility": float(last["atr"]) if "atr" in last else 0.02,
                        "trend_strength": tech_score,
                        "ema_ratio": float(last["ema_fast"] / last["ema_slow"]) if last["ema_slow"] != 0 else 1.0,
                        "direction": signal_type,
                        "symbol": symbol,
                        "volume_ratio": float(last["volume_ratio"]) if "volume_ratio" in last else 1.0,
                        "ema_fast": float(last["ema_fast"]),
                        "ema_slow": float(last["ema_slow"]),
                        "ema_long": float(last["ema_long"])
                    }
                    
                    # Get ML prediction
                    ml_result = self.ml_manager.predict(symbol, ml_features)
                    
                    if ml_result and ml_result.get('success'):
                        confidence = ml_result.get('confidence', 0)
                        if confidence >= self.confidence_threshold:  # From INI via config_loader
                            ml_prediction_score = ml_result.get('prediction_score', tech_score)
                            
                            # Blend scores according to ML_BLEND_WEIGHT from INI
                            blended_score = tech_score * (1 - self.ml_blend_weight) + ml_prediction_score * self.ml_blend_weight
                            
                            logger.info(
                                f"🤖 {symbol} | ML ADJUSTED SCORE: {blended_score:.1f} "
                                f"(Tech: {tech_score:.1f}, ML: {ml_prediction_score:.1f}, "
                                f"Conf: {confidence:.2f}, Weight: {self.ml_blend_weight})"
                            )
                            
                            final_score = blended_score
                            ml_used = True
                            reasons.append(f"ML adjusted score (confidence: {confidence:.2f})")
                        else:
                            logger.debug(f"📉 {symbol} | ML skipped - low confidence: {confidence:.2f}")
                    else:
                        logger.debug(f"⚠️ {symbol} | ML prediction failed or returned no result")
                        
                except Exception as e:
                    logger.error(f"❌ Error during ML prediction for {symbol}: {e}")

            # Use final_score for decision making
            if final_score < min_required:
                logger.debug(
                    f"❌ {symbol} ❌ Signal too weak: strength={final_score:.1f}, "
                    f"required={min_required:.1f} (base={self.min_signal_strength}, mode={self.strictness_mode})"
                )
                return None

            # Volume Filter
            if self.enable_volume_filter and last["volume_ratio"] < self.min_volume_ratio:
                logger.debug(
                    f"❌ {symbol} ❌ Volume too low: ratio={last['volume_ratio']:.2f}, "
                    f"min={self.min_volume_ratio:.2f}"
                )
                return None

            if last["volume_ratio"] >= self.min_volume_ratio:
                reasons.append(f"Volume ratio high: {last['volume_ratio']:.2f}x above MA")
            
            entry_price = float(last["close"])  # fallback
            if hasattr(self, 'client') and self.client and hasattr(self.client, 'get_current_price'):
                try:
                    live_price = self.client.get_current_price(symbol)
                    if live_price is not None and live_price > 0:
                        entry_price = float(live_price)
                        logger.debug(f"📡 {symbol}: Using LIVE price for signal: {entry_price:.6f} (vs candle close {last['close']:.6f})")
                    else:
                        logger.warning(f"⚠️ {symbol}: get_current_price returned invalid value ({live_price}), using candle close")
                except Exception as e:
                    logger.warning(f"⚠️ {symbol}: Failed to fetch live price for signal: {e}, using candle close")

            # Calculate Stops
            stops = self.calculate_dynamic_stops(df, signal_type, entry_price)

            logger.success(
                f"✅ SIGNAL → {symbol} {signal_type} @ {entry_price:.4f} " 
                f"| SL={stops['stop_loss']:.4f}, TP={stops['take_profit']:.4f} "
                f"| Strength={final_score:.1f}/{min_required:.1f}, RR={stops['risk_reward']:.2f}"
                f"{' [ML]' if ml_used else ''}"
            )

            return {
                "symbol": symbol,
                "side": signal_type,
                "price": entry_price,
                "stop_loss": stops["stop_loss"],
                "take_profit": stops["take_profit"],
                "strength": final_score,
                "trend": trend,
                "rsi": float(last["rsi"]),
                "atr": stops["atr"],
                "risk_reward": stops["risk_reward"],
                "volume_ratio": float(last["volume_ratio"]),
                "reasons": reasons,
                "is_momentum": False,
                "signal_type": "regular",
                "ema_fast": float(last["ema_fast"]),
                "ema_slow": float(last["ema_slow"]),
                "ema_long": float(last["ema_long"]),
                "ml_used": ml_used,
                "ml_confidence": ml_result.get('confidence', 0) if 'ml_result' in locals() and ml_result else 0
            }

        except Exception as e:
            logger.error(f"💥 Analysis failed for {symbol}: {e}")
            return None

    # =====================================================================
    # MAIN SIGNAL GENERATION PIPELINE
    # =====================================================================
    def generate_signals(self) -> List[Dict[str, Any]]:
        """
        🏭 Unified Signal Generation Pipeline
        Combines momentum and regular signals with intelligent deduplication and ranking.
        """
        signals: List[Dict[str, Any]] = []

        # Phase 1: Generate Momentum Signals
        if self.momentum_enabled:
            logger.info("🔍 Scanning for momentum signals...")
            momentum_signals = self.scan_for_momentum()
            signals.extend(momentum_signals)
            if momentum_signals:
                logger.success(f"🚀 Found {len(momentum_signals)} momentum signals")

        # Track symbols with momentum signals to avoid duplication
        momentum_symbols = {s["symbol"] for s in signals if s.get("is_momentum", False)}

        # Phase 2: Generate Regular Strategy Signals
        logger.info("📊 Analyzing regular strategy signals...")
        for symbol in self.symbols:
            if symbol in momentum_symbols:
                logger.debug(f"⏩ Skipping {symbol} - already has momentum signal")
                continue

            try:
                # Fetch klines
                kl = self.client.client.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=self.klines_interval,
                    limit=self.klines_limit,
                )

                if kl.get("retCode") != 0:
                    logger.warning(f"⚠️ {symbol}: Kline fetch failed - {kl.get('retMsg', 'Unknown error')}")
                    continue

                lst = kl.get("result", {}).get("list", [])
                if not lst:
                    logger.debug(f"⚠️ {symbol}: Empty kline response")
                    continue

                # Create and process DataFrame
                # Create and process DataFrame
                df = pd.DataFrame(lst, columns=["open_time", "open", "high", "low", "close", "volume", "turnover"])
                df[["open", "high", "low", "close", "volume", "turnover"]] = df[
                    ["open", "high", "low", "close", "volume", "turnover"]
                ].apply(pd.to_numeric)

                # Analyze symbol
                result = self.analyze_symbol(symbol, df)
                if result:
                    signals.append(result)
                    
                    risk_reward = result.get('risk_reward', 2.0)
                    
                    logger.success(
                        f"📈 Strategy signal: {symbol} {result['side']} "
                        f"(Strength: {result['strength']:.1f}, RR: {risk_reward:.2f})"
                    )

            except Exception as e:
                logger.error(f"💥 Error generating signal for {symbol}: {e}")
                continue

        # Phase 3: Filter and Rank Signals
        if signals:
            signals = self.filter_best_signals(signals)

        # Summary
        momentum_count = len([s for s in signals if s.get("is_momentum", False)])
        regular_count = len([s for s in signals if not s.get("is_momentum", False)])
        
        logger.info(
            f"🎯 SIGNAL SUMMARY: Total={len(signals)} "
            f"(Momentum: {momentum_count}, Regular: {regular_count})"
        )

        if signals:
            top_signal = signals[0]
            logger.info(
                f"🏆 TOP SIGNAL: {top_signal['symbol']} {top_signal['side']} "
                f"at {top_signal['price']:.4f} (Strength: {top_signal['strength']:.1f})"
            )

        return signals