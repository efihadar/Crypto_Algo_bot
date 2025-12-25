# bot/smart_safety.py
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from loguru import logger
import pandas as pd

from config_loader import get_config_loader

@dataclass
class SafetyAction:
    action: str          # "CLOSE_ALL" / "CLOSE_SYMBOL"
    symbol: Optional[str]
    reason: str
    priority: str = "HIGH"  # HIGH / MEDIUM / LOW

class SmartSafetyManager:
    """
    מנגנון בטיחות חכם:
    A) יציאה לפי שינוי טרנד / אות הפוך חזק מהאסטרטגיה
    B) יציאה "חכמה" לפי ATR (הפסד גדול מהתנודתיות)
    C) פאניק־אקסיט גלובלי לפי Drawdown / קריסת BTC
    """
    def __init__(self, client, strategy):
        """
        client – אותו client שאתה כבר משתמש בו (BybitClientWrapper)
        strategy – האובייקט של EnhancedBybitStrategy
        """
        self.client = client
        self.strategy = strategy

        cfg = get_config_loader()

        self.enabled = cfg.get("smart_safety", "ENABLE_SMART_SAFETY", True, bool)

        # A: trend exit
        self.trend_enabled = cfg.get("smart_safety", "TREND_EXIT_ENABLED", True, bool)
        self.trend_min_strength = cfg.get(
            "smart_safety", "TREND_EXIT_MIN_STRENGTH", 60, int
        )

        # B: ATR-based extra loss limit
        self.trailing_enabled = cfg.get(
            "smart_safety", "TRAILING_ATR_ENABLED", True, bool
        )
        self.trailing_max_loss_atr = cfg.get(
            "smart_safety", "TRAILING_MAX_LOSS_ATR", 1.0, float
        )

        # C: global safety
        self.global_enabled = cfg.get(
            "smart_safety", "GLOBAL_SAFETY_ENABLED", True, bool
        )
        self.global_max_dd_pct = cfg.get(
            "smart_safety", "GLOBAL_MAX_DRAWDOWN_PCT", 35.0, float
        )
        self.global_btc_drop_pct = cfg.get(
            "smart_safety", "GLOBAL_BTC_DROP_PCT", 3.0, float
        )
        self.global_lookback_min = cfg.get(
            "smart_safety", "GLOBAL_LOOKBACK_MIN", 5, int
        )

        logger.success(
            f"🛡️ SmartSafetyManager init → enabled={self.enabled}, "
            f"trend={self.trend_enabled}, trailing={self.trailing_enabled}, global={self.global_enabled}"
        )

    def evaluate(self, open_positions: List[Dict[str, Any]], account_status: Dict[str, Any]) -> List[SafetyAction]:
        """
        Main entry point: Evaluates portfolio and individual positions for safety triggers.
        Args:
            open_positions: List of active position dictionaries.
            account_status: Dictionary containing 'balance' and 'equity'.
        Returns: List of SafetyAction objects to be executed by the OrderManager.
        """
        actions: List[SafetyAction] = []
        
        # 1. Kill-switch check
        if not self.enabled:
            return actions

        # 2. Global Safety Check (Panic Exit / BTC Crash)
        if self.global_enabled:
            global_action = self._check_global_safety(account_status)
            if global_action:
                actions.append(global_action)

                return actions

        # 3. Individual Symbol Safety (Trend Exit / ATR Trailing)
        if not open_positions:
            return actions

        for pos in open_positions:
            try:
                sym_actions = self._check_symbol_safety(pos)
                if sym_actions:
                    actions.extend(sym_actions)
            except Exception as e:

                symbol = pos.get("symbol", "Unknown")
                logger.error(f"❌ Safety evaluation failed for {symbol}: {str(e)}")

        return actions

    # ------------------------------------------------------------------
    #  Global safety – Drawdown / BTC crash
    # ------------------------------------------------------------------
    def _check_global_safety(self, account_status: Dict[str, Any]) -> Optional[SafetyAction]:
        """
        Checks for account-wide safety triggers: Global Drawdown and BTC Market Crash.
        """
        balance = float(account_status.get("balance") or 0.0)
        equity = float(account_status.get("equity") or 0.0)
        
        if balance <= 0:
            return None

        # 1. Global Portfolio Drawdown Check
        dd_pct = max(0.0, (balance - equity) / balance * 100.0)
        if dd_pct >= self.global_max_dd_pct:
            reason = (
                f"Global drawdown {dd_pct:.1f}% >= max {self.global_max_dd_pct:.1f}% - Emergency Liquidation"
            )
            logger.warning(f"🛑 {reason}")
            return SafetyAction(
                action="CLOSE_ALL",
                symbol=None,
                reason=reason,
                priority="CRITICAL",
            )

        # 2. BTC Crash Check (Market Sentiment Proxy)
        try:
            kl = self.client.client.get_kline(
                category="linear",
                symbol="BTCUSDT",
                interval="1",
                limit=max(self.global_lookback_min + 1, 10),
            )
            
            if kl.get("retCode") != 0:
                return None
                
            lst = kl.get("result", {}).get("list", [])
            if len(lst) < self.global_lookback_min:
                return None

            # Process BTC data
            df = pd.DataFrame(
                lst,
                columns=["open_time", "open", "high", "low", "close", "volume", "turnover"]
            )
            df["close"] = pd.to_numeric(df["close"])
            
            # Calculate drop from the peak within the lookback window
            current_price = float(df["close"].iloc[0]) # Bybit returns newest first
            recent_prices = df["close"].iloc[0 : self.global_lookback_min]
            peak_price = recent_prices.max()
            
            drop_from_peak = ((peak_price - current_price) / peak_price) * 100.0

            if drop_from_peak >= self.global_btc_drop_pct:
                reason = (
                    f"BTC Crash: -{drop_from_peak:.2f}% from peak in last "
                    f"{self.global_lookback_min}m (Threshold: {self.global_btc_drop_pct}%)"
                )
                logger.warning(f"🛑 {reason}")
                return SafetyAction(
                    action="CLOSE_ALL",
                    symbol=None,
                    reason=reason,
                    priority="CRITICAL",
                )
                
        except Exception as e:
            logger.error(f"❌ Global BTC safety check failed: {str(e)}")

        return None

    # ------------------------------------------------------------------
    #  A + B 
    # ------------------------------------------------------------------
    def _check_symbol_safety(self, pos: Dict[str, Any]) -> List[SafetyAction]:
        """
        Evaluates specific position safety: Trend Reversal and ATR-based Loss Limits.
        """
        actions: List[SafetyAction] = []

        symbol = pos.get("symbol")
        # Supports both 'side' (Bybit standard) and 'direction' formats
        side = pos.get("side") or pos.get("direction")
        entry_price = float(pos.get("entry_price") or 0.0)
        
        if not symbol or not side or entry_price <= 0:
            return actions

        try:
            # 1. Fetch market data for the specific symbol
            kl = self.client.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=self.strategy.klines_interval,
                limit=self.strategy.klines_limit,
            )
            
            if kl.get("retCode") != 0:
                logger.error(f"{symbol} ❌ Kline fetch failed: {kl.get('retMsg')}")
                return actions

            lst = kl.get("result", {}).get("list", [])
            if not lst:
                return actions

            # 2. Prepare DataFrame
            df = pd.DataFrame(
                lst,
                columns=["open_time", "open", "high", "low", "close", "volume", "turnover"]
            )
            # Convert to numeric and ensure correct chronological order (oldest to newest) for indicators
            df[["open", "high", "low", "close", "volume"]] = df[
                ["open", "high", "low", "close", "volume"]
            ].apply(pd.to_numeric)
            
            df = df.iloc[::-1].reset_index(drop=True) # Flip to ASCENDING order

            # 3. Calculate Strategy Indicators
            df = self.strategy.calculate_indicators(df)
            if df.empty or len(df) < 10:
                return actions

            last_row = df.iloc[-1]
            current_price = float(last_row["close"])
            atr = float(last_row.get("atr") or 0.0)

            # 4. A: Trend Reversal Check (Opposite Signal Strength)
            if self.trend_enabled:
                sig = self.strategy.analyze_symbol(symbol, df)
                if sig and sig.get("side"):
                    sig_side = sig["side"].upper()
                    strength = float(sig.get("strength") or 0.0)

                    if self._is_opposite(side, sig_side) and strength >= self.trend_min_strength:
                        reason = (
                            f"{symbol}: Trend Reversal - Side: {side}, "
                            f"New Signal: {sig_side} (Strength: {strength:.1f} >= {self.trend_min_strength})"
                        )
                        logger.warning(f"🛑 {reason}")
                        actions.append(
                            SafetyAction(
                                action="CLOSE_SYMBOL",
                                symbol=symbol,
                                reason=reason,
                                priority="HIGH",
                            )
                        )

            # 5. B: ATR-Based Trailing/Max Loss Check
            if self.trailing_enabled and atr > 0:
                # Calculate absolute loss in price units
                is_buy = side.lower() in ["buy", "long"]
                loss = (entry_price - current_price) if is_buy else (current_price - entry_price)

                if loss > 0:  # Only evaluate if currently in drawdown
                    max_allowed_loss = self.trailing_max_loss_atr * atr
                    if loss >= max_allowed_loss:
                        reason = (
                            f"{symbol}: ATR Limit Exceeded - Loss: {loss:.4f} >= "
                            f"{self.trailing_max_loss_atr} * ATR ({atr:.4f})"
                        )
                        logger.warning(f"🛑 {reason}")
                        actions.append(
                            SafetyAction(
                                action="CLOSE_SYMBOL",
                                symbol=symbol,
                                reason=reason,
                                priority="MEDIUM",
                            )
                        )

        except Exception as e:
            logger.error(f"{symbol} ❌ Individual safety check failed: {str(e)}")

        return actions

    @staticmethod
    def _is_opposite(position_side: str, signal_side: str) -> bool:
        """
        position_side "Buy"/"Sell"
        signal_side "BUY"/"SELL"
        """
        ps = position_side.strip().upper()
        ss = signal_side.strip().upper()
        return (ps == "BUY" and ss == "SELL") or (ps == "SELL" and ss == "BUY")
