# regime_detector.py
"""
Market Regime Detection
Detects current market conditions and adapts strategy accordingly
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger
from enum import Enum

class MarketRegime(Enum):
    """Market regime types."""
    TRENDING_BULL = "TRENDING_BULL"      # Strong uptrend
    TRENDING_BEAR = "TRENDING_BEAR"      # Strong downtrend
    RANGING = "RANGING"                  # Sideways/choppy
    HIGH_VOLATILITY = "HIGH_VOLATILITY"  # Volatile/uncertain
    LOW_VOLATILITY = "LOW_VOLATILITY"    # Quiet/compressing

class RegimeDetector:
    """Detects and tracks market regime."""
    
    def __init__(self, session, config_loader):
        """
        Args:
            session: BybitSession instance
            config_loader: ConfigLoader instance
        """
        self.session = session
        self.cfg = config_loader
        
        # Load settings
        section = "regime_detection"
        self.enabled = self.cfg.get(section, "ENABLED", True, bool)
        self.lookback_candles = self.cfg.get(section, "LOOKBACK_CANDLES", 100, int)
        self.regime_interval = self.cfg.get(section, "REGIME_INTERVAL", "1h", str)
        
        # Thresholds
        self.trend_threshold = self.cfg.get(section, "TREND_THRESHOLD", 0.25, float)
        self.high_vol_threshold = self.cfg.get(section, "HIGH_VOL_THRESHOLD", 2.0, float)
        self.low_vol_threshold = self.cfg.get(section, "LOW_VOL_THRESHOLD", 0.5, float)
        
        # Cache
        self._regime_cache: Dict[str, Tuple[MarketRegime, float]] = {}
        
        logger.info(f"🌊 RegimeDetector initialized (enabled={self.enabled})")
    
    def detect_regime(self, symbol: str, df: Optional[pd.DataFrame] = None) -> MarketRegime:
        """
        Detect current market regime for a symbol.
        Args:
            symbol: Trading symbol
            df: Optional pre-fetched DataFrame 
        Returns:
            MarketRegime enum
        """
        if not self.enabled:
            return MarketRegime.RANGING
        
        try:
            # Check cache
            if symbol in self._regime_cache:
                regime, timestamp = self._regime_cache[symbol]
                if (pd.Timestamp.now().timestamp() - timestamp) < 300:  # 5min cache
                    return regime
            
            # Fetch data if not provided
            if df is None or len(df) < 50:
                df = self._fetch_regime_data(symbol)
            
            if df is None or len(df) < 50:
                logger.warning(f"⚠️  Insufficient data for {symbol} regime detection")
                return MarketRegime.RANGING
            
            # Calculate regime indicators
            regime = self._calculate_regime(df)
            
            # Cache result
            self._regime_cache[symbol] = (regime, pd.Timestamp.now().timestamp())
            
            logger.debug(f"🌊 {symbol} regime: {regime.value}")
            
            return regime
            
        except Exception as e:
            logger.error(f"❌ Regime detection failed for {symbol}: {e}")
            return MarketRegime.RANGING
    
    def _fetch_regime_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch kline data for regime detection."""
        try:
            klines = self.session.get_klines(
                symbol=symbol,
                interval=self.regime_interval,
                limit=self.lookback_candles
            )
            
            if not klines:
                return None
            
            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume", "turnover"
            ])
            
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch regime data for {symbol}: {e}")
            return None
    
    def _calculate_regime(self, df: pd.DataFrame) -> MarketRegime:
        """
        Calculate market regime based on technical indicators.
        Args:
            df: DataFrame with OHLCV data
        Returns:
            MarketRegime enum
        """
        try:
            # 1. Calculate EMAs
            df["ema_fast"] = df["close"].ewm(span=20).mean()
            df["ema_slow"] = df["close"].ewm(span=50).mean()
            
            # 2. Calculate ATR (volatility)
            df["high_low"] = df["high"] - df["low"]
            df["high_close"] = abs(df["high"] - df["close"].shift())
            df["low_close"] = abs(df["low"] - df["close"].shift())
            df["true_range"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
            df["atr"] = df["true_range"].rolling(14).mean()
            
            # 3. Calculate trend strength
            df["trend_strength"] = abs(df["ema_fast"] - df["ema_slow"]) / df["close"] * 100
            
            # 4. Calculate returns
            df["returns"] = df["close"].pct_change()
            
            # Get recent values
            last = df.iloc[-1]
            recent = df.iloc[-20:]  # Last 20 periods
            
            # Calculate metrics
            avg_trend_strength = recent["trend_strength"].mean()
            current_atr_pct = (last["atr"] / last["close"]) * 100
            recent_atr_pct = (recent["atr"] / recent["close"] * 100).mean()
            
            # Volatility ratio (current vs historical)
            vol_ratio = current_atr_pct / recent_atr_pct if recent_atr_pct > 0 else 1.0
            
            # Trend direction
            ema_slope = (df["ema_fast"].iloc[-1] - df["ema_fast"].iloc[-20]) / df["ema_fast"].iloc[-20]
            
            # --- Regime Decision Logic ---
            
            # High Volatility (overrides other regimes)
            if vol_ratio >= self.high_vol_threshold:
                return MarketRegime.HIGH_VOLATILITY
            
            # Low Volatility
            if vol_ratio <= self.low_vol_threshold:
                return MarketRegime.LOW_VOLATILITY
            
            # Trending
            if avg_trend_strength >= self.trend_threshold:
                if ema_slope > 0.02:  # 2% positive slope
                    return MarketRegime.TRENDING_BULL
                elif ema_slope < -0.02:  # 2% negative slope
                    return MarketRegime.TRENDING_BEAR
            
            # Default: Ranging
            return MarketRegime.RANGING
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate regime: {e}")
            return MarketRegime.RANGING
    
    def get_regime_adjustments(self, regime: MarketRegime) -> Dict:
        """
        Get strategy adjustments for current regime.
        Args:
            regime: MarketRegime enum
        Returns:
            Dictionary with adjustment parameters
        """
        adjustments = {
            MarketRegime.TRENDING_BULL: {
                "position_size_mult": 1.2,
                "sl_atr_mult": 2.0,
                "tp_atr_mult": 4.0,
                "trailing_activation": 1.0,
                "allow_buy": True,
                "allow_sell": False,
                "strategy": "trend_following",
            },
            MarketRegime.TRENDING_BEAR: {
                "position_size_mult": 1.0,
                "sl_atr_mult": 2.0,
                "tp_atr_mult": 3.5,
                "trailing_activation": 0.8,
                "allow_buy": False,
                "allow_sell": True,
                "strategy": "trend_following",
            },
            MarketRegime.RANGING: {
                "position_size_mult": 0.7,
                "sl_atr_mult": 1.5,
                "tp_atr_mult": 2.5,
                "trailing_activation": 0.5,
                "allow_buy": True,
                "allow_sell": True,
                "strategy": "mean_reversion",
            },
            MarketRegime.HIGH_VOLATILITY: {
                "position_size_mult": 0.5,
                "sl_atr_mult": 3.0,
                "tp_atr_mult": 5.0,
                "trailing_activation": 1.5,
                "allow_buy": True,
                "allow_sell": True,
                "strategy": "breakout",
            },
            MarketRegime.LOW_VOLATILITY: {
                "position_size_mult": 1.3,
                "sl_atr_mult": 1.0,
                "tp_atr_mult": 2.0,
                "trailing_activation": 0.3,
                "allow_buy": True,
                "allow_sell": True,
                "strategy": "breakout_anticipation",
            },
        }
        
        return adjustments.get(regime, adjustments[MarketRegime.RANGING])
    
    def get_regime_report(self, symbols: list) -> Dict:
        """
        Generate regime report for multiple symbols.
        Args:
            symbols: List of symbols to analyze
        Returns:
            Report dictionary
        """
        if not self.enabled:
            return {"enabled": False}
        
        try:
            regimes = {}
            regime_counts = {r: 0 for r in MarketRegime}
            
            for symbol in symbols:
                try:
                    regime = self.detect_regime(symbol)
                    regimes[symbol] = regime.value
                    regime_counts[regime] += 1
                except Exception as e:
                    logger.warning(f"⚠️ Failed to detect regime for {symbol}: {e}")
                    continue
            
            # Determine dominant regime
            dominant = max(regime_counts.items(), key=lambda x: x[1])
            
            return {
                "enabled": True,
                "symbols_analyzed": len(regimes),
                "regimes": regimes,
                "regime_distribution": {r.value: c for r, c in regime_counts.items()},
                "dominant_regime": dominant[0].value,
                "dominant_count": dominant[1],
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate regime report: {e}")
            return {"enabled": True, "error": str(e)}
