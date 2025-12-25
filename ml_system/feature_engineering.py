# ml_system/feature_engineering.py
"""
Professional Feature Engineering for Trading ML Systems
Extracts 90+ features across 9 categories: Price, Technical, Volume, Signal, Context,
Multi-Timeframe, Bollinger Bands, Support/Resistance, Candlestick Patterns.
Elite Version 1.1: Optimized for stability, removed RuntimeWarnings, improved Math.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import hashlib
from .config import ml_config

class FeatureEngineer:
    """
    Handles feature extraction and engineering for ML models.
    Optimized for high-frequency calculation and statistical stability.
    """
    
    # Constants for optimization
    MAX_LOOKBACK = 500  # Max rows needed for indicator stability
    EPSILON = 1e-10     # Small epsilon to prevent division by zero
    
    def __init__(self):
        self.feature_window = ml_config.feature_window
        self.use_technical = ml_config.use_technical_indicators
        self.use_volume = ml_config.use_volume_features
        self.use_price = ml_config.use_price_features
        
        # Enhanced caching strategy
        self._cache = {} 
        self._last_hash = None
        self._last_features = None

    def _generate_data_hash(self, df: pd.DataFrame) -> str:
        """Generates a quick hash of the latest data point to validate cache."""
        if df.empty:
            return ""
        # Hash based on the last timestamp and last close price
        last_idx = str(df.index[-1])
        last_close = str(df['close'].iloc[-1])
        return hashlib.md5(f"{last_idx}_{last_close}".encode()).hexdigest()

    def extract_features(self, df: pd.DataFrame, signal: Dict) -> Optional[Dict]:
        """
        Extract comprehensive features from price data and signal.
        """
        try:
            # 1. Validation and Optimization
            if not isinstance(df, pd.DataFrame) or len(df) < 30:
                logger.debug(f"⚠️ Insufficient data: {len(df) if isinstance(df, pd.DataFrame) else 'Invalid'}")
                return None
            
            # Check cache
            current_hash = self._generate_data_hash(df)
            if self._last_hash == current_hash and self._last_features is not None:
                signal_features = self._extract_signal_features(signal, df)
                features = self._last_features.copy()
                features.update(signal_features)
                return features

            # Optimization: Slice DataFrame
            if len(df) > self.MAX_LOOKBACK:
                df_calc = df.iloc[-self.MAX_LOOKBACK:].copy()
            else:
                df_calc = df.copy()

            # Ensure numeric types and handle missing columns safely
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df_calc.columns for col in required_cols):
                logger.error(f"❌ Missing required columns in dataframe")
                return None
            
            df_numeric = df_calc[required_cols].astype(np.float64)
            
            # Replace 0 volume with epsilon to avoid division errors later
            df_numeric['volume'] = df_numeric['volume'].replace(0, self.EPSILON)
            
            features = {}
            
            # 2. Feature Extraction Pipelines
            if self.use_price:
                features.update(self._extract_price_features(df_numeric, signal))
            
            if self.use_technical:
                features.update(self._extract_technical_features(df_numeric))
            
            if self.use_volume:
                features.update(self._extract_volume_features(df_numeric))
            
            # Context and other structural features
            features.update(self._extract_context_features(df_numeric))
            features.update(self._extract_multi_timeframe_features(df_numeric))
            features.update(self._extract_bollinger_features(df_numeric))
            features.update(self._extract_sr_features(df_numeric))
            features.update(self._extract_candle_features(df_numeric))
            
            # Cache market features
            self._last_features = features.copy()
            self._last_hash = current_hash
            
            # Add signal specific features
            features.update(self._extract_signal_features(signal, df_numeric))
            
            # 3. Post-Processing & Sanitization
            clean_features = {}
            for key, value in features.items():
                if np.isfinite(value):
                    clean_features[key] = float(value)
                else:
                    clean_features[key] = 0.0
            
            self._sanitize_features(clean_features)
            
            return clean_features
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            return None

    def _sanitize_features(self, features: Dict[str, float]) -> None:
        """In-place sanitization and clipping of feature values."""
        for key in features:
            val = features[key]
            # Clip ratios to avoid outliers
            if "ratio" in key or "normalized" in key or "position" in key or "zscore" in key:
                features[key] = np.clip(val, -5.0, 5.0)
            elif "pct" in key or "distance" in key:
                features[key] = np.clip(val, -1.0, 1.0)
            elif "rsi" in key:
                features[key] = np.clip(val, 0.0, 100.0)
            elif "volatility" in key:
                features[key] = np.clip(val, 0.0, 2.0)
            elif "momentum" in key:
                features[key] = np.clip(val, -1.0, 1.0)

    # =========================================================================
    # PRICE FEATURES
    # =========================================================================
    def _extract_price_features(self, df: pd.DataFrame, signal: Dict) -> Dict:
        """Extract price-based features including Log Returns."""
        close = df['close']
        high = df['high']
        low = df['low']
        
        current_price = float(signal.get('price', close.iloc[-1]))
        
        # Safe Logarithmic returns (prevents RuntimeWarning)
        # We ensure the ratio is always positive before logging
        price_ratio = close / (close.shift(1) + self.EPSILON)
        log_ret = np.log(np.maximum(price_ratio, self.EPSILON)).fillna(0)
        
        # Rolling quantiles
        window = min(50, len(close))
        roll = close.rolling(window=window)
        q50 = roll.quantile(0.5).iloc[-1]
        q75 = roll.quantile(0.75).iloc[-1]
        q25 = roll.quantile(0.25).iloc[-1]
        
        # Recent High/Low
        recent_window = min(20, len(high))
        recent_high = high.iloc[-recent_window:].max()
        recent_low = low.iloc[-recent_window:].min()
        
        # Skewness
        ret_skew = log_ret.rolling(window=20).skew().iloc[-1]
        
        return {
            'price_distance_from_high': (recent_high - current_price) / (recent_high + self.EPSILON),
            'price_distance_from_low': (current_price - recent_low) / (recent_low + self.EPSILON),
            'price_vs_median': (current_price - q50) / (q50 + self.EPSILON),
            'price_vs_q75': (current_price - q75) / (q75 + self.EPSILON),
            'price_vs_q25': (current_price - q25) / (q25 + self.EPSILON),
            'high_low_range': (recent_high - recent_low) / (recent_low + self.EPSILON),
            'returns_skew': 0.0 if np.isnan(ret_skew) else ret_skew,
        }
    
    # =========================================================================
    # TECHNICAL INDICATOR FEATURES
    # =========================================================================
    def _extract_technical_features(self, df: pd.DataFrame) -> Dict:
        """Extract technical features including ADX and smoothed RSI."""
        close = df['close']
        high = df['high']
        low = df['low']
        current_price = close.iloc[-1]
        
        # --- EMA ---
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()
        
        # --- MACD ---
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        hist = macd_line - signal_line
        
        # --- RSI (Wilder's Smoothing) ---
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / (avg_loss + self.EPSILON)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # --- ATR ---
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()
        current_atr = atr.iloc[-1]

        # --- ADX (Trend Strength) ---
        up = high - high.shift(1)
        down = low.shift(1) - low
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        
        # Handle division by zero in ADX calc
        atr_s = atr + self.EPSILON
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_s)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_s)
        
        dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + self.EPSILON)
        adx = dx.ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        
        # --- Stoch RSI ---
        min_rsi = rsi.rolling(14).min()
        max_rsi = rsi.rolling(14).max()
        stoch_rsi = (rsi - min_rsi) / ((max_rsi - min_rsi) + self.EPSILON)
        
        return {
            'ema_fast_distance': (current_price - ema_12.iloc[-1]) / (current_price + self.EPSILON),
            'ema_slow_distance': (current_price - ema_26.iloc[-1]) / (current_price + self.EPSILON),
            'ema_long_distance': (current_price - ema_50.iloc[-1]) / (current_price + self.EPSILON),
            'ema_trend': 1 if ema_12.iloc[-1] > ema_26.iloc[-1] else -1,
            'ema_spread': (ema_12.iloc[-1] - ema_26.iloc[-1]) / (current_price + self.EPSILON),
            'rsi': current_rsi,
            'rsi_normalized': (current_rsi - 50.0) / 50.0,
            'rsi_overbought': 1 if current_rsi > 70 else 0,
            'rsi_oversold': 1 if current_rsi < 30 else 0,
            'stoch_rsi': stoch_rsi.iloc[-1] if not np.isnan(stoch_rsi.iloc[-1]) else 0.5,
            'atr_pct': current_atr / (current_price + self.EPSILON),
            'macd_value': macd_line.iloc[-1] / (current_price + self.EPSILON),
            'macd_histogram': hist.iloc[-1] / (current_price + self.EPSILON),
            'macd_signal': 1 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1,
            'macd_crossover': 1 if (hist.iloc[-1] > 0 and hist.iloc[-2] <= 0) else (-1 if (hist.iloc[-1] < 0 and hist.iloc[-2] >= 0) else 0),
            'adx': adx,
            'trend_strength': 1 if adx > 25 else 0,
        }
    
    # =========================================================================
    # VOLUME FEATURES
    # =========================================================================
    def _extract_volume_features(self, df: pd.DataFrame) -> Dict:
        """Extract volume features including VWAP deviation."""
        volume = df['volume']
        close = df['close']
        
        if len(volume) < 20:
            return {}
        
        vol_ma_20 = volume.rolling(20).mean()
        vol_ma_5 = volume.rolling(5).mean()
        
        # VWAP
        vwap_window = min(len(df), 24)
        vwap_num = (close * volume).rolling(window=vwap_window).sum()
        vwap_denom = volume.rolling(window=vwap_window).sum()
        vwap = vwap_num / (vwap_denom + self.EPSILON)
        
        current_vol = volume.iloc[-1]
        vol_std = volume.rolling(20).std().iloc[-1]
        vol_z = (current_vol - vol_ma_20.iloc[-1]) / (vol_std + self.EPSILON)
        
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        
        return {
            'volume_ratio': current_vol / (vol_ma_20.iloc[-1] + self.EPSILON),
            'volume_ratio_short': current_vol / (vol_ma_5.iloc[-1] + self.EPSILON),
            'volume_trend': 1 if vol_ma_5.iloc[-1] > vol_ma_20.iloc[-1] else -1,
            'volume_zscore': vol_z,
            'volume_spike': 1 if vol_z > 2.0 else 0,
            'volume_dry': 1 if vol_z < -1.0 else 0,
            'obv_trend': 1 if obv.iloc[-1] > obv.iloc[-5] else -1,
            'vpt_trend': 1 if (close.pct_change() * volume).cumsum().iloc[-1] > 0 else -1,
            'vwap_distance': (close.iloc[-1] - vwap.iloc[-1]) / (vwap.iloc[-1] + self.EPSILON),
        }
    
    # =========================================================================
    # SIGNAL FEATURES
    # =========================================================================
    def _extract_signal_features(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """Extract features from the trading signal parameters."""
        side = str(signal.get('side', 'BUY')).upper()
        strength = float(signal.get('strength', 50.0))
        entry_price = float(signal.get('price', df['close'].iloc[-1]))
        sl = float(signal.get('stop_loss', 0.0))
        tp = float(signal.get('take_profit', 0.0))
        
        side_mult = 1 if side == 'BUY' else -1
        
        if side == 'BUY':
            sl_dist = (entry_price - sl) / entry_price if sl > 0 else 0.02
            tp_dist = (tp - entry_price) / entry_price if tp > 0 else 0.05
        else:
            sl_dist = (sl - entry_price) / entry_price if sl > 0 else 0.02
            tp_dist = (entry_price - tp) / entry_price if tp > 0 else 0.05
            
        rr = tp_dist / (sl_dist + self.EPSILON)
        
        return {
            'signal_side': side_mult,
            'signal_strength': strength / 100.0,
            'signal_strength_high': 1 if strength >= 80 else 0,
            'signal_strength_low': 1 if strength < 40 else 0,
            'sl_distance_pct': sl_dist,
            'tp_distance_pct': tp_dist,
            'risk_reward_ratio': rr,
            'is_momentum': 1 if signal.get('is_momentum', False) else 0,
        }
    
    # =========================================================================
    # CONTEXT FEATURES
    # =========================================================================
    def _extract_context_features(self, df: pd.DataFrame) -> Dict:
        """Extract market regime and volatility features."""
        close = df['close']
        
        # Log returns
        price_ratio = close / (close.shift(1) + self.EPSILON)
        log_ret = np.log(np.maximum(price_ratio, self.EPSILON)).fillna(0)
        
        vol_short = log_ret.rolling(5).std()
        vol_long = log_ret.rolling(20).std()
        
        # Efficiency Ratio
        change = (close - close.shift(10)).abs()
        volatility_sum = (close - close.shift(1)).abs().rolling(10).sum()
        efficiency_ratio = change / (volatility_sum + self.EPSILON)
        
        highs = df['high'].iloc[-10:]
        lows = df['low'].iloc[-10:]
        hh = (highs > highs.shift(1)).sum() / 9.0
        ll = (lows < lows.shift(1)).sum() / 9.0
        
        return {
            'volatility': vol_long.iloc[-1],
            'volatility_20': vol_long.iloc[-1],
            'volatility_expanding': 1 if vol_short.iloc[-1] > vol_long.iloc[-1] else 0,
            'efficiency_ratio': efficiency_ratio.iloc[-1],
            'momentum_5bar': (close.iloc[-1] / (close.iloc[-6] + self.EPSILON) - 1) if len(close) > 6 else 0,
            'momentum_10bar': (close.iloc[-1] / (close.iloc[-11] + self.EPSILON) - 1) if len(close) > 11 else 0,
            'momentum_20bar': (close.iloc[-1] / (close.iloc[-21] + self.EPSILON) - 1) if len(close) > 21 else 0,
            'trend_consistency': efficiency_ratio.iloc[-1],
            'higher_highs_ratio': hh,
            'lower_lows_ratio': ll,
        }
    
    # =========================================================================
    # MULTI-TIMEFRAME FEATURES
    # =========================================================================
    def _extract_multi_timeframe_features(self, df: pd.DataFrame) -> Dict:
        """Extract relationships between different time constants."""
        close = df['close']
        ema_5 = close.ewm(span=5, adjust=False).mean()
        ema_13 = close.ewm(span=13, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()
        ema_200 = close.ewm(span=200, adjust=False).mean() if len(close) > 200 else ema_50
        
        curr = close.iloc[-1]
        
        bullish_alignment = (ema_5.iloc[-1] > ema_13.iloc[-1] > ema_50.iloc[-1])
        bearish_alignment = (ema_5.iloc[-1] < ema_13.iloc[-1] < ema_50.iloc[-1])
        
        roc_short = close.pct_change(3)
        roc_long = close.pct_change(10)
        
        return {
            'mtf_relative_trend_short': (ema_5.iloc[-1] / (ema_13.iloc[-1] + self.EPSILON)) - 1,
            'mtf_relative_trend_medium': (ema_13.iloc[-1] / (ema_50.iloc[-1] + self.EPSILON)) - 1,
            'mtf_relative_trend_long': (ema_50.iloc[-1] / (ema_200.iloc[-1] + self.EPSILON)) - 1,
            'mtf_divergence': 1 if (roc_short.iloc[-1] < 0 and roc_long.iloc[-1] > 0) else 0,
            'mtf_momentum_short': roc_short.iloc[-1],
            'mtf_momentum_medium': roc_long.iloc[-1],
            'mtf_momentum_long': close.pct_change(20).iloc[-1],
            'mtf_momentum_ratio': roc_short.iloc[-1] / (np.abs(roc_long.iloc[-1]) + self.EPSILON),
            'mtf_trend_alignment': 1.0 if bullish_alignment else (-1.0 if bearish_alignment else 0.0),
            'mtf_trend_acceleration': (ema_5.diff().iloc[-1] - ema_13.diff().iloc[-1]) / (curr + self.EPSILON),
        }
    
    # =========================================================================
    # BOLLINGER BANDS FEATURES
    # =========================================================================
    def _extract_bollinger_features(self, df: pd.DataFrame) -> Dict:
        """Extract Bollinger Band metrics."""
        close = df['close']
        if len(close) < 20: return {}
            
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        
        curr = close.iloc[-1]
        u_val = upper.iloc[-1]
        l_val = lower.iloc[-1]
        
        width = (u_val - l_val) / (sma.iloc[-1] + self.EPSILON)
        avg_width = ((upper - lower) / (sma + self.EPSILON)).rolling(50).mean().iloc[-1]
        
        pct_b = (curr - l_val) / (u_val - l_val + self.EPSILON)
        
        return {
            'bb_position': (pct_b - 0.5) * 2,
            'bb_percent_b': pct_b,
            'bb_width': width,
            'bb_width_vs_avg': width / (avg_width + self.EPSILON),
            'bb_squeeze': 1 if width < (avg_width * 0.8) else 0,
            'bb_expansion': 1 if width > (avg_width * 1.2) else 0,
            'bb_touch_upper': 1 if curr >= (u_val * 0.995) else 0,
            'bb_touch_lower': 1 if curr <= (l_val * 1.005) else 0,
        }
    
    # =========================================================================
    # SUPPORT/RESISTANCE FEATURES
    # =========================================================================
    def _extract_sr_features(self, df: pd.DataFrame) -> Dict:
        """Extract dynamic S/R features."""
        close = df['close']
        high = df['high']
        low = df['low']
        curr = close.iloc[-1]
        
        prev_h = high.iloc[-2]
        prev_l = low.iloc[-2]
        prev_c = close.iloc[-2]
        pivot = (prev_h + prev_l + prev_c) / 3
        
        window = 20
        swing_h = high.iloc[-window:].max()
        swing_l = low.iloc[-window:].min()
        
        range_pos = (curr - swing_l) / (swing_h - swing_l + self.EPSILON)
        
        return {
            'sr_distance_to_pivot': (curr - pivot) / (curr + self.EPSILON),
            'sr_distance_to_r1': (2*pivot - prev_l - curr) / (curr + self.EPSILON),
            'sr_distance_to_s1': (curr - (2*pivot - prev_h)) / (curr + self.EPSILON),
            'sr_distance_to_swing_high': (swing_h - curr) / (curr + self.EPSILON),
            'sr_distance_to_swing_low': (curr - swing_l) / (curr + self.EPSILON),
            'sr_position_in_range': range_pos,
            'sr_near_resistance': 1 if (swing_h - curr)/curr < 0.005 else 0,
            'sr_near_support': 1 if (curr - swing_l)/curr < 0.005 else 0,
            'sr_breakout_high': 1 if curr >= swing_h else 0,
            'sr_breakout_low': 1 if curr <= swing_l else 0,
        }
    
    # =========================================================================
    # CANDLESTICK PATTERN FEATURES
    # =========================================================================
    def _extract_candle_features(self, df: pd.DataFrame) -> Dict:
        """Extract candlestick morphology features."""
        open_p = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        
        c_o = open_p.iloc[-1]
        c_c = close.iloc[-1]
        c_h = high.iloc[-1]
        c_l = low.iloc[-1]
        
        body = abs(c_c - c_o)
        rng = c_h - c_l + self.EPSILON
        
        u_wick = c_h - max(c_c, c_o)
        l_wick = min(c_c, c_o) - c_l
        
        body_pct = body / rng
        u_wick_pct = u_wick / rng
        l_wick_pct = l_wick / rng
        
        avg_body = abs(close - open_p).rolling(10).mean().iloc[-1] + self.EPSILON
        body_rel = body / avg_body
        
        bullish = 1 if c_c > c_o else 0
        
        # Engulfing pattern
        p_o = open_p.iloc[-2]
        p_c = close.iloc[-2]
        engulfing = 0
        if bullish and p_c < p_o and c_c > p_o and c_o < p_c:
            engulfing = 1
        elif not bullish and p_c > p_o and c_c < p_o and c_o > p_c:
            engulfing = -1
            
        return {
            'candle_body_ratio': body_pct,
            'candle_upper_wick_ratio': u_wick_pct,
            'candle_lower_wick_ratio': l_wick_pct,
            'candle_is_bullish': 1 if bullish else -1,
            'candle_is_doji': 1 if body_pct < 0.1 else 0,
            'candle_is_hammer': 1 if l_wick_pct > 0.6 and u_wick_pct < 0.1 else 0,
            'candle_is_shooting_star': 1 if u_wick_pct > 0.6 and l_wick_pct < 0.1 else 0,
            'candle_is_marubozu': 1 if body_pct > 0.9 else 0,
            'candle_bullish_engulfing': 1 if engulfing == 1 else 0,
            'candle_bearish_engulfing': 1 if engulfing == -1 else 0,
            'candle_consecutive_bullish': 0,
            'candle_consecutive_bearish': 0,
            'candle_three_bullish': 0,
            'candle_three_bearish': 0,
            'candle_body_vs_avg': body_rel,
        }
    
    # =========================================================================
    # TRAINING DATA PREPARATION
    # =========================================================================
    def prepare_features_for_training(self, trades_data: List[Dict]) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        Prepare features and labels from trade history with robust scaling.
        """
        try:
            if not trades_data or len(trades_data) < ml_config.min_samples_to_train:
                logger.warning(f"⚠️ Insufficient trades for training: {len(trades_data)}")
                return None
            
            valid_trades = [t for t in trades_data if isinstance(t.get('features'), dict) and t['features']]
            
            if len(valid_trades) < ml_config.min_samples_to_train:
                return None
            
            X = pd.DataFrame([t['features'] for t in valid_trades])
            y = pd.Series([1 if float(t.get('pnl', 0)) > 0 else 0 for t in valid_trades])
            
            X.fillna(X.median(), inplace=True)
            X.fillna(0, inplace=True)
            X.replace([np.inf, -np.inf], 0, inplace=True)
            
            # Robust Winsorization
            for col in X.select_dtypes(include=[np.number]).columns:
                lower = X[col].quantile(0.01)
                upper = X[col].quantile(0.99)
                X[col] = X[col].clip(lower, upper)
            
            logger.success(f"✅ Prepared {len(X)} samples with {len(X.columns)} features")
            return X, y
            
        except Exception as e:
            logger.error(f"❌ Data preparation failed: {e}")
            return None
    
    def get_feature_names(self) -> List[str]:
        """Dynamic feature name retrieval."""
        dummy_df = pd.DataFrame(np.random.randn(50, 5) * 100 + 100, 
                              columns=['open', 'high', 'low', 'close', 'volume'])
        dummy_df['volume'] = abs(dummy_df['volume']) * 1000
        dummy_signal = {'side': 'BUY', 'strength': 50, 'price': 100}
        
        old_cache = self._last_hash
        self._last_hash = None
        
        feats = self.extract_features(dummy_df, dummy_signal)
        
        self._last_hash = old_cache
        
        if feats:
            return sorted(list(feats.keys())) # Return sorted to ensure consistency
        return []

    def get_feature_metadata(self) -> List[Dict[str, Any]]:
        names = self.get_feature_names()
        meta = []
        for name in names:
            meta.append({
                'name': name,
                'type': 'continuous' if 'is_' not in name else 'binary',
                'description': "Auto-generated elite feature"
            })
        return meta
    
    def get_feature_count(self) -> int:
        return len(self.get_feature_names())
    
    def clear_cache(self):
        self._cache.clear()
        self._last_hash = None
        self._last_features = None
        logger.debug("🧹 Feature cache cleared")

# Global instance
feature_engineer = FeatureEngineer()