# correlation_manager.py
"""
Position Correlation Manager
Prevents opening correlated positions (e.g., 5 altcoins = same trade)
"""
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from datetime import datetime, timedelta


class CorrelationManager:
    """Manages position correlation to avoid over-concentration."""
    
    def __init__(self, session, config_loader):
        """
        Args:
            session: BybitSession instance
            config_loader: ConfigLoader instance
        """
        self.session = session
        self.cfg = config_loader
        
        # Load settings
        section = "correlation"
        self.enabled = self.cfg.get(section, "ENABLED", True, bool)
        self.max_correlation = self.cfg.get(section, "MAX_CORRELATION", 0.7, float)
        self.lookback_days = self.cfg.get(section, "LOOKBACK_DAYS", 30, int)
        self.max_correlated_positions = self.cfg.get(section, "MAX_CORRELATED_POSITIONS", 3, int)
        self.correlation_interval = self.cfg.get(section, "CORRELATION_INTERVAL", "1h", str)
        
        # Cache
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._last_update: float = 0
        self._cache_ttl: int = 3600  # 1 hour
        
        logger.info(f"🔗 CorrelationManager initialized (enabled={self.enabled}, max_corr={self.max_correlation})")
    
    def calculate_correlation_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """
        Calculate correlation matrix for given symbols.
        Args:
            symbols: List of trading symbols
        Returns:
            Correlation matrix (DataFrame)
        """
        try:
            # Check cache
            now = time.time()
            if self._correlation_matrix is not None and (now - self._last_update) < self._cache_ttl:
                logger.debug("📊 Using cached correlation matrix")
                return self._correlation_matrix
            
            logger.info(f"📊 Calculating correlation matrix for {len(symbols)} symbols...")
            
            # Fetch price data
            end_time = int(time.time() * 1000)
            start_time = end_time - (self.lookback_days * 24 * 3600 * 1000)
            
            price_data = {}
            
            for symbol in symbols:
                try:
                    klines = self.session.get_klines(
                        symbol=symbol,
                        interval=self.correlation_interval,
                        limit=min(1000, self.lookback_days * 24)  # Max klines
                    )
                    
                    if not klines:
                        continue
                    
                    # Extract close prices
                    closes = [float(k[4]) for k in klines]  # Index 4 = close
                    price_data[symbol] = closes
                    
                    time.sleep(0.05)  # Rate limit
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to fetch data for {symbol}: {e}")
                    continue
            
            if len(price_data) < 2:
                logger.warning("⚠️  Not enough data to calculate correlation")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(price_data)
            
            # Align lengths (use shortest)
            min_len = min(len(v) for v in df.values.T)
            df = df.iloc[-min_len:]
            
            # Calculate returns
            returns = df.pct_change().dropna()
            
            # Calculate correlation
            corr_matrix = returns.corr()
            
            # Cache result
            self._correlation_matrix = corr_matrix
            self._last_update = now
            
            logger.success(f"✅ Correlation matrix calculated ({len(corr_matrix)} symbols)")
            
            return corr_matrix
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate correlation matrix: {e}")
            return pd.DataFrame()
    
    def get_correlated_symbols(self, symbol: str, open_positions: List[str],threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Find symbols correlated with given symbol.
        Args:
            symbol: Target symbol
            open_positions: List of currently open position symbols
            threshold: Correlation threshold (default: self.max_correlation)
        Returns:
            List of (symbol, correlation) tuples
        """
        if not self.enabled:
            return []
        
        threshold = threshold or self.max_correlation
        
        try:
            # Calculate correlation matrix if needed
            all_symbols = list(set([symbol] + open_positions))
            corr_matrix = self.calculate_correlation_matrix(all_symbols)
            
            if corr_matrix.empty or symbol not in corr_matrix.columns:
                return []
            
            # Get correlations with target symbol
            correlations = corr_matrix[symbol].drop(symbol)  # Exclude self
            
            # Filter by threshold
            high_corr = correlations[correlations.abs() >= threshold]
            
            # Return as list of tuples
            result = [(sym, corr) for sym, corr in high_corr.items() if sym in open_positions]
            result.sort(key=lambda x: abs(x[1]), reverse=True)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to get correlated symbols: {e}")
            return []
    
    def can_open_position(
        self, 
        symbol: str, 
        open_positions: List[str]
    ) -> Tuple[bool, str]:
        """
        Check if opening a position on symbol would violate correlation limits.
        
        Args:
            symbol: Symbol to check
            open_positions: List of currently open position symbols
            
        Returns:
            (can_open, reason) tuple
        """
        if not self.enabled:
            return True, "Correlation check disabled"
        
        if not open_positions:
            return True, "No open positions"
        
        try:
            # Get correlated positions
            correlated = self.get_correlated_symbols(symbol, open_positions)
            
            if not correlated:
                return True, "No correlated positions"
            
            # Check limit
            if len(correlated) >= self.max_correlated_positions:
                corr_str = ", ".join([f"{sym}({corr:.2f})" for sym, corr in correlated[:3]])
                reason = (
                    f"Too many correlated positions ({len(correlated)}/{self.max_correlated_positions}): "
                    f"{corr_str}"
                )
                logger.warning(f"🚫 {symbol} blocked: {reason}")
                return False, reason
            
            # Log correlated positions
            if correlated:
                corr_str = ", ".join([f"{sym}({corr:.2f})" for sym, corr in correlated])
                logger.info(f"⚠️ {symbol} has {len(correlated)} correlated position(s): {corr_str}")
            
            return True, "Correlation check passed"
            
        except Exception as e:
            logger.error(f"❌ Correlation check failed for {symbol}: {e}")
            # Fail open (allow trade on error)
            return True, f"Correlation check error: {e}"
    
    def get_correlation_report(self, open_positions: List[str]) -> Dict:
        """
        Generate correlation report for open positions.
        
        Args:
            open_positions: List of open position symbols
            
        Returns:
            Report dictionary
        """
        if not self.enabled or not open_positions:
            return {"enabled": self.enabled, "positions": 0}
        
        try:
            corr_matrix = self.calculate_correlation_matrix(open_positions)
            
            if corr_matrix.empty:
                return {"enabled": True, "positions": len(open_positions), "error": "No data"}
            
            # Find highest correlations
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    sym1 = corr_matrix.columns[i]
                    sym2 = corr_matrix.columns[j]
                    corr = corr_matrix.iloc[i, j]
                    
                    if abs(corr) >= self.max_correlation:
                        high_corr_pairs.append({
                            "pair": f"{sym1}-{sym2}",
                            "correlation": round(corr, 3)
                        })
            
            high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            # Calculate average correlation
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            avg_corr = upper_triangle.stack().mean()
            
            return {
                "enabled": True,
                "positions": len(open_positions),
                "avg_correlation": round(avg_corr, 3),
                "max_correlation": round(corr_matrix.max().max(), 3),
                "high_corr_pairs": high_corr_pairs[:5],  # Top 5
                "total_high_corr": len(high_corr_pairs),
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate correlation report: {e}")
            return {"enabled": True, "error": str(e)}
