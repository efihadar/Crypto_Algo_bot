# data_feeds.py
"""
Data Feeds Module - Bybit Market Data Provider
Provides market data fetching for USDT Perpetual contracts:
- Kline/Candlestick data
- Current prices
- 24h statistics
- Orderbook data
- Funding rates

Features:
- Rate limiting
- Request caching
- Retry logic for transient failures
- Thread-safe operations
"""
import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone
from functools import lru_cache
import pandas as pd
from loguru import logger

try:
    from pybit.unified_trading import HTTP
    PYBIT_AVAILABLE = True
except ImportError:
    logger.error("❌ pybit not installed. Please add 'pybit' to requirements.")
    PYBIT_AVAILABLE = False
    HTTP = None

# ============================================================
# 📁 PATHS & CONFIG LOADER
# ============================================================
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

try:
    from config_loader import get_config_loader
    CONFIG_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ config_loader missing — using defaults")
    get_config_loader = None
    CONFIG_AVAILABLE = False

# ============================================================
# 🔧 BYBIT CONFIG (USDT Perpetual / linear)
# ============================================================
class BybitConfig:
    """Configuration container for Bybit API settings."""
    
    API_KEY: str = os.getenv("BYBIT_API_KEY", "")
    API_SECRET: str = os.getenv("BYBIT_API_SECRET", "")
    TESTNET: bool = os.getenv("BYBIT_TESTNET", "false").lower() in ("1", "true", "yes", "y")
    
    # Request settings
    RECV_WINDOW: int = 20000
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    
    # Cache settings
    PRICE_CACHE_TTL: int = 5  # seconds
    KLINE_CACHE_TTL: int = 30  # seconds
    STATS_CACHE_TTL: int = 60  # seconds

    @classmethod
    def rate_limit(cls) -> int:
        """
        Get API rate limit from config.
        
        Returns:
            Calls per second limit
        """
        if CONFIG_AVAILABLE and get_config_loader:
            try:
                cfg = get_config_loader()
                return cfg.get("api", "API_RATE_LIMIT_CALLS_PER_SECOND", 5, int)
            except Exception as e:
                logger.debug(f"⚠️ rate_limit from config failed: {e}")
        return 5

    @classmethod
    def is_configured(cls) -> bool:
        """Check if API credentials are configured."""
        return bool(cls.API_KEY and cls.API_SECRET)

# ============================================================
# ⏱️ THREAD-SAFE RATE LIMITER
# ============================================================
class RateLimiter:
    """Thread-safe rate limiter for API calls."""

    def __init__(self, calls_per_second: Optional[int] = None):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_second: Maximum calls per second (default from config)
        """
        self.calls_per_second = calls_per_second or BybitConfig.rate_limit()
        self._last_call = 0.0
        self._lock = threading.Lock()
        logger.debug(f"📊 Rate limiter initialized: {self.calls_per_second} calls/sec")

    def wait(self) -> None:
        """Sleep if needed to respect rate limit."""
        with self._lock:
            now = time.time()
            min_interval = 1.0 / float(self.calls_per_second)
            elapsed = now - self._last_call

            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                time.sleep(sleep_time)

            self._last_call = time.time()

    def set_rate(self, calls_per_second: int) -> None:
        """Update the rate limit."""
        with self._lock:
            self.calls_per_second = max(1, calls_per_second)
            logger.debug(f"📊 Rate limit updated: {self.calls_per_second} calls/sec")

# ============================================================
# 🔌 BYBIT CLIENT MANAGER
# ============================================================
class BybitClientManager:
    """
    Manages Bybit HTTP client with lazy initialization and reconnection.
    """

    def __init__(self):
        self._client: Optional[HTTP] = None
        self._lock = threading.Lock()
        self._last_error: Optional[str] = None
        self._error_count: int = 0
        self._initialized: bool = False
    
    @property
    def client(self) -> Optional[HTTP]:
        """Get or create the Bybit HTTP client."""
        if not PYBIT_AVAILABLE:
            return None
            
        if not BybitConfig.is_configured():
            if self._last_error != "not_configured":
                logger.error("❌ BYBIT_API_KEY / BYBIT_API_SECRET not configured")
                self._last_error = "not_configured"
            return None
        
        with self._lock:
            if self._client is None:
                self._initialize_client()
            return self._client
    
    def _initialize_client(self) -> None:
        """Initialize the HTTP client."""
        try:
            self._client = HTTP(
                api_key=BybitConfig.API_KEY,
                api_secret=BybitConfig.API_SECRET,
                testnet=BybitConfig.TESTNET,
                recv_window=BybitConfig.RECV_WINDOW,
            )
            self._initialized = True
            self._error_count = 0
            self._last_error = None
            logger.success(
                f"🔌 Bybit client initialized (testnet={BybitConfig.TESTNET})"
            )
        except Exception as e:
            self._last_error = str(e)
            self._error_count += 1
            logger.error(f"❌ Failed to initialize Bybit client: {e}")
    
    def reconnect(self) -> bool:
        """Force reconnection of the client."""
        with self._lock:
            self._client = None
            self._initialize_client()
            return self._client is not None
    
    def is_ready(self) -> bool:
        """Check if client is ready for use."""
        return self.client is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status information."""
        return {
            "initialized": self._initialized,
            "ready": self._client is not None,
            "error_count": self._error_count,
            "last_error": self._last_error,
            "testnet": BybitConfig.TESTNET,
        }

# Global instances
rate_limiter = RateLimiter()
client_manager = BybitClientManager()

# ============================================================
# 📦 CACHE MANAGER
# ============================================================
class CacheManager:
    """Simple thread-safe cache with TTL."""
    
    def __init__(self):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str, ttl: int = 60) -> Optional[Any]:
        """
        Get cached value if not expired.
        """
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < ttl:
                    return value
                else:
                    del self._cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Store value in cache."""
        with self._lock:
            self._cache[key] = (value, time.time())
    
    def clear(self, prefix: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            prefix: If provided, only clear keys starting with this prefix
            
        Returns:
            Number of entries cleared
        """
        with self._lock:
            if prefix:
                keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
                for k in keys_to_delete:
                    del self._cache[k]
                return len(keys_to_delete)
            else:
                count = len(self._cache)
                self._cache.clear()
                return count

cache = CacheManager()

# ============================================================
# 🧩 HELPERS
# ============================================================
def detect_asset_type(symbol: str) -> str:
    """
    Detect the type of asset from symbol.
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Asset type ('crypto', 'unknown')
    """
    s = symbol.upper()
    crypto_markers = (
        "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOT", "LINK", 
        "DOGE", "TRX", "LTC", "USDT", "MATIC", "AVAX", "ATOM",
        "UNI", "SHIB", "APT", "ARB", "OP", "SUI", "SEI", "PEPE",
        "WLD", "INJ", "FTM", "NEAR", "ALGO", "FIL", "ICP", "VET"
    )
    if any(m in s for m in crypto_markers):
        return "crypto"
    return "unknown"

def map_timeframe(tf: str) -> str:
    """
    Map human-readable timeframe to Bybit interval format.
    
    Args:
        tf: Timeframe string (e.g., '15m', '1h', '4h')
        
    Returns:
        Bybit interval string
    """
    mapping = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "6h": "360",
        "12h": "720",
        "1d": "D",
        "1w": "W",
        "1M": "M",
    }
    return mapping.get(tf.lower(), tf)

def normalize_symbol(symbol: str) -> str:
    """
    Normalize symbol to Bybit format.
    
    Args:
        symbol: Raw symbol
        
    Returns:
        Normalized symbol with USDT suffix
    """
    s = symbol.upper().strip()
    if not s.endswith("USDT"):
        s += "USDT"
    return s

def _make_request_with_retry(func,*args,max_retries: int = None,retry_delay: float = None,**kwargs) -> Optional[Dict]:
    """
    Execute API request with retry logic.
    """
    max_retries = max_retries or BybitConfig.MAX_RETRIES
    retry_delay = retry_delay or BybitConfig.RETRY_DELAY
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            rate_limiter.wait()
            response = func(*args, **kwargs)
            
            if isinstance(response, dict):
                if response.get("retCode") == 0:
                    return response
                else:
                    error_msg = response.get("retMsg", "Unknown error")
                    logger.warning(f"⚠️ API error (attempt {attempt + 1}): {error_msg}")
                    last_error = error_msg
            else:
                return response
                
        except Exception as e:
            last_error = str(e)
            logger.warning(f"⚠️ Request failed (attempt {attempt + 1}/{max_retries}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay * (attempt + 1))
    
    logger.error(f"❌ Request failed after {max_retries} attempts: {last_error}")
    return None

# ============================================================
# 📈 KLINES (USDT PERPETUAL / LINEAR)
# ============================================================
def get_candles(symbol: str,limit: int = 200,interval: str = "15",use_cache: bool = True,) -> Optional[pd.DataFrame]:
    """
    Fetch kline/candlestick data from Bybit.
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT" or "BTC")
        limit: Number of candles to fetch (max 200)
        interval: Bybit interval format ("1","5","15","60","240","D"...)
        use_cache: Whether to use cached data if available
        
    Returns:
        DataFrame with time, open, high, low, close, volume, turnover
        or None on failure
    """
    client = client_manager.client
    if not client:
        logger.error("❌ Bybit client not available — cannot fetch candles")
        return None

    try:
        symbol = normalize_symbol(symbol)
        
        # Check cache
        cache_key = f"klines_{symbol}_{interval}_{limit}"
        if use_cache:
            cached = cache.get(cache_key, BybitConfig.KLINE_CACHE_TTL)
            if cached is not None:
                logger.debug(f"📊 Using cached klines for {symbol}")
                return cached.copy()

        # Fetch from API
        response = _make_request_with_retry(
            client.get_kline,
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=limit,
        )

        if not response:
            return None

        rows = response.get("result", {}).get("list", [])
        if not rows:
            logger.warning(f"⚠️ No kline data returned for {symbol}")
            return None

        # Build DataFrame
        df = pd.DataFrame(
            rows,
            columns=["time", "open", "high", "low", "close", "volume", "turnover"],
        )

        # Reverse to chronological order (oldest first)
        df = df.iloc[::-1].reset_index(drop=True)

        # Convert types
        df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df["low"] = pd.to_numeric(df["low"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")

        # Cache result
        cache.set(cache_key, df.copy())
        
        logger.debug(f"📊 Fetched {len(df)} candles for {symbol} ({interval})")
        return df

    except Exception as e:
        logger.error(f"❌ get_candles error for {symbol}: {e}")
        return None

def get_market_data(symbol: str,count: int = 200,timeframe: str = "15m",use_cache: bool = True,) -> Optional[pd.DataFrame]:
    """
    Convenience wrapper that accepts human-readable timeframe.
    
    Args:
        symbol: Trading symbol
        count: Number of candles
        timeframe: Human-readable timeframe (e.g., '15m', '1h', '4h')
        use_cache: Whether to use cache
        
    Returns:
        DataFrame ready for strategy use
    """
    if detect_asset_type(symbol) != "crypto":
        logger.warning(f"⚠️ {symbol} not detected as crypto — skipping")
        return None

    interval = map_timeframe(timeframe)
    df = get_candles(symbol, limit=count, interval=interval, use_cache=use_cache)
    
    if df is None or df.empty:
        logger.warning(f"⚠️ No valid candles for {symbol} ({timeframe})")
        return None

    return df

def get_multi_timeframe_data(symbol: str,timeframes: List[str] = None,count: int = 100,) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Fetch data for multiple timeframes at once.
    
    Args:
        symbol: Trading symbol
        timeframes: List of timeframes (default: ['15m', '1h', '4h'])
        count: Number of candles per timeframe
        
    Returns:
        Dict mapping timeframe to DataFrame
    """
    if timeframes is None:
        timeframes = ["15m", "1h", "4h"]
    
    result = {}
    for tf in timeframes:
        result[tf] = get_market_data(symbol, count=count, timeframe=tf)
        
    return result

# ============================================================
# 💰 PRICE & 24H STATS (USDT PERP / LINEAR)
# ============================================================
def get_current_price(symbol: str, use_cache: bool = True) -> Optional[float]:
    """
    Get the last traded price for a USDT Perpetual contract.
    
    Args:
        symbol: Trading symbol
        use_cache: Whether to use cached price
        
    Returns:
        Current price as float or None
    """
    client = client_manager.client
    if not client:
        return None

    try:
        symbol = normalize_symbol(symbol)
        
        # Check cache
        cache_key = f"price_{symbol}"
        if use_cache:
            cached = cache.get(cache_key, BybitConfig.PRICE_CACHE_TTL)
            if cached is not None:
                return cached

        response = _make_request_with_retry(
            client.get_tickers,
            category="linear",
            symbol=symbol,
        )

        if not response:
            return None

        ticker_list = response.get("result", {}).get("list", [])
        if not ticker_list:
            logger.warning(f"⚠️ No ticker data for {symbol}")
            return None

        price = float(ticker_list[0]["lastPrice"])
        
        # Cache result
        cache.set(cache_key, price)
        
        return price

    except Exception as e:
        logger.error(f"❌ get_current_price error for {symbol}: {e}")
        return None

def get_prices_batch(symbols: List[str]) -> Dict[str, Optional[float]]:
    """
    Get current prices for multiple symbols efficiently.
    
    Args:
        symbols: List of trading symbols
        
    Returns:
        Dict mapping symbol to price
    """
    client = client_manager.client
    if not client:
        return {s: None for s in symbols}

    try:
        # Fetch all tickers at once
        rate_limiter.wait()
        response = client.get_tickers(category="linear")
        
        if response.get("retCode") != 0:
            logger.error(f"❌ get_tickers error: {response.get('retMsg')}")
            return {s: None for s in symbols}

        # Build lookup dict
        ticker_dict = {}
        for ticker in response.get("result", {}).get("list", []):
            sym = ticker.get("symbol", "")
            price = ticker.get("lastPrice")
            if sym and price:
                ticker_dict[sym] = float(price)

        # Map requested symbols to prices
        result = {}
        for symbol in symbols:
            normalized = normalize_symbol(symbol)
            result[symbol] = ticker_dict.get(normalized)
            
            # Cache individual prices
            if result[symbol] is not None:
                cache.set(f"price_{normalized}", result[symbol])

        return result

    except Exception as e:
        logger.error(f"❌ get_prices_batch error: {e}")
        return {s: None for s in symbols}

def get_24h_stats(symbol: str, use_cache: bool = True) -> Optional[Dict[str, float]]:
    """
    Get 24-hour statistics for a USDT Perpetual contract.
    
    Args:
        symbol: Trading symbol
        use_cache: Whether to use cached data
        
    Returns:
        Dict with last_price, high_24h, low_24h, volume_24h, turnover_24h, price_change_pct
        or None on failure
    """
    client = client_manager.client
    if not client:
        return None

    try:
        symbol = normalize_symbol(symbol)
        
        # Check cache
        cache_key = f"stats24h_{symbol}"
        if use_cache:
            cached = cache.get(cache_key, BybitConfig.STATS_CACHE_TTL)
            if cached is not None:
                return cached

        response = _make_request_with_retry(
            client.get_tickers,
            category="linear",
            symbol=symbol,
        )

        if not response:
            return None

        ticker_list = response.get("result", {}).get("list", [])
        if not ticker_list:
            logger.warning(f"⚠️ No 24h stats for {symbol}")
            return None

        t = ticker_list[0]
        stats = {
            "symbol": symbol,
            "last_price": float(t.get("lastPrice", 0)),
            "high_24h": float(t.get("highPrice24h", 0)),
            "low_24h": float(t.get("lowPrice24h", 0)),
            "volume_24h": float(t.get("volume24h", 0)),
            "turnover_24h": float(t.get("turnover24h", 0)),
            "price_change_pct": float(t.get("price24hPcnt", 0)) * 100.0,
            "bid_price": float(t.get("bid1Price", 0)),
            "ask_price": float(t.get("ask1Price", 0)),
            "open_interest": float(t.get("openInterest", 0)),
            "funding_rate": float(t.get("fundingRate", 0)) * 100.0,  # Convert to percentage
        }

        # Cache result
        cache.set(cache_key, stats)
        
        return stats

    except Exception as e:
        logger.error(f"❌ get_24h_stats error for {symbol}: {e}")
        return None

# ============================================================
# 📚 ORDERBOOK
# ============================================================
def get_orderbook(
    symbol: str,
    limit: int = 25,
) -> Optional[Dict[str, Any]]:
    """
    Get orderbook data for a symbol.
    
    Args:
        symbol: Trading symbol
        limit: Depth limit (1, 25, 50, 100, 200)
        
    Returns:
        Dict with bids, asks, and spread info
    """
    client = client_manager.client
    if not client:
        return None

    try:
        symbol = normalize_symbol(symbol)

        response = _make_request_with_retry(
            client.get_orderbook,
            category="linear",
            symbol=symbol,
            limit=limit,
        )

        if not response:
            return None

        result = response.get("result", {})
        bids = result.get("b", [])  # [[price, size], ...]
        asks = result.get("a", [])

        if not bids or not asks:
            return None

        # Calculate spread
        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0

        return {
            "symbol": symbol,
            "bids": [[float(p), float(s)] for p, s in bids],
            "asks": [[float(p), float(s)] for p, s in asks],
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_pct": spread_pct,
            "timestamp": result.get("ts"),
        }

    except Exception as e:
        logger.error(f"❌ get_orderbook error for {symbol}: {e}")
        return None

# ============================================================
# 💸 FUNDING RATES
# ============================================================
def get_funding_rate(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get current funding rate for a symbol.
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Dict with funding_rate, next_funding_time, etc.
    """
    client = client_manager.client
    if not client:
        return None

    try:
        symbol = normalize_symbol(symbol)

        response = _make_request_with_retry(
            client.get_tickers,
            category="linear",
            symbol=symbol,
        )

        if not response:
            return None

        ticker_list = response.get("result", {}).get("list", [])
        if not ticker_list:
            return None

        t = ticker_list[0]
        funding_rate = float(t.get("fundingRate", 0))
        
        return {
            "symbol": symbol,
            "funding_rate": funding_rate,
            "funding_rate_pct": funding_rate * 100,
            "next_funding_time": t.get("nextFundingTime"),
        }

    except Exception as e:
        logger.error(f"❌ get_funding_rate error for {symbol}: {e}")
        return None

def get_funding_history(
    symbol: str,
    limit: int = 50,
) -> Optional[List[Dict[str, Any]]]:
    """
    Get historical funding rates.
    
    Args:
        symbol: Trading symbol
        limit: Number of records
        
    Returns:
        List of funding rate records
    """
    client = client_manager.client
    if not client:
        return None

    try:
        symbol = normalize_symbol(symbol)

        response = _make_request_with_retry(
            client.get_funding_rate_history,
            category="linear",
            symbol=symbol,
            limit=limit,
        )

        if not response:
            return None

        records = response.get("result", {}).get("list", [])
        
        return [
            {
                "symbol": r.get("symbol"),
                "funding_rate": float(r.get("fundingRate", 0)),
                "funding_rate_pct": float(r.get("fundingRate", 0)) * 100,
                "funding_rate_timestamp": r.get("fundingRateTimestamp"),
            }
            for r in records
        ]

    except Exception as e:
        logger.error(f"❌ get_funding_history error for {symbol}: {e}")
        return None

# ============================================================
# 📋 AVAILABLE SYMBOLS
# ============================================================
def get_available_symbols(status: str = "Trading") -> List[str]:
    """
    Get list of available trading symbols.
    
    Args:
        status: Filter by status ('Trading', 'Settling', etc.)
        
    Returns:
        List of symbol strings
    """
    client = client_manager.client
    if not client:
        return []

    try:
        response = _make_request_with_retry(
            client.get_instruments_info,
            category="linear",
            status=status,
        )

        if not response:
            return []

        symbols = [
            item["symbol"]
            for item in response.get("result", {}).get("list", [])
        ]
        
        logger.debug(f"📋 Found {len(symbols)} available symbols")
        return symbols

    except Exception as e:
        logger.error(f"❌ get_available_symbols error: {e}")
        return []

# ============================================================
# 🔧 UTILITY FUNCTIONS
# ============================================================
def clear_cache(symbol: Optional[str] = None) -> int:
    """
    Clear cached data.
    
    Args:
        symbol: If provided, only clear cache for this symbol
        
    Returns:
        Number of cache entries cleared
    """
    if symbol:
        symbol = normalize_symbol(symbol)
        # Clear all cache entries for this symbol
        count = 0
        for prefix in ["price_", "klines_", "stats24h_"]:
            count += cache.clear(f"{prefix}{symbol}")
        return count
    else:
        return cache.clear()

def check_connection() -> bool:
    """
    Check if data feed connection is working.
    
    Returns:
        True if connection is healthy
    """
    try:
        client = client_manager.client
        if not client:
            return False
        
        # Try to fetch a simple ticker
        rate_limiter.wait()
        response = client.get_tickers(category="linear", symbol="BTCUSDT")
        
        return response.get("retCode") == 0
        
    except Exception as e:
        logger.debug(f"Connection check failed: {e}")
        return False

def get_data_feed_status() -> Dict[str, Any]:
    """
    Get comprehensive status of data feeds.
    
    Returns:
        Dict with status information
    """
    return {
        "client": client_manager.get_status(),
        "rate_limiter": {
            "calls_per_second": rate_limiter.calls_per_second,
        },
        "connection_ok": check_connection(),
        "pybit_available": PYBIT_AVAILABLE,
        "config_available": CONFIG_AVAILABLE,
    }

# ============================================================
# 🧪 TESTING / DEBUG
# ============================================================
if __name__ == "__main__":
    # Quick test when running directly
    logger.info("🧪 Testing data_feeds module...")
    
    status = get_data_feed_status()
    logger.info(f"Status: {status}")
    
    if status["connection_ok"]:
        # Test price fetch
        price = get_current_price("BTCUSDT")
        logger.info(f"BTC Price: ${price}")
        
        # Test 24h stats
        stats = get_24h_stats("BTCUSDT")
        logger.info(f"24h Stats: {stats}")
        
        # Test klines
        df = get_candles("BTCUSDT", limit=10)
        if df is not None:
            logger.info(f"Klines:\n{df.tail()}")
    else:
        logger.error("❌ Connection test failed")