# db_utils.py
"""
Professional Database Utilities for Trading Bot
Provides PostgreSQL database operations with comprehensive error handling and monitoring.
Production-ready for live trading environments.
"""
import os
import json
import time
import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from loguru import logger
from typing import List, Dict, Any, Optional, Union

# ============================================================
# DB CONFIG
# ============================================================
DB_HOST = os.getenv("DB_HOST", os.getenv("POSTGRES_HOST", "localhost"))
DB_PORT = int(os.getenv("DB_PORT", os.getenv("POSTGRES_PORT", "5432")))
DB_NAME = os.getenv("DB_NAME", os.getenv("POSTGRES_DB", "trading_db"))
DB_USER = os.getenv("DB_USER", os.getenv("POSTGRES_USER", "trader"))
DB_PASS = os.getenv("DB_PASS", os.getenv("POSTGRES_PASSWORD", "trader_pass"))

# Connection retry settings
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

logger.info(f"🗄️ DB Config: {DB_HOST}:{DB_PORT} → {DB_NAME}")

# ============================================================
# CONNECTION MANAGER
# ============================================================
@contextmanager
def get_connection(retries: int = MAX_RETRIES):
    """Get database connection with retry logic and comprehensive error handling"""
    conn = None
    last_error = None
    
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASS,
                host=DB_HOST,
                port=DB_PORT,
                connect_timeout=10,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5
            )
            break  # Success
            
        except psycopg2.OperationalError as e:
            last_error = e
            if attempt < retries - 1:
                logger.warning(
                    f"⚠️ DB connection attempt {attempt + 1}/{retries} failed: {e}"
                )
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"❌ DB connection failed after {retries} attempts: {e}")
                
        except Exception as e:
            last_error = e
            logger.error(f"❌ DB connection error: {e}")
            break
    
    try:
        yield conn
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.debug(f"⚠️ Failed to close connection: {e}")

def check_connection() -> bool:
    """
    Test database connectivity with comprehensive error handling.
    """
    try:
        with get_connection(retries=1) as conn:
            if not conn:
                return False
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                result = cur.fetchone()
                return result is not None and result[0] == 1
    except Exception as e:
        logger.debug(f"DB health check failed: {e}")
        return False

# ============================================================
# INIT TABLES (signals + executions + metrics + heartbeat)
# ============================================================
def init_tables() -> bool:
    """
    Initialize all required database tables with comprehensive error handling.
    Full professional version with all original columns + auto-fix for missing columns.
    """
    with get_connection() as conn:
        if not conn:
            logger.error("❌ Cannot initialize tables: no DB connection")
            return False
            
        try:
            cur = conn.cursor()

            # 1. SIGNALS TABLE
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id BIGSERIAL PRIMARY KEY,
                    ts_utc TIMESTAMPTZ DEFAULT NOW(),
                    symbol TEXT NOT NULL,
                    direction TEXT,
                    price DOUBLE PRECISION,
                    sl DOUBLE PRECISION,
                    tp DOUBLE PRECISION,
                    timeframe TEXT,
                    confidence DOUBLE PRECISION,
                    session_tag TEXT,
                    raw JSONB
                );
                CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts_utc DESC);
                CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
                CREATE INDEX IF NOT EXISTS idx_signals_confidence ON signals(confidence);
            """)
            
            # 2. TRADE EXECUTIONS TABLE (All columns restored)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trade_executions (
                    id BIGSERIAL PRIMARY KEY,
                    signal_id BIGINT,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    closed_at TIMESTAMPTZ,
                    symbol VARCHAR(20) NOT NULL,
                    direction VARCHAR(10),
                    entry_price NUMERIC(18,8),
                    exit_price NUMERIC(18,8),
                    size NUMERIC(18,8),
                    pnl NUMERIC(18,8),
                    status VARCHAR(20) DEFAULT 'opened',
                    ticket TEXT,
                    sl NUMERIC(18,8),
                    tp NUMERIC(18,8),
                    meta JSONB,
                    error_code INTEGER,
                    comment TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_exec_status ON trade_executions(status);
                CREATE INDEX IF NOT EXISTS idx_exec_symbol ON trade_executions(symbol);
                CREATE INDEX IF NOT EXISTS idx_exec_created ON trade_executions(created_at);
                CREATE INDEX IF NOT EXISTS idx_exec_closed ON trade_executions(closed_at);
            """)

            # 3. ACCOUNT METRICS (Restored + Auto-fix for created_at)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS account_metrics (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    balance NUMERIC(18,8),
                    equity NUMERIC(18,8),
                    profit NUMERIC(18,8),
                    margin NUMERIC(18,8) DEFAULT 0,
                    free_margin NUMERIC(18,8) DEFAULT 0,
                    margin_level NUMERIC(12,4) DEFAULT 0
                );
            """)

            # --- MIGRATION CHECK: Fix the 'created_at' DB error ---
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name='account_metrics' AND column_name='created_at'
                    ) THEN
                        ALTER TABLE account_metrics ADD COLUMN created_at TIMESTAMPTZ DEFAULT NOW();
                    END IF;
                END$$;
            """)

            # 4. BOT HEARTBEAT
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_heartbeat (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    ts_utc TIMESTAMPTZ DEFAULT NOW(),
                    status VARCHAR(20) NOT NULL,
                    cycle_count INTEGER DEFAULT 0,
                    step VARCHAR(100),
                    equity NUMERIC(18,8),
                    balance NUMERIC(18,8),
                    open_positions INTEGER,
                    message TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_heartbeat_timestamp ON bot_heartbeat(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_heartbeat_status ON bot_heartbeat(status);
            """)

            # 5. DAILY SUMMARY
            cur.execute("""
                CREATE TABLE IF NOT EXISTS daily_summary (
                    id BIGSERIAL PRIMARY KEY,
                    date DATE NOT NULL UNIQUE,
                    starting_balance NUMERIC(18,8),
                    ending_balance NUMERIC(18,8),
                    total_pnl NUMERIC(18,8),
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    win_rate NUMERIC(6,2),
                    max_drawdown NUMERIC(18,8),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_summary(date);
                CREATE INDEX IF NOT EXISTS idx_daily_created ON daily_summary(created_at);
            """)

            conn.commit()
            logger.success("✅ Full DB schema initialized and migrations applied")
            return True

        except Exception as e:
            logger.error(f"❌ init_tables failed: {e}")
            if conn:
                conn.rollback()
            return False

# ============================================================
# HEARTBEAT
# ============================================================
def write_heartbeat(
    status: str = "running",
    cycle_count: int = 0,
    step: str = "",
    equity: Optional[float] = None,
    balance: Optional[float] = None,
    open_positions: Optional[int] = None,
    message: str = "",
    ts_utc: Optional[str] = None,
    **kwargs  # Accept additional fields gracefully
) -> bool:
    """
    Write heartbeat to database with comprehensive error handling.
    
    Args:
        status: Bot status (running/error/stopped/idle)
        cycle_count: Current cycle number
        step: Current execution step
        equity: Account equity
        balance: Account balance
        open_positions: Number of open positions
        message: Additional message
        ts_utc: Timestamp (ISO format) - IGNORED, timestamp auto-generated
        
    Returns:
        True if successful, False otherwise
    """
    with get_connection() as conn:
        if not conn:
            logger.warning("⚠️ write_heartbeat: no DB connection")
            return False
            
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO bot_heartbeat 
                    (status, cycle_count, step, equity, balance, open_positions, message)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(status)[:20] if status else "unknown",
                        int(cycle_count) if cycle_count is not None else 0,
                        str(step)[:100] if step else "",
                        float(equity) if equity is not None else None,
                        float(balance) if balance is not None else None,
                        int(open_positions) if open_positions is not None else None,
                        str(message)[:500] if message else "",
                    )
                )
                conn.commit()
                logger.debug("✅ Heartbeat written to DB")
                return True
                
        except Exception as e:
            logger.error(f"❌ write_heartbeat failed: {e}")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return False

def get_latest_heartbeat() -> Optional[Dict[str, Any]]:
    """
    Get the most recent heartbeat with comprehensive error handling.
    Returns:
        Dict with heartbeat data or None
    """
    with get_connection() as conn:
        if not conn:
            return None
            
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("""
                SELECT * FROM bot_heartbeat
                ORDER BY timestamp DESC
                LIMIT 1;
            """)
            row = cur.fetchone()
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            logger.error(f"❌ get_latest_heartbeat failed: {e}")
            return None

def cleanup_old_heartbeats(days: int = 7) -> int:
    """
    Remove heartbeat records older than specified days with comprehensive error handling.
    Args:
        days: Number of days to keep
    Returns:
        Number of deleted records
    """
    with get_connection() as conn:
        if not conn:
            return 0
            
        try:
            cur = conn.cursor()
            cur.execute(
                """
                DELETE FROM bot_heartbeat
                WHERE timestamp < NOW() - INTERVAL '%s days'
                RETURNING id;
                """,
                (days,)
            )
            deleted = cur.rowcount
            conn.commit()
            
            if deleted > 0:
                logger.info(f"🧹 Cleaned up {deleted} old heartbeat records")
            return deleted
            
        except Exception as e:
            logger.error(f"❌ cleanup_old_heartbeats failed: {e}")
            if conn:
                conn.rollback()
            return 0

# ============================================================
# SIGNALS
# ============================================================
def insert_signal(
    symbol: str,
    direction: str,
    price: float,
    sl: float,
    tp: float,
    timeframe: str,
    confidence: float,
    session_tag: str,
    raw: Optional[Dict] = None
) -> Optional[int]:
    """
    Insert a new signal into the database with comprehensive validation.
    Returns: Signal ID if successful, None otherwise
    """
    try:
        # Validate inputs
        if not symbol or not isinstance(symbol, str):
            logger.error("❌ Invalid symbol for insert_signal")
            return None
        
        if not direction or not isinstance(direction, str):
            logger.error("❌ Invalid direction for insert_signal")
            return None
        
        if not isinstance(price, (int, float)) or price <= 0:
            logger.error("❌ Invalid price for insert_signal")
            return None
        
        with get_connection() as conn:
            if not conn:
                return None
                
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO signals 
                    (symbol, direction, price, sl, tp, timeframe, confidence, session_tag, raw)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (
                        symbol[:50],
                        direction[:20],
                        float(price),
                        float(sl) if sl is not None else None,
                        float(tp) if tp is not None else None,
                        str(timeframe)[:20] if timeframe else "",
                        float(confidence) if confidence is not None else 0.0,
                        str(session_tag)[:50] if session_tag else "",
                        json.dumps(raw or {})
                    )
                )
                signal_id = cur.fetchone()[0]
                conn.commit()
                logger.debug(f"✅ Signal inserted: ID={signal_id}, {symbol} {direction}")
                return signal_id

            except Exception as e:
                logger.error(f"❌ insert_signal failed: {e}")
                if conn:
                    conn.rollback()
                return None
                
    except Exception as e:
        logger.error(f"❌ insert_signal validation failed: {e}")
        return None

def get_recent_signals(symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get recent signals from database with comprehensive error handling.
    Args:
        symbol: Filter by symbol (optional)
        limit: Maximum number of records
    Returns: List of signal dicts
    """
    with get_connection() as conn:
        if not conn:
            return []
            
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            if symbol:
                cur.execute(
                    """
                    SELECT * FROM signals
                    WHERE symbol = %s
                    ORDER BY ts_utc DESC
                    LIMIT %s;
                    """,
                    (symbol[:50] if symbol else "", limit)
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM signals
                    ORDER BY ts_utc DESC
                    LIMIT %s;
                    """,
                    (limit,)
                )
                
            results = []
            for row in cur.fetchall():
                row_dict = dict(row)
                # Convert Decimal to float
                for key in row_dict:
                    if hasattr(row_dict[key], 'is_finite') and callable(getattr(row_dict[key], 'is_finite', None)):
                        row_dict[key] = float(row_dict[key])
                results.append(row_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ get_recent_signals failed: {e}")
            return []

# ============================================================
# TRADE EXECUTIONS
# ============================================================
def record_execution(
    signal_id: Optional[int],
    symbol: str,
    direction: str,
    entry: float,
    size: float,
    sl: Optional[float] = None,
    tp: Optional[float] = None,
    meta: Optional[Dict] = None,
    ticket: Optional[str] = None
) -> Optional[int]:
    """
    Record a new trade execution (open trade) with comprehensive validation.
    Returns: Trade ID if successful, None otherwise
    """
    try:
        # Validate inputs
        if not symbol or not isinstance(symbol, str):
            logger.error("❌ Invalid symbol for record_execution")
            return None
        
        if not direction or not isinstance(direction, str):
            logger.error("❌ Invalid direction for record_execution")
            return None
        
        if not isinstance(entry, (int, float)) or entry <= 0:
            logger.error("❌ Invalid entry price for record_execution")
            return None
        
        if not isinstance(size, (int, float)) or size <= 0:
            logger.error("❌ Invalid size for record_execution")
            return None
        
        with get_connection() as conn:
            if not conn:
                return None

            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO trade_executions
                    (signal_id, symbol, direction, entry_price, size, sl, tp, status, meta, ticket)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'opened', %s, %s)
                    RETURNING id;
                    """,
                    (
                        int(signal_id) if signal_id is not None else None,
                        symbol[:20],
                        direction[:10],
                        float(entry),
                        float(size),
                        float(sl) if sl is not None else None,
                        float(tp) if tp is not None else None,
                        json.dumps(meta or {}),
                        str(ticket)[:100] if ticket else None
                    )
                )
                trade_id = cur.fetchone()[0]
                conn.commit()
                logger.debug(f"✅ Trade recorded: ID={trade_id}, {symbol} {direction}")
                return trade_id

            except Exception as e:
                logger.error(f"❌ record_execution failed: {e}")
                if conn:
                    conn.rollback()
                return None
                
    except Exception as e:
        logger.error(f"❌ record_execution validation failed: {e}")
        return None

def close_trade(
    trade_id: int,
    exit_price: float,
    pnl: float,
    comment: Optional[str] = None,
    features: Optional[Dict] = None
) -> bool:
    """
    Close a trade and record the result with comprehensive error handling.
    """
    try:
        # Validate inputs
        if not isinstance(trade_id, int) or trade_id <= 0:
            logger.error("❌ Invalid trade_id for close_trade")
            return False
        
        if not isinstance(exit_price, (int, float)) or exit_price <= 0:
            logger.error("❌ Invalid exit_price for close_trade")
            return False
        
        if not isinstance(pnl, (int, float)):
            logger.error("❌ Invalid pnl for close_trade")
            return False
        
        with get_connection() as conn:
            if not conn:
                return False
                
            try:
                cur = conn.cursor()
                
                cur.execute(
                    """
                    SELECT symbol, direction, entry_price, sl, tp, size
                    FROM trade_executions
                    WHERE id = %s;
                    """,
                    (trade_id,)
                )
                trade_data = cur.fetchone()
                
                if not trade_data:
                    logger.warning(f"⚠️ Trade {trade_id} not found")
                    return False
                
                symbol, direction, entry_price, sl, tp, size = trade_data
                
                cur.execute(
                    """
                    UPDATE trade_executions
                    SET 
                        exit_price = %s,
                        pnl = %s,
                        status = 'closed',
                        closed_at = NOW(),
                        updated_at = NOW(),
                        comment = COALESCE(%s, comment)
                    WHERE id = %s AND status = 'opened';
                    """,
                    (
                        float(exit_price),
                        float(pnl),
                        str(comment)[:500] if comment else None,
                        trade_id
                    )
                )
                
                affected = cur.rowcount
                conn.commit()
                
                if affected > 0:
                    logger.debug(f"✅ Trade closed: ID={trade_id}, PnL={pnl:.4f}")
                    
                    # Import ML storage locally to avoid circular import
                    try:
                        from ml_system.data_storage import get_ml_storage
                        ml_storage = get_ml_storage()
                        
                        ml_storage.save_trade({
                            'symbol': str(symbol),
                            'side': str(direction),
                            'entry_price': float(entry_price) if entry_price is not None else 0.0,
                            'exit_price': float(exit_price),
                            'sl_used': float(sl) if sl is not None else None,
                            'tp_used': float(tp) if tp is not None else None,
                            'pnl': float(pnl),
                            'outcome': 'profit' if pnl > 0 else 'loss',
                            'price_move_pct': ((exit_price - entry_price) / entry_price) * 100 if entry_price and entry_price > 0 else 0.0,
                            'features': features or {},
                            'db_trade_id': int(trade_id)
                        })
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to save trade to ML storage: {e}")
                    
                    return True
                else:
                    logger.warning(f"⚠️ Trade {trade_id} not found or already closed")
                    return False

            except Exception as e:
                logger.error(f"❌ close_trade failed: {e}")
                if conn:
                    conn.rollback()
                return False
                
    except Exception as e:
        logger.error(f"❌ close_trade validation failed: {e}")
        return False

def get_trade_by_id(trade_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a specific trade by ID with comprehensive error handling.
    Returns: Trade dict or None
    """
    try:
        if not isinstance(trade_id, int) or trade_id <= 0:
            logger.error("❌ Invalid trade_id for get_trade_by_id")
            return None
        
        with get_connection() as conn:
            if not conn:
                return None
                
            try:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute(
                    "SELECT * FROM trade_executions WHERE id = %s;",
                    (trade_id,)
                )
                row = cur.fetchone()
                if row:
                    row_dict = dict(row)
                    # Convert Decimal to float
                    for key in row_dict:
                        if hasattr(row_dict[key], 'is_finite') and callable(getattr(row_dict[key], 'is_finite', None)):
                            row_dict[key] = float(row_dict[key])
                    return row_dict
                return None
                
            except Exception as e:
                logger.error(f"❌ get_trade_by_id failed: {e}")
                return None
                
    except Exception as e:
        logger.error(f"❌ get_trade_by_id validation failed: {e}")
        return None

def get_open_trades(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get all open trades with comprehensive error handling.
    Args:
        symbol: Filter by symbol (optional)
    Returns: List of open trade dicts
    """
    with get_connection() as conn:
        if not conn:
            return []
            
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            if symbol:
                cur.execute(
                    """
                    SELECT * FROM trade_executions
                    WHERE status = 'opened' AND symbol = %s
                    ORDER BY created_at DESC;
                    """,
                    (symbol[:20] if symbol else "",)
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM trade_executions
                    WHERE status = 'opened'
                    ORDER BY created_at DESC;
                    """
                )
                
            results = []
            for row in cur.fetchall():
                row_dict = dict(row)
                # Convert Decimal to float
                for key in row_dict:
                    if hasattr(row_dict[key], 'is_finite') and callable(getattr(row_dict[key], 'is_finite', None)):
                        row_dict[key] = float(row_dict[key])
                results.append(row_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ get_open_trades failed: {e}")
            return []

def get_closed_trades(
    symbol: Optional[str] = None,
    days: int = 7,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get closed trades within a time period with comprehensive error handling.
    
    Args:
        symbol: Filter by symbol (optional)
        days: Number of days to look back
        limit: Maximum records
        
    Returns:
        List of closed trade dicts
    """
    with get_connection() as conn:
        if not conn:
            return []
            
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            if symbol:
                cur.execute(
                    """
                    SELECT * FROM trade_executions
                    WHERE status = 'closed' 
                      AND symbol = %s
                      AND closed_at > NOW() - INTERVAL '%s days'
                    ORDER BY closed_at DESC
                    LIMIT %s;
                    """,
                    (symbol[:20] if symbol else "", days, limit)
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM trade_executions
                    WHERE status = 'closed'
                      AND closed_at > NOW() - INTERVAL '%s days'
                    ORDER BY closed_at DESC
                    LIMIT %s;
                    """,
                    (days, limit)
                )
                
            results = []
            for row in cur.fetchall():
                row_dict = dict(row)
                # Convert Decimal to float
                for key in row_dict:
                    if hasattr(row_dict[key], 'is_finite') and callable(getattr(row_dict[key], 'is_finite', None)):
                        row_dict[key] = float(row_dict[key])
                results.append(row_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ get_closed_trades failed: {e}")
            return []

def update_trade_meta(trade_id: int, meta: Dict[str, Any]) -> bool:
    """
    Update the meta field of a trade (merge with existing) with comprehensive error handling.
    
    Args:
        trade_id: Trade ID
        meta: Dict to merge into existing meta
        
    Returns:
        True if successful
    """
    try:
        if not isinstance(trade_id, int) or trade_id <= 0:
            logger.error("❌ Invalid trade_id for update_trade_meta")
            return False
        
        if not isinstance(meta, dict):
            logger.error("❌ Invalid meta for update_trade_meta")
            return False
        
        with get_connection() as conn:
            if not conn:
                return False
                
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE trade_executions
                    SET 
                        meta = COALESCE(meta, '{}'::jsonb) || %s::jsonb,
                        updated_at = NOW()
                    WHERE id = %s;
                    """,
                    (json.dumps(meta), trade_id)
                )
                conn.commit()
                return cur.rowcount > 0
                
            except Exception as e:
                logger.error(f"❌ update_trade_meta failed: {e}")
                if conn:
                    conn.rollback()
                return False
                
    except Exception as e:
        logger.error(f"❌ update_trade_meta validation failed: {e}")
        return False

def update_trade_sl_tp(
    trade_id: int,
    sl: Optional[float] = None,
    tp: Optional[float] = None
) -> bool:
    """
    Update SL/TP for a trade with comprehensive error handling.
    """
    try:
        if not isinstance(trade_id, int) or trade_id <= 0:
            logger.error("❌ Invalid trade_id for update_trade_sl_tp")
            return False
        
        with get_connection() as conn:
            if not conn:
                return False
                
            try:
                cur = conn.cursor()
                
                updates = []
                params = []
                
                if sl is not None:
                    updates.append("sl = %s")
                    params.append(float(sl))
                if tp is not None:
                    updates.append("tp = %s")
                    params.append(float(tp))
                    
                if not updates:
                    return True
                    
                updates.append("updated_at = NOW()")
                params.append(trade_id)
                
                cur.execute(
                    f"""
                    UPDATE trade_executions
                    SET {", ".join(updates)}
                    WHERE id = %s;
                    """,
                    tuple(params)
                )
                conn.commit()
                return cur.rowcount > 0
                
            except Exception as e:
                logger.error(f"❌ update_trade_sl_tp failed: {e}")
                if conn:
                    conn.rollback()
                return False
                
    except Exception as e:
        logger.error(f"❌ update_trade_sl_tp validation failed: {e}")
        return False

def get_trade_statistics(days: int = 30) -> Dict[str, Any]:
    """
    Get trading statistics for a period with comprehensive error handling.
    
    Args:
        days: Number of days to analyze
        
    Returns:
        Dict with statistics
    """
    with get_connection() as conn:
        if not conn:
            return {}
            
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """
                SELECT 
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
                    COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_trades,
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    COALESCE(AVG(pnl), 0) as avg_pnl,
                    COALESCE(AVG(CASE WHEN pnl > 0 THEN pnl END), 0) as avg_win,
                    COALESCE(AVG(CASE WHEN pnl < 0 THEN pnl END), 0) as avg_loss,
                    COALESCE(MAX(pnl), 0) as max_win,
                    COALESCE(MIN(pnl), 0) as max_loss
                FROM trade_executions
                WHERE status = 'closed'
                  AND closed_at > NOW() - INTERVAL '%s days';
                """,
                (days,)
            )
            row = cur.fetchone()
            
            if not row:
                return {}
                
            stats = dict(row)
            
            # Calculate win rate
            total = stats.get("total_trades", 0)
            wins = stats.get("winning_trades", 0)
            stats["win_rate"] = (wins / total * 100) if total > 0 else 0
            
            # Calculate profit factor
            avg_win = abs(float(stats.get("avg_win", 0) or 0))
            avg_loss = abs(float(stats.get("avg_loss", 0) or 0))
            stats["profit_factor"] = (avg_win / avg_loss) if avg_loss > 0 else 0
            
            # Convert Decimal to float
            for key in stats:
                if hasattr(stats[key], 'is_finite') and callable(getattr(stats[key], 'is_finite', None)):
                    stats[key] = float(stats[key])
                    
            return stats
            
        except Exception as e:
            logger.error(f"❌ get_trade_statistics failed: {e}")
            return {}

# ============================================================
# ACCOUNT METRICS
# ============================================================
def record_metrics(
    balance: float,
    equity: float,
    profit: float,
    margin: float = 0.0,
    free_margin: float = 0.0,
    margin_level: float = 0.0
) -> bool:
    """
    Record account metrics snapshot with comprehensive error handling.
    """
    # Enhanced logging for debugging
    logger.debug(
        f"record_metrics called: "
        f"balance={balance}, equity={equity}, profit={profit}, "
        f"margin={margin}, free_margin={free_margin}, margin_level={margin_level}"
    )
    
    try:
        # Validate inputs
        if not isinstance(balance, (int, float)):
            logger.error("❌ Invalid balance for record_metrics")
            return False
        
        if not isinstance(equity, (int, float)):
            logger.error("❌ Invalid equity for record_metrics")
            return False
        
        if not isinstance(profit, (int, float)):
            logger.error("❌ Invalid profit for record_metrics")
            return False
        
        with get_connection() as conn:
            if not conn:
                logger.error("❌ record_metrics: no DB connection")
                return False

            try:
                with conn.cursor() as cur:
                    # ✅ Add RETURNING clause to verify insertion
                    cur.execute(
                        """
                        INSERT INTO account_metrics
                        (balance, equity, profit, margin, free_margin, margin_level)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id, timestamp;
                        """,
                        (
                            float(balance),
                            float(equity),
                            float(profit),
                            float(margin),
                            float(free_margin),
                            float(margin_level)
                        )
                    )
                    
                    # Get the inserted row info
                    row = cur.fetchone()
                    if row:
                        row_id, timestamp = row
                        logger.debug(f"Inserted metrics: ID={row_id}, timestamp={timestamp}")
                    
                    conn.commit()
                    logger.success(
                        f"Metrics recorded: Balance=${balance:.2f}, Equity=${equity:.2f}, "
                        f"Profit=${profit:.2f}"
                    )
                    return True

            except Exception as e:
                logger.error(f"❌ record_metrics failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                if conn:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                return False
                
    except Exception as e:
        logger.error(f"❌ record_metrics validation failed: {e}")
        return False

def safe_record_metrics(
    balance: float,
    equity: float,
    profit: float,
    margin: float = 0.0,
    free_margin: float = 0.0,
    margin_level: float = 0.0
) -> bool:
    """Saves account metrics to DB with robust error handling."""
    try:
        from db_utils import get_connection
        with get_connection() as conn:
            if not conn:
                logger.error("❌ DB Connection failed in safe_record_metrics")
                return False
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO account_metrics 
                    (balance, equity, profit, margin, free_margin, margin_level, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                """, (
                    float(balance) if balance is not None else 0.0,
                    float(equity) if equity is not None else 0.0,
                    float(profit) if profit is not None else 0.0,
                    float(margin) if margin is not None else 0.0,
                    float(free_margin) if free_margin is not None else 0.0,
                    float(margin_level) if margin_level is not None else 0.0
                ))
                conn.commit()
        return True
    except Exception as e:
        logger.error(f"❌ safe_record_metrics DB error: {e}")
        return False
    
def latest_metrics() -> Dict[str, Any]:
    """
    Get the most recent account metrics with comprehensive error handling.
    Returns:
        Dict with metric values or empty dict
    """
    with get_connection() as conn:
        if not conn:
            return {}
            
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """
                SELECT * FROM account_metrics
                ORDER BY timestamp DESC
                LIMIT 1;
                """
            )
            row = cur.fetchone()
            
            if not row:
                return {}
                
            result = dict(row)
            # Convert Decimal to float
            for key in result:
                if hasattr(result[key], 'is_finite') and callable(getattr(result[key], 'is_finite', None)):
                    result[key] = float(result[key])
            return result
            
        except Exception as e:
            logger.error(f"❌ latest_metrics failed: {e}")
            return {}

def get_metrics_history(hours: int = 24) -> List[Dict[str, Any]]:
    """
    Get metrics history for a time period with comprehensive error handling.
    
    Args:
        hours: Number of hours to look back
        
    Returns:
        List of metric dicts
    """
    with get_connection() as conn:
        if not conn:
            return []
            
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """
                SELECT * FROM account_metrics
                WHERE timestamp > NOW() - INTERVAL '%s hours'
                ORDER BY timestamp DESC;
                """,
                (hours,)
            )
            
            results = []
            for row in cur.fetchall():
                d = dict(row)
                for key in d:
                    if hasattr(d[key], 'is_finite') and callable(getattr(d[key], 'is_finite', None)):
                        d[key] = float(d[key])
                results.append(d)
            return results
            
        except Exception as e:
            logger.error(f"❌ get_metrics_history failed: {e}")
            return []

def cleanup_old_metrics(days: int = 30) -> int:
    """
    Remove old metrics records with comprehensive error handling.
    
    Args:
        days: Number of days to keep
        
    Returns:
        Number of deleted records
    """
    with get_connection() as conn:
        if not conn:
            return 0
            
        try:
            cur = conn.cursor()
            cur.execute(
                """
                DELETE FROM account_metrics
                WHERE timestamp < NOW() - INTERVAL '%s days'
                RETURNING id;
                """,
                (days,)
            )
            deleted = cur.rowcount
            conn.commit()
            
            if deleted > 0:
                logger.info(f"🧹 Cleaned up {deleted} old metric records")
            return deleted
            
        except Exception as e:
            logger.error(f"❌ cleanup_old_metrics failed: {e}")
            if conn:
                conn.rollback()
            return 0

# ============================================================
# DAILY SUMMARY
# ============================================================
def update_daily_summary(
    date: Optional[datetime] = None,
    starting_balance: Optional[float] = None,
    ending_balance: Optional[float] = None,
    total_pnl: Optional[float] = None,
    total_trades: Optional[int] = None,
    winning_trades: Optional[int] = None,
    losing_trades: Optional[int] = None,
    max_drawdown: Optional[float] = None
) -> bool:
    """
    Update or insert daily summary record with comprehensive error handling.
    
    Returns:
        True if successful
    """
    try:
        if date is None:
            date = datetime.now(timezone.utc).date()
        elif isinstance(date, datetime):
            date = date.date()
            
        with get_connection() as conn:
            if not conn:
                return False
                
            try:
                cur = conn.cursor()
                
                # Calculate win rate if we have the data
                win_rate = None
                if total_trades and total_trades > 0 and winning_trades is not None:
                    win_rate = (winning_trades / total_trades) * 100
                
                cur.execute(
                    """
                    INSERT INTO daily_summary 
                    (date, starting_balance, ending_balance, total_pnl, 
                     total_trades, winning_trades, losing_trades, win_rate, max_drawdown)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date) DO UPDATE SET
                        ending_balance = COALESCE(EXCLUDED.ending_balance, daily_summary.ending_balance),
                        total_pnl = COALESCE(EXCLUDED.total_pnl, daily_summary.total_pnl),
                        total_trades = COALESCE(EXCLUDED.total_trades, daily_summary.total_trades),
                        winning_trades = COALESCE(EXCLUDED.winning_trades, daily_summary.winning_trades),
                        losing_trades = COALESCE(EXCLUDED.losing_trades, daily_summary.losing_trades),
                        win_rate = COALESCE(EXCLUDED.win_rate, daily_summary.win_rate),
                        max_drawdown = COALESCE(EXCLUDED.max_drawdown, daily_summary.max_drawdown),
                        updated_at = NOW();
                    """,
                    (
                        date,
                        float(starting_balance) if starting_balance is not None else None,
                        float(ending_balance) if ending_balance is not None else None,
                        float(total_pnl) if total_pnl is not None else None,
                        int(total_trades) if total_trades is not None else None,
                        int(winning_trades) if winning_trades is not None else None,
                        int(losing_trades) if losing_trades is not None else None,
                        float(win_rate) if win_rate is not None else None,
                        float(max_drawdown) if max_drawdown is not None else None
                    )
                )
                conn.commit()
                return True
                
            except Exception as e:
                logger.error(f"❌ update_daily_summary failed: {e}")
                if conn:
                    conn.rollback()
                return False
                
    except Exception as e:
        logger.error(f"❌ update_daily_summary validation failed: {e}")
        return False

def get_daily_summaries(days: int = 30) -> List[Dict[str, Any]]:
    """
    Get daily summaries for a period with comprehensive error handling.
    
    Returns:
        List of daily summary dicts
    """
    with get_connection() as conn:
        if not conn:
            return []
            
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """
                SELECT * FROM daily_summary
                WHERE date > CURRENT_DATE - INTERVAL '%s days'
                ORDER BY date DESC;
                """,
                (days,)
            )
            
            results = []
            for row in cur.fetchall():
                d = dict(row)
                for key in d:
                    if hasattr(d[key], 'is_finite') and callable(getattr(d[key], 'is_finite', None)):
                        d[key] = float(d[key])
                results.append(d)
            return results
            
        except Exception as e:
            logger.error(f"❌ get_daily_summaries failed: {e}")
            return []

# ============================================================
# CLEANUP / MAINTENANCE
# ============================================================
def cleanup_old_data(
    heartbeat_days: int = 7,
    metrics_days: int = 30,
    signals_days: int = 30
) -> Dict[str, int]:
    """
    Clean up old data from all tables with comprehensive error handling.
    
    Args:
        heartbeat_days: Days to keep heartbeats
        metrics_days: Days to keep metrics
        signals_days: Days to keep signals
        
    Returns:
        Dict with counts of deleted records per table
    """
    deleted = {
        "heartbeats": 0,
        "metrics": 0,
        "signals": 0,
    }
    
    deleted["heartbeats"] = cleanup_old_heartbeats(heartbeat_days)
    deleted["metrics"] = cleanup_old_metrics(metrics_days)
    
    # Cleanup old signals
    with get_connection() as conn:
        if conn:
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    DELETE FROM signals
                    WHERE ts_utc < NOW() - INTERVAL '%s days'
                    RETURNING id;
                    """,
                    (signals_days,)
                )
                deleted["signals"] = cur.rowcount
                conn.commit()
            except Exception as e:
                logger.error(f"❌ cleanup signals failed: {e}")
                if conn:
                    conn.rollback()
    
    total = sum(deleted.values())
    if total > 0:
        logger.info(f"🧹 Cleanup complete: {deleted}")
    
    return deleted