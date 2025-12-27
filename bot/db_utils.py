# db_utils.py
"""
Professional Database Utilities for Trading Bot
Production-grade PostgreSQL operations with advanced error handling, monitoring, and performance optimization.
Designed for 24/7 live trading environments.
"""
import os
import json
import time
import math
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Union
from decimal import Decimal
import psycopg2
import psycopg2.extras
from loguru import logger

# ============================================================
# DB CONFIG — ENHANCED WITH VALIDATION & LOGGING
# ============================================================
DB_HOST = os.getenv("DB_HOST", os.getenv("POSTGRES_HOST", "localhost"))
DB_PORT = int(os.getenv("DB_PORT", os.getenv("POSTGRES_PORT", "5432")))
DB_NAME = os.getenv("DB_NAME", os.getenv("POSTGRES_DB", "trading_db"))
DB_USER = os.getenv("DB_USER", os.getenv("POSTGRES_USER", "trader"))
DB_PASS = os.getenv("DB_PASS", os.getenv("POSTGRES_PASSWORD", "trader_pass"))

# Connection settings
MAX_RETRIES = 3
RETRY_DELAY = 1.0
CONN_TIMEOUT = 10
KEEPALIVE_IDLE = 30
KEEPALIVE_INTERVAL = 10
KEEPALIVE_COUNT = 5

logger.info(f"🗄️ Database Configuration → {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# ============================================================
# CONNECTION MANAGER — PRODUCTION GRADE
# ============================================================
@contextmanager
def get_connection(retries: int = MAX_RETRIES):
    """
    Production-grade database connection with retry logic, keepalive, and comprehensive error handling.
    """
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
                connect_timeout=CONN_TIMEOUT,
                keepalives=1,
                keepalives_idle=KEEPALIVE_IDLE,
                keepalives_interval=KEEPALIVE_INTERVAL,
                keepalives_count=KEEPALIVE_COUNT,
                application_name="trading_bot"
            )
            logger.success(f"✅ Database connection established (attempt {attempt + 1})")
            break
            
        except psycopg2.OperationalError as e:
            last_error = e
            if attempt < retries - 1:
                wait_time = RETRY_DELAY * (attempt + 1)
                logger.warning(f"⚠️ DB connection attempt {attempt + 1}/{retries} failed: {e} — retrying in {wait_time}s")
                time.sleep(wait_time)
            else:
                logger.critical(f"❌ DB connection FAILED after {retries} attempts: {e}")
                
        except Exception as e:
            last_error = e
            logger.error(f"❌ Unexpected DB connection error: {e}")
            break
    
    try:
        if conn is None:
            raise Exception("Failed to establish database connection after all retries")
        yield conn
    finally:
        if conn:
            try:
                conn.close()
                logger.debug("✅ Database connection closed gracefully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to close DB connection: {e}")

def check_connection() -> bool:
    """
    Verify database connectivity with minimal overhead.
    """
    try:
        with get_connection(retries=1) as conn:
            if not conn:
                return False
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                result = cur.fetchone()
                healthy = result is not None and result[0] == 1
                if healthy:
                    logger.debug("✅ Database health check passed")
                return healthy
    except Exception as e:
        logger.debug(f"❌ DB health check failed: {e}")
        return False

# ============================================================
# INIT TABLES — ROBUST SCHEMA MANAGEMENT
# ============================================================
def init_tables() -> bool:
    """
    Initialize or migrate database schema with zero-downtime compatibility.
    Automatically handles column additions and index creation.
    """
    with get_connection() as conn:
        if not conn:
            logger.error("❌ Cannot initialize tables: no database connection")
            return False
            
        try:
            cur = conn.cursor()

            # Signals Table
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
            """)

            # Indexes for Signals
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts_utc DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_confidence ON signals(confidence);")

            # Trade Executions Table
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
            """)

            # Indexes for Trade Executions
            cur.execute("CREATE INDEX IF NOT EXISTS idx_exec_status ON trade_executions(status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_exec_symbol ON trade_executions(symbol);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_exec_created ON trade_executions(created_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_exec_closed ON trade_executions(closed_at);")

            # Account Metrics Table
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

            # Migration: Add created_at if missing
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name='account_metrics' AND column_name='created_at'
                    ) THEN
                        ALTER TABLE account_metrics ADD COLUMN created_at TIMESTAMPTZ DEFAULT NOW();
                        RAISE NOTICE '✅ Added created_at column to account_metrics';
                    END IF;
                END$$;
            """)

            # Bot Heartbeat Table
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
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_heartbeat_timestamp ON bot_heartbeat(timestamp DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_heartbeat_status ON bot_heartbeat(status);")

            # Daily Summary Table
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
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_summary(date);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_daily_created ON daily_summary(created_at);")

            conn.commit()
            logger.success("✅ Database schema initialized/migrated successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize tables: {e}")
            if conn:
                conn.rollback()
            return False

# ============================================================
# UTILITY FUNCTIONS — TYPE CONVERSION & ERROR HANDLING
# ============================================================
def _convert_decimal_to_float(data: Any) -> Any:
    """
    Recursively convert Decimal values to float for JSON serialization and API compatibility.
    """
    if isinstance(data, dict):
        return {k: _convert_decimal_to_float(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_convert_decimal_to_float(item) for item in data]
    elif isinstance(data, Decimal):
        return float(data)
    return data

# ============================================================
# HEARTBEAT — ENHANCED MONITORING
# ============================================================
def write_heartbeat(status: str = "running",cycle_count: int = 0,step: str = "",equity: Optional[float] = None,balance: Optional[float] = None,
    open_positions: Optional[int] = None,message: str = "",ts_utc: Optional[str] = None,**kwargs) -> bool:
    """
    Write heartbeat record for system monitoring and alerting.
    All fields are sanitized and validated before insertion.
    """
    with get_connection() as conn:
        if not conn:
            logger.warning("⚠️ write_heartbeat: no database connection available")
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
                logger.trace("💓 Heartbeat recorded successfully")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to write heartbeat: {e}")
            if conn:
                conn.rollback()
            return False

def get_latest_heartbeat() -> Optional[Dict[str, Any]]:
    """
    Retrieve the most recent heartbeat record for system status monitoring.
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
                return _convert_decimal_to_float(dict(row))
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch latest heartbeat: {e}")
            return None

def cleanup_old_heartbeats(days: int = 7) -> int:
    """
    Remove outdated heartbeat records to maintain database performance.
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
                logger.info(f"🧹 Cleaned {deleted} heartbeat records older than {days} days")
            return deleted
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup heartbeats: {e}")
            if conn:
                conn.rollback()
            return 0

# ============================================================
# SIGNALS — ENHANCED INSERT & QUERY
# ============================================================
def insert_signal(symbol: str,direction: str,price: float,sl: float,tp: float,timeframe: str,confidence: float,session_tag: str,
raw: Optional[Dict] = None) -> Optional[int]:
    """
    Insert a new trading signal with full validation and error handling.
    Returns signal ID on success, None on failure.
    """
    try:
        if not symbol or not isinstance(symbol, str):
            logger.error("❌ Invalid symbol parameter")
            return None
        if not direction or not isinstance(direction, str):
            logger.error("❌ Invalid direction parameter")
            return None
        if not isinstance(price, (int, float)) or price <= 0:
            logger.error("❌ Invalid price parameter")
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
                logger.info(f"✅ Signal recorded: ID={signal_id}, {symbol} {direction} @ {price}")
                return signal_id

            except Exception as e:
                logger.error(f"❌ Failed to insert signal: {e}")
                if conn:
                    conn.rollback()
                return None
                
    except Exception as e:
        logger.error(f"❌ Signal validation failed: {e}")
        return None

def get_recent_signals(symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Fetch recent signals with optional symbol filtering.
    Returns list of signal dictionaries.
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
                    (symbol[:50], limit)
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
                
            results = [dict(row) for row in cur.fetchall()]
            return [_convert_decimal_to_float(r) for r in results]
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch recent signals: {e}")
            return []

# ============================================================
# TRADE EXECUTIONS — PRODUCTION-GRADE OPERATIONS
# ============================================================
def record_execution(signal_id: Optional[int],symbol: str,direction: str,entry: float,size: float,sl: Optional[float] = None,
    tp: Optional[float] = None,meta: Optional[Dict] = None,ticket: Optional[str] = None) -> Optional[int]:
    """
    Record a new trade execution (open position).
    Returns trade ID on success, None on failure.
    """
    try:
        # Validate inputs
        if not isinstance(symbol, str) or not symbol.strip():
            logger.error("❌ Invalid symbol: must be non-empty string")
            return None
        if not isinstance(direction, str) or not direction.strip():
            logger.error("❌ Invalid direction: must be non-empty string")
            return None
        if not isinstance(entry, (int, float)) or entry <= 0:
            logger.error(f"❌ Invalid entry price: {entry} (must be > 0)")
            return None
        if not isinstance(size, (int, float)) or size <= 0:
            logger.error(f"❌ Invalid position size: {size} (must be > 0)")
            return None
        if sl is not None and (not isinstance(sl, (int, float)) or sl <= 0):
            logger.error(f"❌ Invalid stop loss: {sl}")
            return None
        if tp is not None and (not isinstance(tp, (int, float)) or tp <= 0):
            logger.error(f"❌ Invalid take profit: {tp}")
            return None
        if signal_id is not None and not isinstance(signal_id, int):
            logger.error(f"❌ Invalid signal_id: {signal_id} (must be int or None)")
            return None

        # Sanitize strings
        symbol_clean = symbol.strip()[:50]  # Max 50 chars
        direction_clean = direction.strip().upper()[:20]  # Max 20 chars
        ticket_clean = str(ticket)[:100] if ticket else None  # Max 100 chars

        # Serialize meta to JSON safely
        meta_json = None
        if meta is not None:
            try:
                meta_json = json.dumps(meta, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"⚠️ Failed to serialize meta to JSON: {e}. Using empty dict.")
                meta_json = json.dumps({})

        with get_connection() as conn:
            if not conn:
                logger.error("❌ Database connection failed")
                return None

            try:
                cur = conn.cursor()

                cur.execute(
                    """
                    INSERT INTO trade_executions (
                        signal_id, symbol, direction, entry_price, size, 
                        sl, tp, status, meta, ticket, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, 
                        %s, %s, 'opened', %s, %s, NOW()
                    )
                    RETURNING id;
                    """,
                    (
                        signal_id,
                        symbol_clean,
                        direction_clean,
                        float(entry),
                        float(size),
                        float(sl) if sl is not None else None,
                        float(tp) if tp is not None else None,
                        meta_json,
                        ticket_clean
                    )
                )

                result = cur.fetchone()
                if not result:
                    raise ValueError("No ID returned from INSERT")

                trade_id = int(result[0])
                conn.commit()

                logger.info(
                    f"✅ Trade opened: ID={trade_id}, {symbol_clean} "
                    f"{direction_clean} x{size} @ {entry}"
                )
                return trade_id

            except Exception as e:
                logger.error(f"❌ Failed to record trade execution: {e}")
                if conn:
                    conn.rollback()
                return None

    except Exception as e:
        logger.error(f"❌ Trade execution validation failed: {e}")
        return None

def close_trade(trade_id: int,exit_price: float,pnl: float,comment: Optional[str] = None,features: Optional[Dict] = None) -> bool:
    """
    Close an existing trade and update its PnL and status.
    Optionally save features to ML training dataset.
    """
    try:
        if not isinstance(trade_id, int) or trade_id <= 0:
            logger.error("❌ Invalid trade_id")
            return False
        if not isinstance(exit_price, (int, float)) or exit_price <= 0:
            logger.error("❌ Invalid exit_price")
            return False
        if not isinstance(pnl, (int, float)):
            logger.error("❌ Invalid pnl")
            return False
        
        with get_connection() as conn:
            if not conn:
                return False
                
            try:
                cur = conn.cursor()
                
                # Fetch trade details for ML storage
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
                    logger.warning(f"⚠️ Trade {trade_id} not found for closing")
                    return False
                
                symbol, direction, entry_price, sl, tp, size = trade_data
                
                # Update trade record
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
                    logger.info(f"✅ Trade closed: ID={trade_id}, PnL={pnl:+.4f} USDT")
                    
                    # Save to ML training set if features provided
                    if features is not None:
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
                            logger.debug(f"🧠 Trade {trade_id} saved to ML training dataset")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to save trade to ML storage: {e}")
                    
                    return True
                else:
                    logger.warning(f"⚠️ Trade {trade_id} already closed or not found")
                    return False

            except Exception as e:
                logger.error(f"❌ Failed to close trade {trade_id}: {e}")
                if conn:
                    conn.rollback()
                return False
                
    except Exception as e:
        logger.error(f"❌ Close trade validation failed: {e}")
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

# ============================================================
# TRADE QUERIES — ENHANCED PERFORMANCE & LOGGING
# ============================================================

def get_open_trades(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve all currently open trades with optional symbol filtering.
    Returns empty list on failure or if no connection available.
    """
    start_time = time.time()
    
    with get_connection() as conn:
        if not conn:
            logger.warning("⚠️ get_open_trades: no database connection")
            return []
            
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            if symbol:
                logger.debug(f"🔍 Fetching open trades for symbol: {symbol}")
                cur.execute(
                    """
                    SELECT * FROM trade_executions
                    WHERE status = 'opened' AND symbol = %s
                    ORDER BY created_at DESC;
                    """,
                    (symbol[:20],)
                )
            else:
                logger.debug("🔍 Fetching all open trades")
                cur.execute(
                    """
                    SELECT * FROM trade_executions
                    WHERE status = 'opened'
                    ORDER BY created_at DESC;
                    """
                )
                
            rows = cur.fetchall()
            results = [_convert_decimal_to_float(dict(row)) for row in rows]
            
            duration = time.time() - start_time
            logger.info(f"✅ Retrieved {len(results)} open trades in {duration:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch open trades: {e}")
            return []

def get_closed_trades(symbol: Optional[str] = None,days: int = 7,limit: int = 100) -> List[Dict[str, Any]]:
    """
    Retrieve closed trades within specified time window.
    Supports optional symbol filtering and result limiting.
    """
    start_time = time.time()
    
    with get_connection() as conn:
        if not conn:
            logger.warning("⚠️ get_closed_trades: no database connection")
            return []
            
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            params = []
            base_query = """
                SELECT * FROM trade_executions
                WHERE status = 'closed'
                  AND closed_at > NOW() - INTERVAL '%s days'
                ORDER BY closed_at DESC
                LIMIT %s
            """
            
            if symbol:
                logger.debug(f"🔍 Fetching closed trades for {symbol} (last {days} days)")
                query = base_query.replace("ORDER BY", "AND symbol = %s ORDER BY")
                params = [symbol[:20], days, limit]
            else:
                logger.debug(f"🔍 Fetching all closed trades (last {days} days)")
                query = base_query
                params = [days, limit]
                
            cur.execute(query, params)
            
            rows = cur.fetchall()
            results = [_convert_decimal_to_float(dict(row)) for row in rows]
            
            duration = time.time() - start_time
            logger.info(f"✅ Retrieved {len(results)} closed trades in {duration:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch closed trades: {e}")
            return []

# ============================================================
# TRADE UPDATES — ATOMIC & SAFE
# ============================================================

def update_trade_meta(trade_id: int, meta: Dict[str, Any]) -> bool:
    """
    Atomically update trade metadata by merging with existing values.
    Uses JSONB operators for efficient partial updates.
    """
    start_time = time.time()
    
    try:
        if not isinstance(trade_id, int) or trade_id <= 0:
            logger.error(f"❌ Invalid trade_id: {trade_id}")
            return False
        
        if not isinstance(meta, dict):
            logger.error("❌ Meta must be a dictionary")
            return False
        
        with get_connection() as conn:
            if not conn:
                logger.warning("⚠️ update_trade_meta: no database connection")
                return False
                
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE trade_executions
                    SET 
                        meta = COALESCE(meta, '{}'::jsonb) || %s::jsonb,
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING id;
                    """,
                    (json.dumps(meta), trade_id)
                )
                
                updated = cur.fetchone()
                conn.commit()
                
                duration = time.time() - start_time
                if updated:
                    logger.info(f"✅ Updated meta for trade {trade_id} in {duration:.3f}s")
                    return True
                else:
                    logger.warning(f"⚠️ Trade {trade_id} not found for meta update")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Failed to update trade meta: {e}")
                if conn:
                    conn.rollback()
                return False
                
    except Exception as e:
        logger.error(f"❌ Validation failed in update_trade_meta: {e}")
        return False

def update_trade_sl_tp(trade_id: int,sl: Optional[float] = None,tp: Optional[float] = None) -> bool:
    """
    Update stop loss and/or take profit levels for a specific trade.
    Only updates provided values, leaves others unchanged.
    """
    start_time = time.time()
    
    try:
        if not isinstance(trade_id, int) or trade_id <= 0:
            logger.error(f"❌ Invalid trade_id: {trade_id}")
            return False
        
        # Early exit if no updates requested
        if sl is None and tp is None:
            logger.debug(f"ℹ️ No SL/TP updates requested for trade {trade_id}")
            return True
        
        with get_connection() as conn:
            if not conn:
                logger.warning("⚠️ update_trade_sl_tp: no database connection")
                return False
                
            try:
                cur = conn.cursor()
                
                set_clauses = []
                params = []
                
                if sl is not None:
                    set_clauses.append("sl = %s")
                    params.append(float(sl))
                    logger.debug(f"📉 Updating SL to {sl} for trade {trade_id}")
                    
                if tp is not None:
                    set_clauses.append("tp = %s")
                    params.append(float(tp))
                    logger.debug(f"📈 Updating TP to {tp} for trade {trade_id}")
                
                set_clauses.append("updated_at = NOW()")
                params.append(trade_id)
                
                query = f"""
                    UPDATE trade_executions
                    SET {", ".join(set_clauses)}
                    WHERE id = %s
                    RETURNING id;
                """
                
                cur.execute(query, tuple(params))
                updated = cur.fetchone()
                conn.commit()
                
                duration = time.time() - start_time
                if updated:
                    logger.info(f"✅ Updated SL/TP for trade {trade_id} in {duration:.3f}s")
                    return True
                else:
                    logger.warning(f"⚠️ Trade {trade_id} not found for SL/TP update")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Failed to update trade SL/TP: {e}")
                if conn:
                    conn.rollback()
                return False
                
    except Exception as e:
        logger.error(f"❌ Validation failed in update_trade_sl_tp: {e}")
        return False

# ============================================================
# STATISTICS — ENHANCED ANALYTICS
# ============================================================

def get_trade_statistics(days: int = 30) -> Dict[str, Any]:
    """
    Generate comprehensive trading statistics for performance analysis.
    Includes win rate, profit factor, average gains/losses, and more.
    """
    start_time = time.time()
    
    with get_connection() as conn:
        if not conn:
            logger.warning("⚠️ get_trade_statistics: no database connection")
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
                logger.info(f"ℹ️ No trades found in last {days} days")
                return {}
                
            stats = _convert_decimal_to_float(dict(row))
            
            # Calculate derived metrics
            total_trades = stats.get("total_trades", 0)
            winning_trades = stats.get("winning_trades", 0)
            stats["win_rate"] = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
            
            avg_win = abs(stats.get("avg_win", 0.0))
            avg_loss = abs(stats.get("avg_loss", 0.0))
            stats["profit_factor"] = (avg_win / avg_loss) if avg_loss > 0 else 0.0
            
            # Add additional metrics
            stats["risk_reward_ratio"] = (avg_win / avg_loss) if avg_loss > 0 else 0.0
            stats["expectancy"] = (stats["win_rate"] / 100 * avg_win) - ((100 - stats["win_rate"]) / 100 * avg_loss)
            
            duration = time.time() - start_time
            logger.info(f"📊 Generated trade statistics for {days} days in {duration:.3f}s")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to generate trade statistics: {e}")
            return {}

# ============================================================
# ACCOUNT METRICS — PRODUCTION GRADE
# ============================================================

def record_metrics(balance: float, equity: float, profit: float, margin: float = 0.0, free_margin: float = 0.0, margin_level: float = 0.0) -> bool:
    """
    Record current account metrics for performance tracking and analysis.
    All values are validated and sanitized before insertion.
    """
    start_time = time.time()
    
    # Input sanitization
    try:
        def sanitize_number(val, name: str) -> Optional[float]:
            if val is None:
                logger.warning(f"⚠️ {name} is None — defaulting to 0.0")
                return 0.0
            if not isinstance(val, (int, float)):
                logger.error(f"❌ {name} is not a number: {type(val)} = {val}")
                return None
            if math.isnan(val) or math.isinf(val):
                logger.warning(f"⚠️ {name} is {val} — defaulting to 0.0")
                return 0.0
            return float(val)

        balance_clean = sanitize_number(balance, "Balance")
        equity_clean = sanitize_number(equity, "Equity")
        profit_clean = sanitize_number(profit, "Profit")
        margin_clean = sanitize_number(margin, "Margin")
        free_margin_clean = sanitize_number(free_margin, "Free Margin")
        margin_level_clean = sanitize_number(margin_level, "Margin Level")

        if None in (balance_clean, equity_clean, profit_clean):
            logger.error("❌ One or more required metrics are invalid")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to sanitize metrics: {e}")
        return False

    logger.debug(
        f"📊 Recording metrics → Balance=${balance_clean:.2f}, Equity=${equity_clean:.2f}, "
        f"Profit=${profit_clean:.2f}, Margin=${margin_clean:.2f}, Free=${free_margin_clean:.2f}, Level={margin_level_clean:.2f}%"
    )
    
    with get_connection() as conn:
        if not conn:
            logger.error("❌ record_metrics: no database connection")
            return False

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO account_metrics
                    (balance, equity, profit, margin, free_margin, margin_level, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    RETURNING id;
                    """,
                    (
                        balance_clean,
                        equity_clean,
                        profit_clean,
                        margin_clean,
                        free_margin_clean,
                        margin_level_clean
                    )
                )
                
                metric_id = cur.fetchone()[0]
                conn.commit()
                
                duration = time.time() - start_time
                logger.success(f"✅ Metrics recorded (ID={metric_id}) in {duration:.3f}s")
                return True

        except Exception as e:
            logger.error(f"❌ Failed to record metrics: {e}")
            if conn:
                conn.rollback()
            return False

def safe_record_metrics(balance: float,equity: float,profit: float,margin: float = 0.0,free_margin: float = 0.0,margin_level: float = 0.0) -> bool:
    """
    Safely record account metrics with comprehensive error handling.
    Will not raise exceptions even on complete database failure.
    """
    try:
        return record_metrics(balance, equity, profit, margin, free_margin, margin_level)
    except Exception as e:
        logger.error(f"❌ safe_record_metrics failed: {e}")
        return False

def latest_metrics() -> Dict[str, Any]:
    """
    Retrieve the most recent account metrics snapshot.
    Returns empty dict if no metrics found or on error.
    """
    start_time = time.time()
    
    with get_connection() as conn:
        if not conn:
            logger.warning("⚠️ latest_metrics: no database connection")
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
                logger.info("ℹ️ No metrics found in database")
                return {}
                
            result = _convert_decimal_to_float(dict(row))
            
            duration = time.time() - start_time
            logger.debug(f"✅ Retrieved latest metrics in {duration:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch latest metrics: {e}")
            return {}

def get_metrics_history(hours: int = 24) -> List[Dict[str, Any]]:
    """
    Retrieve historical metrics for charting and analysis.
    Returns metrics from specified time window ordered chronologically.
    """
    start_time = time.time()
    
    with get_connection() as conn:
        if not conn:
            logger.warning("⚠️ get_metrics_history: no database connection")
            return []
            
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """
                SELECT * FROM account_metrics
                WHERE timestamp > NOW() - INTERVAL '%s hours'
                ORDER BY timestamp ASC;  -- Chronological order for charting
                """,
                (hours,)
            )
            
            rows = cur.fetchall()
            results = [_convert_decimal_to_float(dict(row)) for row in rows]
            
            duration = time.time() - start_time
            logger.info(f"✅ Retrieved {len(results)} metrics records from last {hours} hours in {duration:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch metrics history: {e}")
            return []

def cleanup_old_metrics(days: int = 30) -> int:
    """
    Remove old metrics records to maintain database performance.
    Returns number of records deleted.
    """
    start_time = time.time()
    
    with get_connection() as conn:
        if not conn:
            logger.warning("⚠️ cleanup_old_metrics: no database connection")
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
                duration = time.time() - start_time
                logger.info(f"🧹 Cleaned up {deleted} old metric records (> {days} days) in {duration:.3f}s")
            return deleted
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old metrics: {e}")
            if conn:
                conn.rollback()
            return 0

# ============================================================
# DAILY SUMMARY — BUSINESS INTELLIGENCE
# ============================================================

def update_daily_summary(date: Optional[datetime] = None,starting_balance: Optional[float] = None,ending_balance: Optional[float] = None,total_pnl: Optional[float] = None,
    total_trades: Optional[int] = None,winning_trades: Optional[int] = None,losing_trades: Optional[int] = None,max_drawdown: Optional[float] = None) -> bool:
    """
    Create or update daily performance summary for business intelligence.
    Automatically calculates win rate and handles conflicts gracefully.
    """
    start_time = time.time()
    
    try:
        if date is None:
            date = datetime.now(timezone.utc).date()
        elif isinstance(date, datetime):
            date = date.date()
            
        # Calculate win rate if possible
        win_rate = None
        if total_trades and total_trades > 0 and winning_trades is not None:
            win_rate = (winning_trades / total_trades) * 100
            logger.debug(f"📈 Calculated win rate: {win_rate:.2f}% ({winning_trades}/{total_trades})")
        
        with get_connection() as conn:
            if not conn:
                logger.warning("⚠️ update_daily_summary: no database connection")
                return False
                
            try:
                cur = conn.cursor()
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
                        updated_at = NOW()
                    RETURNING id;
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
                
                summary_id = cur.fetchone()[0]
                conn.commit()
                
                duration = time.time() - start_time
                logger.info(f"✅ Daily summary updated for {date} (ID={summary_id}) in {duration:.3f}s")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to update daily summary: {e}")
                if conn:
                    conn.rollback()
                return False
                
    except Exception as e:
        logger.error(f"❌ Validation failed in update_daily_summary: {e}")
        return False

def get_daily_summaries(days: int = 30) -> List[Dict[str, Any]]:
    """
    Retrieve daily summaries for performance analysis and reporting.
    Returns summaries from specified time window ordered by date.
    """
    start_time = time.time()
    
    with get_connection() as conn:
        if not conn:
            logger.warning("⚠️ get_daily_summaries: no database connection")
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
            
            rows = cur.fetchall()
            results = [_convert_decimal_to_float(dict(row)) for row in rows]
            
            duration = time.time() - start_time
            logger.info(f"✅ Retrieved {len(results)} daily summaries from last {days} days in {duration:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch daily summaries: {e}")
            return []

# ============================================================
# CLEANUP / MAINTENANCE — DATABASE HYGIENE
# ============================================================

def cleanup_old_data(
    heartbeat_days: int = 7,
    metrics_days: int = 30,
    signals_days: int = 30
) -> Dict[str, int]:
    """
    Comprehensive database cleanup to maintain optimal performance.
    Removes old records from all major tables based on retention policies.
    Returns summary of cleanup operations.
    """
    start_time = time.time()
    logger.info("🧹 Starting comprehensive database cleanup...")
    
    deleted = {
        "heartbeats": cleanup_old_heartbeats(heartbeat_days),
        "metrics": cleanup_old_metrics(metrics_days),
        "signals": 0
    }
    
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
                logger.debug(f"🧹 Cleaned {deleted['signals']} old signal records (> {signals_days} days)")
            except Exception as e:
                logger.error(f"❌ Failed to cleanup signals: {e}")
                if conn:
                    conn.rollback()
    
    total_deleted = sum(deleted.values())
    duration = time.time() - start_time
    
    if total_deleted > 0:
        logger.success(f"✅ Database cleanup completed: removed {total_deleted} records in {duration:.3f}s")
        logger.info(f"   Breakdown: {deleted}")
    else:
        logger.info(f"ℹ️ No records to clean up (completed in {duration:.3f}s)")
    
    return deleted