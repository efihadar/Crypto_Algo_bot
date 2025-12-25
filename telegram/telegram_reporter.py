# telegram/telegram_reporter.py
"""
📱 Telegram Reporter - Bot Status & Trading Reports
Features:
- Hourly/Daily/Weekly trading reports
- Account status monitoring
- Bot health checks via heartbeat
- ML performance metrics
- Equity charts generation
"""
import sys
import os
import time
import asyncio
import threading
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
from typing import Dict, Any, Optional, List
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psycopg2
import psycopg2.extras
from loguru import logger

try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ schedule library not installed")
    SCHEDULE_AVAILABLE = False
    schedule = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ─────────────────────────────────────────────────────
# Timezone – Israel (Asia/Jerusalem) with fallback
# ─────────────────────────────────────────────────────
try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Jerusalem"))
except ImportError:
    try:
        import pytz
        LOCAL_TZ = pytz.timezone(os.getenv("APP_TIMEZONE", "Asia/Jerusalem"))
    except ImportError:
        LOCAL_TZ = timezone(timedelta(hours=2))
except Exception:
    LOCAL_TZ = timezone(timedelta(hours=2))

# ─────────────────────────────────────────────────────
# Logging (only configure if running as main)
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, colorize=True, level="INFO")
    
    log_path = os.getenv("TELEGRAM_LOG_PATH", "/app/logs/telegram_reporter.log")
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logger.add(log_path, rotation="1 day", retention="7 days", level="DEBUG")
    except Exception as e:
        logger.warning(f"⚠️ Could not init file logger: {e}")

# ─────────────────────────────────────────────────────
# Telegram library
# ─────────────────────────────────────────────────────
try:
    import telegram
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ python-telegram-bot not installed")
    TELEGRAM_AVAILABLE = False
    telegram = None
    TelegramError = Exception

# ─────────────────────────────────────────────────────
# DB config
# ─────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", os.getenv("POSTGRES_HOST", "localhost"))
DB_PORT = int(os.getenv("DB_PORT", os.getenv("POSTGRES_PORT", "5432")))
DB_NAME = os.getenv("DB_NAME", os.getenv("POSTGRES_DB", "trading_db"))
DB_USER = os.getenv("DB_USER", os.getenv("POSTGRES_USER", "trader"))
DB_PASS = os.getenv("DB_PASS", os.getenv("POSTGRES_PASSWORD", "trader_pass"))

# ─────────────────────────────────────────────────────
# ML DB config
# ─────────────────────────────────────────────────────
ML_DB_HOST = os.getenv("ML_DB_HOST", "ml_postgres")
ML_DB_PORT = int(os.getenv("ML_DB_PORT", "5432"))
ML_DB_NAME = os.getenv("ML_DB_NAME", "ml_db")
ML_DB_USER = os.getenv("ML_DB_USER", "ml_user")
ML_DB_PASS = os.getenv("ML_DB_PASS", "ml_pass")
REPORT_ML_METRICS = os.getenv("REPORT_ML_METRICS", "true").lower() in ("true", "1", "yes")

# ─────────────────────────────────────────────────────
# Database Connection Managers
# ─────────────────────────────────────────────────────
@contextmanager
def get_db_connection():
    """Get main trading database connection."""
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            connect_timeout=10,
        )
        yield conn
    except psycopg2.OperationalError as e:
        logger.error(f"❌ DB connection failed: {e}")
        yield None
    except Exception as e:
        logger.error(f"❌ DB error: {e}")
        yield None
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

@contextmanager
def get_ml_db_connection():
    """Get ML database connection."""
    conn = None
    try:
        conn = psycopg2.connect(
            host=ML_DB_HOST,
            port=ML_DB_PORT,
            dbname=ML_DB_NAME,
            user=ML_DB_USER,
            password=ML_DB_PASS,
            connect_timeout=10,
        )
        yield conn
    except psycopg2.OperationalError as e:
        logger.debug(f"ML DB connection failed: {e}")
        yield None
    except Exception as e:
        logger.debug(f"ML DB error: {e}")
        yield None
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

def table_exists(table: str) -> bool:
    """Check if a table exists in the database."""
    with get_db_connection() as conn:
        if not conn:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute(
                    '''
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema='public' AND table_name=%s
                    ''',
                    (table,),
                )
                return cur.fetchone() is not None
        except Exception as e:
            logger.debug(f"table_exists({table}) → {e}")
            return False

def wait_for_tables(required: List[str], timeout_sec: int = 60) -> bool:
    """Wait for required tables to be created."""
    logger.info(f"🧱 Waiting for tables: {', '.join(required)} ...")
    deadline = time.time() + timeout_sec
    missing = set(required)

    while time.time() < deadline:
        ready = [t for t in list(missing) if table_exists(t)]
        for t in ready:
            missing.discard(t)
        if not missing:
            logger.success("✅ All required tables are present")
            return True
        time.sleep(1.5)

    if missing:
        logger.warning(f"⚠️ Some tables missing after timeout: {', '.join(sorted(missing))}")
    return False

def wait_for_db(max_retries: int = 60, delay: int = 1) -> bool:
    """Wait for database to be ready."""
    logger.info("⏳ Waiting for database to be ready...")
    for i in range(max_retries):
        with get_db_connection() as conn:
            if conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                    logger.success("✅ Database is ready!")
                    return True
                except Exception as e:
                    logger.debug(f"Attempt {i+1}/{max_retries}: DB not ready yet - {e}")
        time.sleep(delay)
    logger.error("❌ Database connection timeout")
    return False

# ─────────────────────────────────────────────────────
# Telegram Reporter Class
# ─────────────────────────────────────────────────────
class TelegramReporter:
    """
    Telegram Reporter for trading bot status and reports.
    
    Features:
    - Send text messages and photos
    - Hourly/Daily/Weekly trading reports
    - Account status monitoring
    - Bot health checks
    - ML performance metrics
    """

    # Rate limiting
    MIN_MESSAGE_INTERVAL = 1.0  # seconds between messages
    
    def __init__(self):
        """Initialize Telegram Reporter."""
        self.enabled: bool = False
        self.bot: Optional["telegram.Bot"] = None
        self.chat_id: Optional[str] = None
        self._lock = threading.Lock()
        self._last_message_time: float = 0
        self._last_health_alert_time: Optional[datetime] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

        if not TELEGRAM_AVAILABLE:
            logger.warning("⚠️ Telegram reporting disabled - library not installed")
            return

        self.enabled = os.getenv("ENABLE_TELEGRAM_ALERTS", "true").lower() in (
            "true", "1", "yes",
        )
        if not self.enabled:
            logger.info("ℹ️ Telegram alerts disabled (ENABLE_TELEGRAM_ALERTS=false)")
            return

        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            logger.warning("⚠️ Telegram credentials not configured")
            self.enabled = False
            return

        try:
            self.bot = telegram.Bot(token=token)
            self.chat_id = chat_id
            
            # Create event loop in a separate thread for async operations
            self._setup_event_loop()
            
            # Test connection
            info = self._run_async(self.bot.get_me())
            username = getattr(info, "username", None) or "Unknown"
            logger.success(f"✅ Telegram bot connected: @{username}")
            logger.info(f"📱 Chat ID: {chat_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Telegram bot: {e}")
            self.enabled = False
            self._cleanup_loop()

    def _setup_event_loop(self) -> None:
        """Set up dedicated event loop for async operations."""
        def run_loop():
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()
        
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()

    def _cleanup_loop(self) -> None:
        """Clean up event loop."""
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=2.0)
            try:
                self._loop.close()
            except Exception:
                pass

    def _run_async(self, coro):
        """Run async coroutine and return result."""
        if not self._loop or self._loop.is_closed():
            raise RuntimeError("Event loop not available")
        
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=30)

    def _rate_limit_wait(self) -> None:
        """Wait to respect rate limiting."""
        with self._lock:
            elapsed = time.time() - self._last_message_time
            if elapsed < self.MIN_MESSAGE_INTERVAL:
                time.sleep(self.MIN_MESSAGE_INTERVAL - elapsed)
            self._last_message_time = time.time()

    # ─────────────────────────────────────────────────────
    # Message Sending
    # ─────────────────────────────────────────────────────
    def send_message(
        self,
        message: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False
    ) -> bool:
        """
        Send a text message via Telegram.
        
        Args:
            message: Message text
            parse_mode: Parse mode (HTML, Markdown, etc.)
            disable_notification: If True, send silently
            
        Returns:
            True if sent successfully
        """
        if not self.enabled or not self.bot:
            return False
            
        try:
            self._rate_limit_wait()
            
            self._run_async(
                self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode=parse_mode,
                    disable_notification=disable_notification
                )
            )
            logger.debug("✅ Telegram message sent")
            return True
            
        except TelegramError as e:
            logger.error(f"❌ Telegram send failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error sending Telegram: {e}")
            return False

    def send_photo(
        self,
        photo_path: str,
        caption: Optional[str] = None,
        parse_mode: str = "HTML"
    ) -> bool:
        """
        Send a photo via Telegram.
        
        Args:
            photo_path: Path to the image file
            caption: Optional caption
            parse_mode: Parse mode for caption
            
        Returns:
            True if sent successfully
        """
        if not self.enabled or not self.bot:
            return False
            
        if not os.path.exists(photo_path):
            logger.error(f"❌ Photo not found: {photo_path}")
            return False
            
        try:
            self._rate_limit_wait()
            
            with open(photo_path, "rb") as f:
                self._run_async(
                    self.bot.send_photo(
                        chat_id=self.chat_id,
                        photo=f,
                        caption=caption,
                        parse_mode=parse_mode
                    )
                )
            logger.debug("✅ Telegram photo sent")
            return True
            
        except TelegramError as e:
            logger.error(f"❌ Telegram photo send failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error sending Telegram photo: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up resources."""
        self._cleanup_loop()
        logger.info("🧹 Telegram reporter cleaned up")

    # ─────────────────────────────────────────────────────
    # Database Queries - Trading
    # ─────────────────────────────────────────────────────
    def get_account_status(self) -> Dict[str, Any]:
        """Get current account status from database."""
        with get_db_connection() as conn:
            if not conn:
                logger.error("❌ get_account_status: No DB connection!")
                return {}
            try:
                with conn.cursor() as cur:
                    # First, check if table has any data
                    cur.execute("SELECT COUNT(*) FROM account_metrics")
                    count = cur.fetchone()[0]
                    logger.info(f"📊 account_metrics table has {count} rows")
                    
                    if count == 0:
                        logger.warning("⚠️ account_metrics table is EMPTY!")
                        
                        # Try to get data from bot_heartbeat as fallback
                        cur.execute('''
                            SELECT balance, equity 
                            FROM bot_heartbeat 
                            WHERE balance IS NOT NULL 
                            ORDER BY timestamp DESC 
                            LIMIT 1
                        ''')
                        hb_row = cur.fetchone()
                        if hb_row and hb_row[0]:
                            logger.info(f"📊 Using heartbeat fallback: balance={hb_row[0]}, equity={hb_row[1]}")
                            return {
                                "balance": float(hb_row[0] or 0),
                                "equity": float(hb_row[1] or 0),
                                "unrealised_pnl": float(hb_row[1] or 0) - float(hb_row[0] or 0),
                                "realised_pnl": 0.0,
                                "used_margin": 0.0,
                                "available": 0.0,
                                "last_update": None,
                                "source": "heartbeat_fallback"
                            }
                        return {}
                    
                    # Get latest metrics
                    cur.execute(
                        '''
                        SELECT balance, equity, profit, margin, free_margin, "timestamp"
                        FROM account_metrics
                        ORDER BY "timestamp" DESC
                        LIMIT 1
                        '''
                    )
                    row = cur.fetchone()
                    
                    if not row:
                        logger.warning("⚠️ No rows returned from account_metrics query")
                        return {}
                    
                    # Log what we got
                    logger.info(f"📊 Raw DB row: {row}")
                        
                    balance = float(row[0] or 0)
                    equity = float(row[1] or 0)
                    realised_pnl = float(row[2] or 0)
                    used_margin = float(row[3] or 0)
                    free_margin = float(row[4] or 0)
                    ts = row[5]
                    unrealised_pnl = equity - balance

                    result = {
                        "balance": balance,
                        "equity": equity,
                        "unrealised_pnl": unrealised_pnl,
                        "realised_pnl": realised_pnl,
                        "used_margin": used_margin,
                        "available": free_margin,
                        "last_update": ts,
                    }
                    
                    logger.success(f"✅ Account status: Balance=${balance:.2f}, Equity=${equity:.2f}")
                    return result
                    
            except Exception as e:
                logger.error(f"❌ Error getting account status: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return {}

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get open positions from database."""
        with get_db_connection() as conn:
            if not conn:
                return []
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        '''
                        SELECT symbol, direction, entry_price, size, created_at
                        FROM trade_executions
                        WHERE status = 'opened'
                        ORDER BY created_at DESC
                        '''
                    )
                    rows = cur.fetchall()
                    return [
                        {
                            "symbol": r[0],
                            "direction": r[1],
                            "entry_price": float(r[2] or 0),
                            "size": float(r[3] or 0),
                            "opened_at": r[4],
                        }
                        for r in rows
                    ]
            except Exception as e:
                logger.error(f"❌ Error getting positions: {e}")
                return []

    def get_today_trades(self) -> Dict[str, Any]:
        """Get today's trading statistics."""
        with get_db_connection() as conn:
            if not conn:
                return self._empty_trades_stats()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        '''
                        SELECT 
                            COUNT(*) AS total,
                            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) AS losses,
                            COALESCE(SUM(pnl), 0) AS total_pnl
                        FROM trade_executions
                        WHERE DATE(created_at) = CURRENT_DATE
                          AND status = 'closed'
                        '''
                    )
                    r = cur.fetchone()
                    if not r:
                        return self._empty_trades_stats()

                    total = r[0] or 0
                    wins = r[1] or 0
                    losses = r[2] or 0
                    total_pnl = float(r[3] or 0)
                    win_rate = (wins / total * 100.0) if total > 0 else 0.0

                    return {
                        "total": total,
                        "wins": wins,
                        "losses": losses,
                        "total_pnl": total_pnl,
                        "win_rate": win_rate,
                    }
            except Exception as e:
                logger.error(f"❌ Error getting today's trades: {e}")
                return self._empty_trades_stats()

    @staticmethod
    def _empty_trades_stats() -> Dict[str, Any]:
        """Return empty trades statistics."""
        return {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
        }

    def get_last_heartbeat(self) -> Dict[str, Any]:
        """Get the last bot heartbeat from database."""
        with get_db_connection() as conn:
            if not conn:
                return {}
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        '''
                        SELECT timestamp, status, cycle_count, message
                        FROM bot_heartbeat
                        ORDER BY timestamp DESC
                        LIMIT 1
                        '''
                    )
                    row = cur.fetchone()
                    if not row:
                        return {}

                    ts, status, cycle_count, message = row

                    # Calculate age
                    age_sec = None
                    try:
                        now_utc = datetime.now(timezone.utc)
                        if ts.tzinfo is None:
                            age_sec = (now_utc - ts.replace(tzinfo=timezone.utc)).total_seconds()
                        else:
                            age_sec = (now_utc - ts.astimezone(timezone.utc)).total_seconds()
                    except Exception:
                        pass

                    return {
                        "timestamp": ts,
                        "status": status,
                        "cycle_count": cycle_count,
                        "message": message,
                        "age_sec": age_sec,
                    }
            except Exception as e:
                logger.error(f"❌ Error getting last heartbeat: {e}")
                return {}

    def get_today_equity_series(self) -> List[Dict[str, Any]]:
        """Get today's equity time series."""
        with get_db_connection() as conn:
            if not conn:
                return []
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        '''
                        SELECT "timestamp", balance, equity
                        FROM account_metrics
                        WHERE DATE("timestamp") = CURRENT_DATE
                        ORDER BY "timestamp" ASC
                        '''
                    )
                    rows = cur.fetchall()
                    return [
                        {
                            "timestamp": r[0],
                            "balance": float(r[1] or 0),
                            "equity": float(r[2] or 0),
                        }
                        for r in rows
                    ]
            except Exception as e:
                logger.error(f"❌ Error getting today's equity series: {e}")
                return []

    def get_trades_last_days(self, days: int = 7) -> Dict[str, Any]:
        """Get trading statistics for the last N days."""
        with get_db_connection() as conn:
            if not conn:
                return self._empty_trades_stats()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        '''
                        SELECT 
                            COUNT(*) AS total,
                            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) AS losses,
                            COALESCE(SUM(pnl), 0) AS total_pnl
                        FROM trade_executions
                        WHERE created_at >= (CURRENT_DATE - INTERVAL '%s days')
                          AND status = 'closed'
                        ''',
                        (days,),
                    )
                    r = cur.fetchone()
                    if not r:
                        return self._empty_trades_stats()

                    total = r[0] or 0
                    wins = r[1] or 0
                    losses = r[2] or 0
                    total_pnl = float(r[3] or 0)
                    win_rate = (wins / total * 100.0) if total > 0 else 0.0

                    return {
                        "total": total,
                        "wins": wins,
                        "losses": losses,
                        "total_pnl": total_pnl,
                        "win_rate": win_rate,
                    }
            except Exception as e:
                logger.error(f"❌ Error getting trades last {days} days: {e}")
                return self._empty_trades_stats()

    def get_balance_change_last_days(self, days: int = 7) -> Dict[str, float]:
        """Get balance change over the last N days."""
        with get_db_connection() as conn:
            if not conn:
                return {"start_balance": 0.0, "end_balance": 0.0, "delta": 0.0}
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        '''
                        SELECT "timestamp", balance
                        FROM account_metrics
                        WHERE "timestamp" >= (CURRENT_DATE - INTERVAL '%s days')
                        ORDER BY "timestamp" ASC
                        ''',
                        (days,),
                    )
                    rows = cur.fetchall()
                    if not rows:
                        return {"start_balance": 0.0, "end_balance": 0.0, "delta": 0.0}

                    start_balance = float(rows[0][1] or 0)
                    end_balance = float(rows[-1][1] or 0)
                    delta = end_balance - start_balance
                    return {
                        "start_balance": start_balance,
                        "end_balance": end_balance,
                        "delta": delta,
                    }
            except Exception as e:
                logger.error(f"❌ Error getting balance change last {days} days: {e}")
                return {"start_balance": 0.0, "end_balance": 0.0, "delta": 0.0}

    def get_trades_last_minutes(self, minutes: int = 60) -> Dict[str, Any]:
        """Get trading statistics for the last N minutes."""
        with get_db_connection() as conn:
            if not conn:
                return self._empty_trades_stats()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        '''
                        SELECT 
                            COUNT(*) AS total,
                            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) AS losses,
                            COALESCE(SUM(pnl), 0) AS total_pnl
                        FROM trade_executions
                        WHERE created_at >= (NOW() AT TIME ZONE 'UTC' - INTERVAL '%s minutes')
                          AND status = 'closed'
                        ''',
                        (minutes,),
                    )
                    r = cur.fetchone()
                    if not r:
                        return self._empty_trades_stats()

                    total = r[0] or 0
                    wins = r[1] or 0
                    losses = r[2] or 0
                    total_pnl = float(r[3] or 0)
                    win_rate = (wins / total * 100.0) if total > 0 else 0.0

                    return {
                        "total": total,
                        "wins": wins,
                        "losses": losses,
                        "total_pnl": total_pnl,
                        "win_rate": win_rate,
                    }
            except Exception as e:
                logger.error(f"❌ Error getting trades last {minutes} minutes: {e}")
                return self._empty_trades_stats()

    def get_positions_opened_last_minutes(self, minutes: int = 60) -> int:
        """Get count of positions opened in the last N minutes."""
        with get_db_connection() as conn:
            if not conn:
                return 0
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        '''
                        SELECT COUNT(*)
                        FROM trade_executions
                        WHERE created_at >= (NOW() AT TIME ZONE 'UTC' - INTERVAL '%s minutes')
                        ''',
                        (minutes,),
                    )
                    r = cur.fetchone()
                    return int(r[0] or 0) if r else 0
            except Exception as e:
                logger.error(f"❌ Error getting positions last {minutes} minutes: {e}")
                return 0

    # ─────────────────────────────────────────────────────
    # Database Queries - ML
    # ─────────────────────────────────────────────────────
    def get_ml_active_models(self) -> List[Dict[str, Any]]:
        """Get currently active ML models."""
        if not REPORT_ML_METRICS:
            return []
            
        with get_ml_db_connection() as conn:
            if not conn:
                return []
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            model_type,
                            model_name,
                            symbol,
                            version,
                            deployed_at,
                            total_predictions,
                            successful_predictions,
                            validation_score
                        FROM ml_models
                        WHERE is_active = true
                        ORDER BY deployed_at DESC
                    """)
                    rows = cur.fetchall()
                    
                    return [
                        {
                            "type": r[0],
                            "name": r[1],
                            "symbol": r[2] or "Global",
                            "version": r[3],
                            "deployed_at": r[4],
                            "predictions": r[5] or 0,
                            "successful": r[6] or 0,
                            "score": float(r[7] or 0),
                        }
                        for r in rows
                    ]
            except Exception as e:
                logger.debug(f"ML models query failed: {e}")
                return []

    def get_ml_predictions_today(self) -> Dict[str, Any]:
        """Get ML prediction statistics for today."""
        if not REPORT_ML_METRICS:
            return {}
            
        with get_ml_db_connection() as conn:
            if not conn:
                return {}
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            COUNT(*) AS total,
                            AVG(model_confidence) AS avg_confidence,
                            AVG(blend_weight) AS avg_blend,
                            COUNT(CASE WHEN safety_adjusted = true THEN 1 END) AS adjusted_count,
                            AVG(ABS(predicted_sl_pct - actual_sl_pct)) AS mae_sl,
                            AVG(ABS(predicted_tp_pct - actual_tp_pct)) AS mae_tp
                        FROM ml_predictions
                        WHERE DATE(created_at) = CURRENT_DATE
                    """)
                    r = cur.fetchone()
                    
                    if not r or r[0] == 0:
                        return {}
                    
                    return {
                        "total": r[0] or 0,
                        "avg_confidence": float(r[1] or 0),
                        "avg_blend": float(r[2] or 0),
                        "adjusted_count": r[3] or 0,
                        "mae_sl": float(r[4] or 0),
                        "mae_tp": float(r[5] or 0),
                    }
            except Exception as e:
                logger.debug(f"ML predictions query failed: {e}")
                return {}

    def get_ml_performance_last_days(self, days: int = 7) -> Dict[str, Any]:
        """Get ML performance metrics for last N days."""
        if not REPORT_ML_METRICS:
            return {}
            
        with get_ml_db_connection() as conn:
            if not conn:
                return {}
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            AVG(win_rate) AS avg_win_rate,
                            AVG(traditional_win_rate) AS traditional_win_rate,
                            AVG(improvement_pct) AS improvement_pct,
                            AVG(mae_sl) AS avg_mae_sl,
                            AVG(mae_tp) AS avg_mae_tp,
                            SUM(total_pnl) AS total_pnl,
                            SUM(traditional_pnl) AS traditional_pnl
                        FROM ml_performance
                        WHERE eval_end_date >= (CURRENT_DATE - INTERVAL '%s days')
                    """, (days,))
                    r = cur.fetchone()
                    
                    if not r:
                        return {}
                    
                    return {
                        "ml_win_rate": float(r[0] or 0),
                        "traditional_win_rate": float(r[1] or 0),
                        "improvement_pct": float(r[2] or 0),
                        "avg_mae_sl": float(r[3] or 0),
                        "avg_mae_tp": float(r[4] or 0),
                        "ml_total_pnl": float(r[5] or 0),
                        "traditional_pnl": float(r[6] or 0),
                    }
            except Exception as e:
                logger.debug(f"ML performance query failed: {e}")
                return {}

    def get_ml_drift_alerts(self) -> List[Dict[str, Any]]:
        """Get recent data drift alerts."""
        if not REPORT_ML_METRICS:
            return []
            
        with get_ml_db_connection() as conn:
            if not conn:
                return []
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            detected_at,
                            drift_type,
                            symbol,
                            drift_score,
                            retraining_triggered
                        FROM ml_drift_detection
                        WHERE detected_at >= (NOW() - INTERVAL '24 hours')
                        ORDER BY detected_at DESC
                        LIMIT 5
                    """)
                    rows = cur.fetchall()
                    
                    return [
                        {
                            "detected_at": r[0],
                            "type": r[1],
                            "symbol": r[2] or "Global",
                            "score": float(r[3]),
                            "retraining": r[4],
                        }
                        for r in rows
                    ]
            except Exception as e:
                logger.debug(f"ML drift alerts query failed: {e}")
                return []

    def get_ml_training_data_count(self) -> Dict[str, int]:
        """Get training data statistics."""
        if not REPORT_ML_METRICS:
            return {}
            
        with get_ml_db_connection() as conn:
            if not conn:
                return {}
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            COUNT(*) AS total,
                            COUNT(CASE WHEN outcome = 'profit' THEN 1 END) AS profitable,
                            COUNT(CASE WHEN outcome = 'loss' THEN 1 END) AS losses,
                            COUNT(DISTINCT symbol) AS symbols
                        FROM ml_training_data
                    """)
                    r = cur.fetchone()
                    
                    if not r:
                        return {}
                    
                    return {
                        "total": r[0] or 0,
                        "profitable": r[1] or 0,
                        "losses": r[2] or 0,
                        "symbols": r[3] or 0,
                    }
            except Exception as e:
                logger.debug(f"ML training data query failed: {e}")
                return {}

    # ─────────────────────────────────────────────────────
    # Chart Generation
    # ─────────────────────────────────────────────────────
    def generate_daily_equity_chart(
        self,
        filename: str = "/app/logs/daily_equity.png"
    ) -> Optional[str]:
        """Generate daily equity chart."""
        series = self.get_today_equity_series()
        if not series:
            logger.warning("⚠️ No equity series for today – cannot build chart")
            return None

        try:
            times = []
            balances = []
            equities = []
            
            for row in series:
                ts = row["timestamp"]
                if hasattr(ts, 'astimezone'):
                    times.append(ts.astimezone(LOCAL_TZ))
                else:
                    times.append(ts)
                balances.append(row["balance"])
                equities.append(row["equity"])

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(times, balances, label="Balance", linewidth=2)
            ax.plot(times, equities, label="Equity", linewidth=2)

            ax.set_title("Daily Balance / Equity", fontsize=14)
            ax.set_xlabel("Time (Local)", fontsize=10)
            ax.set_ylabel("USDT", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend()

            fig.autofmt_xdate()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            fig.tight_layout()
            fig.savefig(filename, dpi=100)
            plt.close(fig)

            logger.info(f"📈 Daily equity chart saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"❌ Failed to generate daily equity chart: {e}")
            return None

    # ─────────────────────────────────────────────────────
    # Report Formatting
    # ─────────────────────────────────────────────────────
    def format_hourly_report(self) -> str:
        """Format hourly trading report."""
        now = datetime.now(LOCAL_TZ)
        account = self.get_account_status()
        positions = self.get_open_positions()
        trades = self.get_today_trades()
        heartbeat = self.get_last_heartbeat()
        hour_stats = self.get_trades_last_minutes(60)
        hour_positions = self.get_positions_opened_last_minutes(60)
        
        # ML metrics
        ml_predictions = self.get_ml_predictions_today() if REPORT_ML_METRICS else {}
        
        # Build account section
        if account:
            balance = float(account.get("balance", 0.0))
            equity = float(account.get("equity", 0.0))
            unrealized = float(account.get("unrealised_pnl", 0.0))
            realised = float(account.get("realised_pnl", 0.0))
            drawdown = max(0.0, ((balance - equity) / balance * 100)) if balance else 0.0
            
            balance_line = f"💰 Balance: ${balance:,.2f}"
            equity_line = f"📊 Equity: ${equity:,.2f}"
            unrealized_line = f"{'🟢' if unrealized >= 0 else '🔴'} Unrealized PnL: ${unrealized:+,.2f}"
            realised_line = f"💵 Realized PnL: ${realised:+,.2f}"
            drawdown_line = f"📉 Drawdown: {drawdown:.2f}%"
        else:
            balance_line = "💰 Balance: N/A"
            equity_line = "📊 Equity: N/A"
            unrealized_line = "🟢 Unrealized PnL: N/A"
            realised_line = "💵 Realized PnL: N/A"
            drawdown_line = "📉 Drawdown: N/A"
        
        # Heartbeat status
        if heartbeat:
            hb_status = heartbeat.get("status", "unknown")
            age_sec = heartbeat.get("age_sec")
            cycle = heartbeat.get("cycle_count", 0)
            if age_sec is not None:
                hb_line = f"❤️ Bot: {hb_status} (last beat {int(age_sec)}s ago, cycle #{cycle})"
            else:
                hb_line = f"❤️ Bot: {hb_status} (cycle #{cycle})"
        else:
            hb_line = "❤️ Bot Status: unknown"
        
        # ML section
        ml_section = ""
        if REPORT_ML_METRICS and ml_predictions and ml_predictions.get("total", 0) > 0:
            ml_section = (
                f"\n🤖 <b>ML Activity Today</b>\n"
                f"   Predictions: {ml_predictions['total']}\n"
                f"   Avg Confidence: {ml_predictions['avg_confidence']:.1%}\n"
                f"   Blend Weight: {ml_predictions['avg_blend']:.1%}\n"
            )
        
        report = (
            "🤖 <b>Bybit Trading Bot Report</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"{balance_line}\n"
            f"{equity_line}\n"
            f"{unrealized_line}\n"
            f"{realised_line}\n"
            f"{drawdown_line}\n\n"
            f"📈 Today's Trades: {trades.get('total', 0)} "
            f"({trades.get('wins', 0)}W/{trades.get('losses', 0)}L)\n"
            f"🏆 Win Rate: {trades.get('win_rate', 0):.1f}%\n"
            f"💵 Total PnL: ${trades.get('total_pnl', 0):+.2f}\n\n"
            f"⏱ Last 1h Trades: {hour_stats.get('total', 0)} "
            f"({hour_stats.get('wins', 0)}W/{hour_stats.get('losses', 0)}L)\n"
            f"💸 Last 1h PnL: ${hour_stats.get('total_pnl', 0):+.2f}\n"
            f"📂 New positions (1h): {hour_positions}\n"
            f"{ml_section}"
            f"\n📂 Open Positions: {len(positions)}\n"
            f"{hb_line}\n"
            f"🕐 {now.strftime('%Y-%m-%d %H:%M')} (Local)"
        )
        return report

    def format_daily_report(self) -> str:
        """Format daily trading report."""
        now = datetime.now(LOCAL_TZ)
        account = self.get_account_status()
        trades = self.get_today_trades()
        
        # ML metrics
        ml_predictions = self.get_ml_predictions_today() if REPORT_ML_METRICS else {}
        training_data = self.get_ml_training_data_count() if REPORT_ML_METRICS else {}
        
        balance = account.get("balance", 0.0)
        equity = account.get("equity", 0.0)
        unrealized = account.get("unrealised_pnl", 0.0)
        
        # ML section
        ml_section = ""
        if REPORT_ML_METRICS:
            if ml_predictions.get("total", 0) > 0:
                ml_section = (
                    f"\n🤖 <b>ML Performance Today</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 Predictions: {ml_predictions['total']}\n"
                    f"🎯 Avg Confidence: {ml_predictions['avg_confidence']:.1%}\n"
                    f"⚖️ Blend Weight: {ml_predictions['avg_blend']:.1%}\n"
                    f"🛡️ Safety Adjusted: {ml_predictions['adjusted_count']}\n"
                    f"📏 MAE SL: {ml_predictions['mae_sl']:.3f}%\n"
                    f"📏 MAE TP: {ml_predictions['mae_tp']:.3f}%\n"
                )
            
            if training_data.get("total", 0) > 0:
                win_pct = training_data['profitable'] / training_data['total'] * 100
                ml_section += (
                    f"\n📚 <b>Training Data</b>\n"
                    f"Total Samples: {training_data['total']}\n"
                    f"Profitable: {training_data['profitable']} ({win_pct:.1f}%)\n"
                    f"Symbols: {training_data['symbols']}\n"
                )
        
        report = (
            "📅 <b>Daily Trading Summary</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Balance: ${balance:,.2f}\n"
            f"📊 Equity: ${equity:,.2f}\n"
            f"{'🟢' if unrealized >= 0 else '🔴'} Unrealized PnL: ${unrealized:+,.2f}\n"
            f"💵 Realized PnL (today): ${trades.get('total_pnl', 0):+.2f}\n\n"
            f"📈 Trades Today: {trades.get('total', 0)}\n"
            f"   Wins: {trades.get('wins', 0)} / Losses: {trades.get('losses', 0)}\n"
            f"🏆 Win Rate: {trades.get('win_rate', 0):.1f}%\n"
            f"{ml_section}"
            f"\n🕐 {now.strftime('%Y-%m-%d %H:%M')} (Local)"
        )
        return report

    def format_weekly_summary(self) -> str:
        """Format weekly trading summary."""
        now = datetime.now(LOCAL_TZ)
        trades_7d = self.get_trades_last_days(7)
        balance_7d = self.get_balance_change_last_days(7)
        
        # ML performance comparison
        ml_perf = self.get_ml_performance_last_days(7) if REPORT_ML_METRICS else {}
        
        # ML section
        ml_section = ""
        if REPORT_ML_METRICS and ml_perf:
            improvement = ml_perf.get('improvement_pct', 0)
            emoji = "📈" if improvement > 0 else "📉"
            
            ml_section = (
                f"\n🤖 <b>ML vs Traditional (7 Days)</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"ML Win Rate: {ml_perf.get('ml_win_rate', 0):.1f}%\n"
                f"Traditional: {ml_perf.get('traditional_win_rate', 0):.1f}%\n"
                f"{emoji} Improvement: {improvement:+.1f}%\n"
                f"ML PnL: ${ml_perf.get('ml_total_pnl', 0):+.2f}\n"
                f"Traditional PnL: ${ml_perf.get('traditional_pnl', 0):+.2f}\n"
            )
        
        report = (
            "📊 <b>Weekly Trading Summary (7 Days)</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Start Balance: ${balance_7d.get('start_balance', 0):,.2f}\n"
            f"💰 End Balance:   ${balance_7d.get('end_balance', 0):,.2f}\n"
            f"📈 Balance Δ: ${balance_7d.get('delta', 0):+.2f}\n\n"
            f"📈 Trades: {trades_7d.get('total', 0)}\n"
            f"   Wins: {trades_7d.get('wins', 0)} / Losses: {trades_7d.get('losses', 0)}\n"
            f"🏆 Win Rate: {trades_7d.get('win_rate', 0):.1f}%\n"
            f"💵 Realized PnL: ${trades_7d.get('total_pnl', 0):+.2f}\n"
            f"{ml_section}"
            f"\n🕐 {now.strftime('%Y-%m-%d %H:%M')} (Local)"
        )
        return report

    # ─────────────────────────────────────────────────────
    # Report Sending
    # ─────────────────────────────────────────────────────
    def send_hourly_report(self) -> bool:
        """Send hourly report."""
        if not self.enabled:
            return False
        try:
            report = self.format_hourly_report()
            ok = self.send_message(report)
            if ok:
                logger.info("📤 Hourly report sent successfully")
            return ok
        except Exception as e:
            logger.error(f"❌ Error sending hourly report: {e}")
            return False

    def send_daily_report(self) -> bool:
        """Send daily report with chart."""
        if not self.enabled:
            return False
        try:
            msg = self.format_daily_report()
            self.send_message(msg)

            chart_path = self.generate_daily_equity_chart()
            if chart_path:
                self.send_photo(
                    chart_path,
                    caption="📈 <b>Daily Balance / Equity Chart</b>",
                )
            return True
        except Exception as e:
            logger.error(f"❌ Error sending daily report: {e}")
            return False

    def send_weekly_summary(self) -> bool:
        """Send weekly summary."""
        if not self.enabled:
            return False
        try:
            msg = self.format_weekly_summary()
            return self.send_message(msg)
        except Exception as e:
            logger.error(f"❌ Error sending weekly summary: {e}")
            return False

    def send_startup_notification(self) -> bool:
        """Send startup notification."""
        if not self.enabled:
            return False
        now = datetime.now(LOCAL_TZ)
        msg = (
            "🤖 <b>Telegram Reporter Started</b>\n"
            f"🕐 {now.strftime('%Y-%m-%d %H:%M')} (Local)\n"
            "─────────────────────\n"
            "✅ Reporter is running\n"
            "📊 Hourly / Daily / Weekly updates enabled\n"
        )
        return self.send_message(msg)

    def send_error_alert(self, message: str) -> bool:
        """Send error alert."""
        if not self.enabled:
            return False
        now = datetime.now(LOCAL_TZ)
        alert_text = (
            "🚨 <b><u>Bot Error Alert</u></b>\n"
            f"{message}\n"
            f"🕐 {now.strftime('%Y-%m-%d %H:%M:%S')} (Local)"
        )
        return self.send_message(alert_text)

    def send_ml_training_notification(self, model_info: Dict) -> bool:
        """Send notification when ML model is trained."""
        if not self.enabled or not REPORT_ML_METRICS:
            return False
        
        now = datetime.now(LOCAL_TZ)
        msg = (
            "🤖 <b>ML Model Training Complete</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"Model: {model_info.get('name', 'Unknown')}\n"
            f"Type: {model_info.get('type', 'Unknown')}\n"
            f"Version: {model_info.get('version', '1.0')}\n"
            f"Samples: {model_info.get('training_samples', 0)}\n"
            f"Validation Score: {model_info.get('validation_score', 0):.4f}\n"
            f"Status: {'✅ Deployed' if model_info.get('deployed') else '⏳ Pending'}\n"
            f"🕐 {now.strftime('%Y-%m-%d %H:%M')}"
        )
        return self.send_message(msg)

    def send_ml_drift_alert(self, drift_info: Dict) -> bool:
        """Send alert when data drift is detected."""
        if not self.enabled or not REPORT_ML_METRICS:
            return False
        
        now = datetime.now(LOCAL_TZ)
        msg = (
            "⚠️ <b>ML Data Drift Detected</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"Type: {drift_info.get('type', 'Unknown')}\n"
            f"Symbol: {drift_info.get('symbol', 'Global')}\n"
            f"Drift Score: {drift_info.get('score', 0):.4f}\n"
            f"Threshold: {drift_info.get('threshold', 0):.4f}\n"
            f"Action: {drift_info.get('action', 'None')}\n"
            f"Retraining: {'✅ Triggered' if drift_info.get('retraining') else '❌ Not triggered'}\n"
            f"🕐 {now.strftime('%Y-%m-%d %H:%M')}"
        )
        return self.send_message(msg)

    # ─────────────────────────────────────────────────────
    # Health Monitoring
    # ─────────────────────────────────────────────────────
    def check_bot_health(self, max_stale_sec: int = 180) -> None:
        """Check bot health via heartbeat."""
        if not self.enabled:
            return

        hb = self.get_last_heartbeat()
        if not hb:
            logger.debug("No heartbeat rows yet.")
            return

        age = hb.get("age_sec")
        status = hb.get("status", "unknown")
        cycle = hb.get("cycle_count", 0)
        msg = hb.get("message") or ""

        if age is None:
            return

        unhealthy = age > max_stale_sec or status not in ("running", "idle")
        if not unhealthy:
            return

        now = datetime.now(LOCAL_TZ)

        # Rate limit health alerts
        if self._last_health_alert_time:
            delta = (now - self._last_health_alert_time).total_seconds()
            if delta < max_stale_sec:
                return

        self._last_health_alert_time = now
        text = (
            f"Bot heartbeat looks unhealthy:\n"
            f"Status: {status}\n"
            f"Last beat age: {int(age)}s\n"
            f"Cycle: {cycle}\n"
            f"Note: {msg}"
        )
        logger.warning(text.replace("\n", " | "))
        self.send_error_alert(text)

# ─────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────
def main():
    """Main function for running the Telegram Reporter as a service."""
    logger.info("🚀 Starting Telegram Reporter...")
    logger.info(f"📊 Database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    
    if REPORT_ML_METRICS:
        logger.info(f"🤖 ML Database: {ML_DB_HOST}:{ML_DB_PORT}/{ML_DB_NAME}")

    if not wait_for_db():
        logger.error("❌ Cannot connect to database, exiting")
        sys.exit(1)

    wait_for_tables(
        ["signals", "trade_executions", "account_metrics", "bot_heartbeat"],
        timeout_sec=90
    )

    reporter = TelegramReporter()
    if not reporter.enabled:
        logger.warning("⚠️ Telegram reporting disabled")
        logger.info("💤 Sleeping indefinitely...")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            logger.info("🛑 Stopped by user")
        return

    reporter.send_startup_notification()

    # Initial hourly report
    reporter.send_hourly_report()

    if SCHEDULE_AVAILABLE and schedule:
        # Schedule reports
        schedule.every().hour.at(":00").do(reporter.send_hourly_report)
        schedule.every().day.at("00:05").do(reporter.send_daily_report)
        schedule.every().friday.at("12:00").do(reporter.send_weekly_summary)
        schedule.every(2).minutes.do(reporter.check_bot_health)

        logger.success("✅ Telegram Reporter initialized")
        logger.info("📊 Hourly / Daily / Weekly reports scheduled")
        logger.info("🔄 Reporter is running...")

        try:
            while True:
                schedule.run_pending()
                time.sleep(30)
        except KeyboardInterrupt:
            logger.info("🛑 Reporter stopped by user")
        except Exception as e:
            logger.error(f"❌ Reporter crashed: {e}")
        finally:
            reporter.cleanup()
    else:
        logger.warning("⚠️ Schedule library not available, running simple loop")
        last_hourly = time.time()
        
        try:
            while True:
                # Check every minute
                time.sleep(60)
                
                # Send hourly report
                if time.time() - last_hourly >= 3600:
                    reporter.send_hourly_report()
                    last_hourly = time.time()
                
                # Check bot health
                reporter.check_bot_health()
                
        except KeyboardInterrupt:
            logger.info("🛑 Reporter stopped by user")
        finally:
            reporter.cleanup()

if __name__ == "__main__":
    main()