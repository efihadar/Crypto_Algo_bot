# watchdog/watchdog.py - Enhanced Bot Monitor
"""
🐶 Watchdog Service - Bot Health Monitor

Monitors the trading bot via heartbeat checks and automatically
restarts if the bot appears stuck or unresponsive.

Features:
- Heartbeat age monitoring
- Stuck cycle detection
- Auto-restart capability
- Telegram alerts
- Graceful shutdown
"""
import os
import sys
import time
import signal
import threading
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
from typing import Optional, Dict, Any
import psycopg2
import requests
from loguru import logger

# ─────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# ─────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG",
    colorize=True
)

log_path = os.getenv("WATCHDOG_LOG_PATH", "/app/logs/watchdog.log")
try:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger.add(log_path, rotation="1 day", retention="7 days", level="DEBUG")
except Exception as e:
    logger.warning(f"⚠️ Could not init file logger: {e}")

# ─────────────────────────────────────────────────────
# DB CONFIG
# ─────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", os.getenv("POSTGRES_HOST", "postgres_db"))
DB_PORT = int(os.getenv("DB_PORT", os.getenv("POSTGRES_PORT", "5432")))
DB_NAME = os.getenv("DB_NAME", os.getenv("POSTGRES_DB", "trading_db"))
DB_USER = os.getenv("DB_USER", os.getenv("POSTGRES_USER", "trader"))
DB_PASS = os.getenv("DB_PASS", os.getenv("POSTGRES_PASSWORD", "trader_pass"))

# Thresholds
MAX_HEARTBEAT_AGE_SEC = int(os.getenv("WATCHDOG_MAX_AGE", "300"))  # 5 min
CHECK_INTERVAL_SEC = int(os.getenv("WATCHDOG_CHECK_INTERVAL", "60"))
MAX_STUCK_CYCLES = int(os.getenv("WATCHDOG_MAX_STUCK_CYCLES", "3"))
MAX_DB_ERRORS = int(os.getenv("WATCHDOG_MAX_DB_ERRORS", "5"))
RESTART_COOLDOWN_SEC = int(os.getenv("WATCHDOG_RESTART_COOLDOWN", "120"))

# Telegram
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Auto-restart settings
AUTO_RESTART_ENABLED = os.getenv("WATCHDOG_AUTO_RESTART", "true").lower() in ("1", "true", "yes")
BOT_CONTAINER_NAME = os.getenv("BOT_CONTAINER_NAME", "crypto_trading_bot")

# ─────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────
class WatchdogState:
    """Thread-safe state container for watchdog."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self.running = True
        self.last_cycle_count = -1
        self.stuck_cycles = 0
        self.last_alert_time = 0.0
        self.last_restart_time = 0.0
        self.consecutive_db_errors = 0
        self.total_restarts = 0
        self.total_alerts = 0
        self.start_time = time.time()
        self.table_exists = False  # Track if table was verified
        self.no_heartbeat_warnings = 0  # Track consecutive no-heartbeat warnings
        
        # Alert cooldown
        self.alert_cooldown_sec = int(os.getenv("WATCHDOG_ALERT_COOLDOWN", "300"))
    
    def reset_stuck_counter(self):
        with self._lock:
            self.stuck_cycles = 0
    
    def increment_stuck(self) -> int:
        with self._lock:
            self.stuck_cycles += 1
            return self.stuck_cycles
    
    def update_cycle(self, new_cycle: int):
        with self._lock:
            self.last_cycle_count = new_cycle
            self.stuck_cycles = 0
    
    def can_send_alert(self) -> bool:
        with self._lock:
            now = time.time()
            if now - self.last_alert_time >= self.alert_cooldown_sec:
                return True
            return False
    
    def mark_alert_sent(self):
        with self._lock:
            self.last_alert_time = time.time()
            self.total_alerts += 1
    
    def can_restart(self) -> bool:
        with self._lock:
            now = time.time()
            if now - self.last_restart_time >= RESTART_COOLDOWN_SEC:
                return True
            return False
    
    def mark_restart(self):
        with self._lock:
            self.last_restart_time = time.time()
            self.total_restarts += 1
            self.last_cycle_count = -1
            self.stuck_cycles = 0
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            uptime = time.time() - self.start_time
            return {
                "uptime_seconds": uptime,
                "uptime_formatted": self._format_duration(uptime),
                "total_restarts": self.total_restarts,
                "total_alerts": self.total_alerts,
                "last_cycle": self.last_cycle_count,
                "stuck_cycles": self.stuck_cycles,
                "db_errors": self.consecutive_db_errors,
            }
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours}h {minutes}m {secs}s"

state = WatchdogState()

# ─────────────────────────────────────────────────────
# SIGNAL HANDLERS
# ─────────────────────────────────────────────────────
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.warning(f"🛑 Received signal {signum}, shutting down...")
    state.running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ─────────────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────────────
def send_alert(msg: str, force: bool = False) -> bool:
    """
    Send alert via Telegram.
    
    Args:
        msg: Alert message
        force: If True, bypass cooldown
        
    Returns:
        True if sent successfully
    """
    # Check cooldown
    if not force and not state.can_send_alert():
        logger.debug("⏸️ Alert cooldown active, skipping")
        return False
    
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning(f"⚠️ Telegram disabled — alert: {msg}")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        
        # Get timestamp
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S UTC")
        
        payload = {
            "chat_id": CHAT_ID,
            "text": f"🚨 <b>WATCHDOG ALERT</b>\n\n{msg}\n\n🕐 {timestamp}",
            "parse_mode": "HTML",
            "disable_notification": False,
        }
        
        resp = requests.post(url, json=payload, timeout=10)
        
        if resp.status_code == 200:
            logger.success("📤 Telegram alert sent")
            state.mark_alert_sent()
            return True
        else:
            logger.error(f"❌ Telegram API error: {resp.status_code} - {resp.text}")
            return False
    
    except requests.Timeout:
        logger.error("❌ Telegram request timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to send telegram alert: {e}")
        return False

def send_status_update(msg: str) -> bool:
    """Send status update (not an alert, so no cooldown check)."""
    if not BOT_TOKEN or not CHAT_ID:
        return False
    
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": f"🐶 <b>Watchdog Status</b>\n\n{msg}",
            "parse_mode": "HTML",
            "disable_notification": True,
        }
        
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code == 200
        
    except Exception as e:
        logger.debug(f"Status update failed: {e}")
        return False

# ─────────────────────────────────────────────────────
# DB CONNECTION
# ─────────────────────────────────────────────────────
@contextmanager
def get_db_connection():
    """
    Context manager for database connection with proper cleanup.
    """
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

def check_db_connection() -> bool:
    """Quick database health check."""
    with get_db_connection() as conn:
        if not conn:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
        except Exception:
            return False

def wait_for_db(max_retries: int = 30, delay: int = 2) -> bool:
    """Wait for database to become available."""
    logger.info("⏳ Waiting for database...")
    
    for i in range(max_retries):
        if check_db_connection():
            logger.success("✅ Database connection established")
            return True
        
        if i < max_retries - 1:
            logger.debug(f"DB not ready, retry {i+1}/{max_retries}...")
            time.sleep(delay)
    
    logger.error("❌ Database connection timeout")
    return False

# ─────────────────────────────────────────────────────
# ENSURE HEARTBEAT TABLE EXISTS
# ─────────────────────────────────────────────────────
def ensure_heartbeat_table() -> bool:
    """
    Create the bot_heartbeat table if it doesn't exist.
    
    Returns:
        True if table exists or was created successfully
    """
    create_sql = """
    CREATE TABLE IF NOT EXISTS bot_heartbeat (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMPTZ DEFAULT NOW(),
        status VARCHAR(50) DEFAULT 'unknown',
        cycle_count INTEGER DEFAULT 0,
        message TEXT,
        step VARCHAR(100),
        equity DECIMAL(20, 8),
        balance DECIMAL(20, 8),
        open_positions INTEGER DEFAULT 0,
        extra_data JSONB
    );
    
    -- Create index for faster lookups
    CREATE INDEX IF NOT EXISTS idx_bot_heartbeat_timestamp 
    ON bot_heartbeat(timestamp DESC);
    """
    
    with get_db_connection() as conn:
        if not conn:
            return False
        
        try:
            with conn.cursor() as cur:
                cur.execute(create_sql)
            conn.commit()
            logger.success("✅ bot_heartbeat table verified/created")
            state.table_exists = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create heartbeat table: {e}")
            return False

def check_table_exists() -> bool:
    """Check if bot_heartbeat table exists."""
    with get_db_connection() as conn:
        if not conn:
            return False
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'bot_heartbeat'
                    );
                """)
                row = cur.fetchone()
                return row[0] if row else False
        except Exception:
            return False

# ─────────────────────────────────────────────────────
# FETCH LATEST HEARTBEAT
# ─────────────────────────────────────────────────────
def get_latest_heartbeat() -> Optional[Dict[str, Any]]:
    """
    Fetch the latest heartbeat from database.
    
    Returns:
        Dict with heartbeat data or None if not found
    """
    # Ensure table exists before querying
    if not state.table_exists:
        if not check_table_exists():
            if not ensure_heartbeat_table():
                logger.debug("⏳ Waiting for heartbeat table to be created by bot...")
                return None
        state.table_exists = True
    
    with get_db_connection() as conn:
        if not conn:
            return None

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT timestamp, status, cycle_count, message, 
                           step, equity, balance, open_positions
                    FROM bot_heartbeat
                    ORDER BY timestamp DESC
                    LIMIT 1;
                """)
                row = cur.fetchone()

                if not row:
                    return None

                # Parse timestamp with timezone handling
                hb_time = row[0]
                if hb_time.tzinfo is None:
                    hb_time = hb_time.replace(tzinfo=timezone.utc)

                return {
                    "timestamp": hb_time,
                    "status": row[1] or "unknown",
                    "cycle_count": row[2] or 0,
                    "message": row[3] or "",
                    "step": row[4] if len(row) > 4 else None,
                    "equity": float(row[5]) if len(row) > 5 and row[5] else None,
                    "balance": float(row[6]) if len(row) > 6 and row[6] else None,
                    "open_positions": int(row[7]) if len(row) > 7 and row[7] else None,
                }
        
        except psycopg2.errors.UndefinedTable:
            # Table doesn't exist yet - this is expected during startup
            logger.debug("⏳ bot_heartbeat table not created yet, waiting for bot...")
            state.table_exists = False
            return None
        
        except Exception as e:
            # Check if it's a "relation does not exist" error
            error_str = str(e).lower()
            if "does not exist" in error_str or "undefined_table" in error_str:
                logger.debug("⏳ bot_heartbeat table not ready, waiting...")
                state.table_exists = False
                return None
            
            logger.error(f"❌ Failed reading heartbeat: {e}")
            return None

def get_heartbeat_count() -> int:
    """Get total number of heartbeat records."""
    if not state.table_exists and not check_table_exists():
        return 0
    
    with get_db_connection() as conn:
        if not conn:
            return 0
        
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM bot_heartbeat;")
                row = cur.fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

# ─────────────────────────────────────────────────────
# RESTART BOT
# ─────────────────────────────────────────────────────
def restart_trading_bot() -> bool:
    """
    Restart the trading bot via Docker.
    
    Returns:
        True if restart was initiated
    """
    if not AUTO_RESTART_ENABLED:
        logger.info("ℹ️ Auto-restart disabled, skipping")
        return False
    
    if not state.can_restart():
        remaining = RESTART_COOLDOWN_SEC - (time.time() - state.last_restart_time)
        logger.warning(f"⏸️ Restart cooldown active, {remaining:.0f}s remaining")
        return False
    
    try:
        logger.warning(f"♻️ Restarting container: {BOT_CONTAINER_NAME}")
        
        # Try different docker commands
        commands = [
            f"docker compose restart {BOT_CONTAINER_NAME}",
            f"docker-compose restart {BOT_CONTAINER_NAME}",
            f"docker restart {BOT_CONTAINER_NAME}",
        ]
        
        for cmd in commands:
            logger.debug(f"Trying: {cmd}")
            result = os.system(cmd)
            
            if result == 0:
                state.mark_restart()
                logger.success("♻️ Bot restart initiated")
                send_alert(
                    f"♻️ <b>Bot Restarted</b>\n\n"
                    f"Command: <code>{cmd}</code>\n"
                    f"Total restarts: {state.total_restarts}",
                    force=True
                )
                return True
        
        # All commands failed
        logger.error("❌ All restart commands failed")
        send_alert(
            f"❌ <b>Restart FAILED</b>\n\n"
            f"Container: {BOT_CONTAINER_NAME}\n"
            f"Manual intervention required!",
            force=True
        )
        return False
    
    except Exception as e:
        logger.error(f"❌ Restart exception: {e}")
        send_alert(f"❌ Restart exception: {e}", force=True)
        return False

# ─────────────────────────────────────────────────────
# HEALTH CHECKS
# ─────────────────────────────────────────────────────
def check_heartbeat_age(hb: Dict[str, Any]) -> Optional[str]:
    """
    Check if heartbeat is too old.
    
    Returns:
        Alert message if too old, None if OK
    """
    now = datetime.now(timezone.utc)
    hb_time = hb["timestamp"]
    
    # Ensure timezone-aware comparison
    if hb_time.tzinfo is None:
        hb_time = hb_time.replace(tzinfo=timezone.utc)
    
    age_sec = (now - hb_time).total_seconds()
    
    if age_sec > MAX_HEARTBEAT_AGE_SEC:
        # Build equity string safely
        equity_str = f"${hb['equity']:.2f}" if hb.get('equity') is not None else "N/A"
        balance_str = f"${hb['balance']:.2f}" if hb.get('balance') is not None else "N/A"
        
        return (
            f"❗ <b>HEARTBEAT TOO OLD</b>\n\n"
            f"Last update: {hb_time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Age: {age_sec:.0f}s (max: {MAX_HEARTBEAT_AGE_SEC}s)\n"
            f"Status: {hb['status']}\n"
            f"Cycle: {hb['cycle_count']}\n"
            f"Step: {hb.get('step', 'N/A')}\n"
            f"Equity: {equity_str}\n"
            f"Balance: {balance_str}"
        )
    
    return None

def check_stuck_cycle(hb: Dict[str, Any]) -> Optional[str]:
    """
    Check if bot is stuck on the same cycle.
    
    Returns:
        Alert message if stuck, None if OK
    """
    current_cycle = hb["cycle_count"]
    
    if current_cycle == state.last_cycle_count and state.last_cycle_count >= 0:
        stuck_count = state.increment_stuck()
        stuck_duration = stuck_count * CHECK_INTERVAL_SEC
        
        logger.warning(
            f"⚠️ Bot stuck on cycle {current_cycle} for {stuck_count} checks "
            f"({stuck_duration}s)"
        )
        
        if stuck_count >= MAX_STUCK_CYCLES:
            return (
                f"⚠️ <b>BOT APPEARS STUCK</b>\n\n"
                f"Cycle: {current_cycle} (unchanged for {stuck_count} checks)\n"
                f"Step: {hb.get('step', 'N/A')}\n"
                f"Duration: {stuck_duration}s\n"
                f"Last heartbeat: {hb['timestamp'].strftime('%H:%M:%S UTC')}"
            )
    else:
        # Cycle progressed
        if state.last_cycle_count >= 0 and current_cycle > state.last_cycle_count:
            logger.success(f"✅ Cycle progressed: {state.last_cycle_count} → {current_cycle}")
        
        state.update_cycle(current_cycle)
    
    return None

def check_error_status(hb: Dict[str, Any]) -> Optional[str]:
    """
    Check if bot is in error status.
    
    Returns:
        Alert message if error, None if OK
    """
    status = hb.get("status", "").lower()
    
    if status == "error":
        message = hb.get("message", "No details")
        return (
            f"🔴 <b>BOT IN ERROR STATE</b>\n\n"
            f"Status: {status}\n"
            f"Message: {message}\n"
            f"Cycle: {hb['cycle_count']}\n"
            f"Step: {hb.get('step', 'N/A')}"
        )
    
    return None

# ─────────────────────────────────────────────────────
# MAIN WATCHDOG LOOP
# ─────────────────────────────────────────────────────
def run_health_check() -> bool:
    """
    Run a single health check iteration.
    
    Returns:
        True if bot is healthy, False otherwise
    """
    hb = get_latest_heartbeat()

    if hb is None:
        state.no_heartbeat_warnings += 1
        
        # Only log warning every few checks to avoid spam
        if state.no_heartbeat_warnings <= 5:
            logger.warning(
                f"⚠️ No heartbeat found ({state.no_heartbeat_warnings}/5) - "
                f"waiting for bot to start..."
            )
        elif state.no_heartbeat_warnings % 10 == 0:
            logger.warning(
                f"⚠️ Still no heartbeat after {state.no_heartbeat_warnings} checks"
            )
        
        # Only alert after extended period with no heartbeat
        if state.no_heartbeat_warnings >= MAX_DB_ERRORS:
            send_alert(
                f"❗ <b>NO HEARTBEAT DATA</b>\n\n"
                f"No heartbeat for {state.no_heartbeat_warnings * CHECK_INTERVAL_SEC}s\n\n"
                f"Possible causes:\n"
                f"• Bot not running\n"
                f"• Bot still starting up\n"
                f"• Database connection issue\n"
                f"• Bot crashed before first heartbeat"
            )
            state.no_heartbeat_warnings = 0  # Reset after alert
        
        return False

    # Reset no-heartbeat counter on success
    state.no_heartbeat_warnings = 0
    state.consecutive_db_errors = 0

    # Extract values for logging
    hb_time = hb["timestamp"]
    hb_status = hb["status"]
    hb_cycle = hb["cycle_count"]
    hb_step = hb.get("step", "unknown")
    hb_positions = hb.get("open_positions", "?")
    
    now = datetime.now(timezone.utc)
    age_sec = (now - hb_time).total_seconds()

    logger.info(
        f"❤️ {hb_time.strftime('%H:%M:%S')} | "
        f"age={age_sec:.0f}s | "
        f"status={hb_status} | "
        f"cycle={hb_cycle} | "
        f"step={hb_step} | "
        f"positions={hb_positions}"
    )

    # Run health checks
    alert_msg = None
    needs_restart = False

    # Check 1: Heartbeat age
    alert_msg = check_heartbeat_age(hb)
    if alert_msg:
        needs_restart = True

    # Check 2: Stuck cycle
    if not alert_msg:
        alert_msg = check_stuck_cycle(hb)
        if alert_msg:
            needs_restart = True

    # Check 3: Error status
    if not alert_msg:
        alert_msg = check_error_status(hb)
        # Don't auto-restart on error status, just alert

    # Handle issues
    if alert_msg:
        send_alert(alert_msg)
        
        if needs_restart and AUTO_RESTART_ENABLED:
            if restart_trading_bot():
                time.sleep(60)  # Wait after restart
                return False
    
    return alert_msg is None

def main():
    """Main watchdog loop."""
    logger.info("=" * 60)
    logger.info("🐶 WATCHDOG STARTING")
    logger.info("=" * 60)
    logger.info(f"Database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    logger.info(f"Bot container: {BOT_CONTAINER_NAME}")
    logger.info(f"Max heartbeat age: {MAX_HEARTBEAT_AGE_SEC}s")
    logger.info(f"Check interval: {CHECK_INTERVAL_SEC}s")
    logger.info(f"Max stuck cycles: {MAX_STUCK_CYCLES}")
    logger.info(f"Auto-restart: {AUTO_RESTART_ENABLED}")
    logger.info(f"Restart cooldown: {RESTART_COOLDOWN_SEC}s")
    logger.info(f"Alert cooldown: {state.alert_cooldown_sec}s")
    logger.info("=" * 60)

    # Wait for database
    if not wait_for_db():
        logger.error("❌ Cannot connect to database, exiting")
        sys.exit(1)

    # Ensure heartbeat table exists
    if not ensure_heartbeat_table():
        logger.warning("⚠️ Could not create heartbeat table, will wait for bot to create it")
    
    # Check if heartbeat table has data
    if state.table_exists:
        hb_count = get_heartbeat_count()
        if hb_count == 0:
            logger.info("⏳ Heartbeat table empty, waiting for bot to start...")
        else:
            logger.info(f"📊 Found {hb_count} heartbeat records")

    # Send startup notification
    send_status_update(
        f"🐶 Watchdog started\n\n"
        f"Monitoring: {BOT_CONTAINER_NAME}\n"
        f"Check interval: {CHECK_INTERVAL_SEC}s\n"
        f"Auto-restart: {'✅' if AUTO_RESTART_ENABLED else '❌'}"
    )

    logger.success("✅ Watchdog running")
    logger.info("─" * 60)

    # Main loop
    while state.running:
        try:
            run_health_check()
            
        except KeyboardInterrupt:
            logger.warning("⌨️ Watchdog interrupted by user")
            break
        
        except Exception as e:
            logger.error(f"❌ Watchdog loop error: {e}")
            state.consecutive_db_errors += 1

        # Wait for next check
        for _ in range(CHECK_INTERVAL_SEC):
            if not state.running:
                break
            time.sleep(1)

    # Shutdown
    logger.info("─" * 60)
    stats = state.get_stats()
    logger.info(f"📊 Final stats: {stats}")
    
    send_status_update(
        f"🐶 Watchdog stopped\n\n"
        f"Uptime: {stats['uptime_formatted']}\n"
        f"Total restarts: {stats['total_restarts']}\n"
        f"Total alerts: {stats['total_alerts']}"
    )
    
    logger.info("👋 Watchdog stopped")

if __name__ == "__main__":
    main()