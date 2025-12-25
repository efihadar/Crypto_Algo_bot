# heartbeat_manager.py
"""
💓 Heartbeat Manager - Bot Health Monitoring System
Tracks and records bot health status to database for monitoring.
Provides graceful degradation when DB is unavailable.
"""
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from loguru import logger

# Try to import write_heartbeat, provide fallback if unavailable
try:
    from bot.db_utils import write_heartbeat
    DB_HEARTBEAT_AVAILABLE = True
except ImportError:
    try:
        from db_utils import write_heartbeat
        DB_HEARTBEAT_AVAILABLE = True
    except ImportError:
        logger.warning("⚠️ write_heartbeat not available - heartbeats will only be logged")
        DB_HEARTBEAT_AVAILABLE = False
        write_heartbeat = None

class HeartbeatManager:
    """
    Enhanced heartbeat manager.
    Writes detailed bot-status rows into DB.

    Fields written to DB:
        - status        : running / error / stopped / idle / starting
        - cycle_count   : trading-loop counter
        - step          : current trading-engine stage
        - equity        : optional (float)
        - balance       : optional (float)
        - open_positions: optional (int)
        - message       : extra info
        - ts_utc        : timestamp
        
    Features:
        - Rate limiting to prevent DB flooding
        - Graceful degradation when DB unavailable
        - Error tracking and recovery status
        - History buffer for recent heartbeats
    """

    # Valid status values
    STATUS_RUNNING = "running"
    STATUS_ERROR = "error"
    STATUS_STOPPED = "stopped"
    STATUS_IDLE = "idle"
    STATUS_STARTING = "starting"
    STATUS_RECOVERING = "recovering"

    def __init__(
        self,
        min_interval_seconds: float = 10.0,
        history_size: int = 100
    ):
        """
        Initialize the HeartbeatManager.
        
        Args:
            min_interval_seconds: Minimum time between DB writes (rate limiting)
            history_size: Number of recent heartbeats to keep in memory
        """
        self.cycle: int = 0
        self.min_interval: float = min_interval_seconds
        self.last_beat_time: float = 0.0
        self.last_status: str = self.STATUS_STARTING
        self.last_error: Optional[str] = None
        self.error_count: int = 0
        self.consecutive_errors: int = 0
        self.db_available: bool = DB_HEARTBEAT_AVAILABLE
        
        # History buffer for debugging
        self.history_size: int = history_size
        self.history: List[Dict[str, Any]] = []
        
        # Track important metrics
        self.start_time: float = time.time()
        self.total_beats: int = 0
        self.successful_beats: int = 0
        self.failed_beats: int = 0
        
        logger.info("💓 Enhanced HeartbeatManager initialized")
        if not self.db_available:
            logger.warning("   ⚠️ DB heartbeat disabled - logging only mode")

    def beat(
        self,
        status: str = "running",
        message: str = "",
        step: str = "",
        equity: Optional[float] = None,
        balance: Optional[float] = None,
        open_positions: Optional[int] = None,
        force: bool = False
    ) -> bool:
        """
        Record a heartbeat.
        
        Args:
            status: Current bot status (running/error/stopped/idle/starting)
            message: Additional message or context
            step: Current execution step (e.g., "fetch_signals", "execute_trades")
            equity: Current account equity
            balance: Current account balance
            open_positions: Number of open positions
            force: If True, bypass rate limiting
            
        Returns:
            True if heartbeat was recorded, False if skipped or failed
            
        Example calls:
            heartbeat.beat(step="fetch_signals")
            heartbeat.beat(status="error", message="connection lost")
            heartbeat.beat(equity=1000.50, balance=950.00, open_positions=3)
        """
        current_time = time.time()
        
        # Rate limiting (unless forced or status change)
        time_since_last = current_time - self.last_beat_time
        status_changed = status != self.last_status
        
        if not force and not status_changed and time_since_last < self.min_interval:
            logger.debug(
                f"💓 Heartbeat skipped (rate limit): {time_since_last:.1f}s < {self.min_interval}s"
            )
            return False
        
        self.cycle += 1
        self.total_beats += 1
        self.last_beat_time = current_time
        self.last_status = status
        
        # Build payload
        payload: Dict[str, Any] = {
            "status": status,
            "cycle_count": self.cycle,
            "message": message,
            "step": step,
            "equity": equity,
            "balance": balance,
            "open_positions": open_positions,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
        }
        
        # Add to history
        self._add_to_history(payload)
        
        # Track errors
        if status == self.STATUS_ERROR:
            self.error_count += 1
            self.consecutive_errors += 1
            self.last_error = message
        else:
            self.consecutive_errors = 0
        
        # Try to write to DB
        success = self._write_to_db(payload)
        
        if success:
            self.successful_beats += 1
            log_level = "debug" if status == self.STATUS_RUNNING else "info"
            getattr(logger, log_level)(
                f"💓 Heartbeat: status={status}, cycle={self.cycle}, step={step}"
            )
        else:
            self.failed_beats += 1
        
        return success

    def _write_to_db(self, payload: Dict[str, Any]) -> bool:
        """
        Write heartbeat to database.
        
        Args:
            payload: Heartbeat data dict
            
        Returns:
            True if successful, False otherwise
        """
        if not self.db_available or write_heartbeat is None:
            # Log only mode
            logger.debug(f"💓 Heartbeat (no DB): {payload}")
            return True
        
        try:
            write_heartbeat(**payload)
            return True
        except Exception as e:
            logger.error(f"❌ Heartbeat DB write failed: {e}")
            # Don't disable DB permanently, it might recover
            return False

    def _add_to_history(self, payload: Dict[str, Any]) -> None:
        """Add heartbeat to in-memory history buffer."""
        self.history.append(payload.copy())
        
        # Trim history if needed
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]

    def error(self, message: str, step: str = "") -> bool:
        """
        Convenience method for error heartbeats.
        
        Args:
            message: Error description
            step: Where the error occurred
            
        Returns:
            True if heartbeat was recorded
        """
        return self.beat(
            status=self.STATUS_ERROR,
            message=message,
            step=step,
            force=True  # Always record errors
        )

    def started(self, message: str = "Bot started") -> bool:
        """Record bot startup."""
        return self.beat(
            status=self.STATUS_STARTING,
            message=message,
            step="initialization",
            force=True
        )

    def stopped(self, message: str = "Bot stopped") -> bool:
        """Record bot shutdown."""
        return self.beat(
            status=self.STATUS_STOPPED,
            message=message,
            step="shutdown",
            force=True
        )

    def idle(self, message: str = "") -> bool:
        """Record idle state (no signals, waiting)."""
        return self.beat(
            status=self.STATUS_IDLE,
            message=message
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get heartbeat statistics.
        
        Returns:
            Dict with uptime, beat counts, error info, etc.
        """
        uptime_seconds = time.time() - self.start_time
        
        return {
            "uptime_seconds": uptime_seconds,
            "uptime_formatted": self._format_uptime(uptime_seconds),
            "total_cycles": self.cycle,
            "total_beats": self.total_beats,
            "successful_beats": self.successful_beats,
            "failed_beats": self.failed_beats,
            "error_count": self.error_count,
            "consecutive_errors": self.consecutive_errors,
            "last_status": self.last_status,
            "last_error": self.last_error,
            "last_beat_time": datetime.fromtimestamp(
                self.last_beat_time, tz=timezone.utc
            ).isoformat() if self.last_beat_time > 0 else None,
            "db_available": self.db_available,
            "beats_per_minute": (self.total_beats / uptime_seconds * 60) if uptime_seconds > 0 else 0,
        }

    def get_recent_history(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent heartbeat history.
        
        Args:
            count: Number of recent heartbeats to return
            
        Returns:
            List of recent heartbeat payloads
        """
        return self.history[-count:] if count < len(self.history) else self.history.copy()

    def get_error_history(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent error heartbeats.
        
        Args:
            count: Number of recent errors to return
            
        Returns:
            List of error heartbeat payloads
        """
        errors = [h for h in self.history if h.get("status") == self.STATUS_ERROR]
        return errors[-count:] if count < len(errors) else errors

    def is_healthy(self, max_consecutive_errors: int = 5) -> bool:
        """
        Check if the bot appears healthy based on heartbeat history.
        
        Args:
            max_consecutive_errors: Threshold for unhealthy state
            
        Returns:
            True if healthy, False if too many consecutive errors
        """
        return self.consecutive_errors < max_consecutive_errors

    def reset_error_count(self) -> None:
        """Reset error counters (e.g., after recovery)."""
        self.consecutive_errors = 0
        self.last_error = None
        logger.info("💓 Error counters reset")

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """Format uptime in human-readable format."""
        days, remainder = divmod(int(seconds), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, secs = divmod(remainder, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")
        
        return " ".join(parts)

# ============================================================
# Convenience function for module-level access
# ============================================================
_default_manager: Optional[HeartbeatManager] = None

def get_heartbeat_manager() -> HeartbeatManager:
    """
    Get the default HeartbeatManager instance.
    Creates one if it doesn't exist.
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = HeartbeatManager()
    return _default_manager

def heartbeat(
    status: str = "running",
    message: str = "",
    step: str = "",
    **kwargs
) -> bool:
    """
    Convenience function for quick heartbeat recording.
    
    Example:
        from bot.heartbeat_manager import heartbeat
        heartbeat(step="fetch_signals")
        heartbeat(status="error", message="API timeout")
    """
    return get_heartbeat_manager().beat(
        status=status,
        message=message,
        step=step,
        **kwargs
    )