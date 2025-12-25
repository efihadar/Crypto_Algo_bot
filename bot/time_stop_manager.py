# time_stop_manager.py
"""
Time-Based Stop Manager
Closes positions after maximum hold time or implements time-decay trailing stops
"""
import time
from typing import Dict, List, Optional, Tuple
from loguru import logger
from datetime import datetime, timedelta


class TimeStopManager:
    """Manages time-based exits for positions."""
    
    def __init__(self, config_loader):
        """
        Args:
            config_loader: ConfigLoader instance
        """
        self.cfg = config_loader
        
        # Load settings
        section = "time_stops"
        self.enabled = self.cfg.get(section, "ENABLED", True, bool)
        self.max_hold_hours = self.cfg.get(section, "MAX_HOLD_HOURS", 72, int)
        self.max_hold_hours_momentum = self.cfg.get(section, "MAX_HOLD_HOURS_MOMENTUM", 4, int)
        
        # Time-decay trailing (tighten stops over time)
        self.enable_time_decay = self.cfg.get(section, "ENABLE_TIME_DECAY", True, bool)
        self.decay_intervals = self._parse_decay_intervals(
            self.cfg.get(section, "DECAY_INTERVALS", "24:1.0,48:0.7,72:0.5", str)
        )
        
        # Inactivity stop (close if no progress)
        self.enable_inactivity_stop = self.cfg.get(section, "ENABLE_INACTIVITY_STOP", True, bool)
        self.inactivity_hours = self.cfg.get(section, "INACTIVITY_HOURS", 12, int)
        self.inactivity_profit_threshold = self.cfg.get(section, "INACTIVITY_PROFIT_THRESHOLD", 0.5, float)
        
        logger.info(
            f"⏱️ TimeStopManager initialized: "
            f"max_hold={self.max_hold_hours}h, "
            f"decay={self.enable_time_decay}, "
            f"inactivity={self.enable_inactivity_stop}"
        )
    
    def _parse_decay_intervals(self, intervals_str: str) -> List[Tuple[int, float]]:
        """
        Parse decay intervals from config string.
        Format: "24:1.0,48:0.7,72:0.5"
        Meaning: After 24h use 1.0x ATR, after 48h use 0.7x ATR, etc.
        Args:
            intervals_str: Config string
        Returns:
            List of (hours, multiplier) tuples
        """
        try:
            intervals = []
            for pair in intervals_str.split(","):
                hours, mult = pair.split(":")
                intervals.append((int(hours), float(mult)))
            intervals.sort(key=lambda x: x[0])  # Sort by hours
            return intervals
        except Exception as e:
            logger.error(f"❌ Failed to parse decay intervals: {e}")
            return [(24, 1.0), (48, 0.7), (72, 0.5)]  # Default
    
    def should_close_by_time(self, position: Dict,current_price: float) -> Tuple[bool, str]:
        """
        Check if position should be closed due to time limits.
        Args:
            position: Position dictionary from order_manager
            current_price: Current market price
        Returns:
            (should_close, reason) tuple
        """
        if not self.enabled:
            return False, "Time stops disabled"
        
        try:
            symbol = position.get("symbol")
            opened_at = position.get("opened_at", time.time())
            is_momentum = position.get("is_momentum", False)
            
            # Calculate age
            age_seconds = time.time() - opened_at
            age_hours = age_seconds / 3600
            
            # Check max hold time
            max_hours = self.max_hold_hours_momentum if is_momentum else self.max_hold_hours
            
            if age_hours >= max_hours:
                logger.warning(
                    f"⏱️ {symbol} reached max hold time: {age_hours:.1f}h >= {max_hours}h"
                )
                return True, f"Max hold time ({age_hours:.1f}h)"
            
            # Check inactivity stop
            if self.enable_inactivity_stop:
                should_close, reason = self._check_inactivity_stop(
                    position, current_price, age_hours
                )
                if should_close:
                    return True, reason
            
            return False, "Time checks passed"
            
        except Exception as e:
            logger.error(f"❌ Time stop check failed: {e}")
            return False, f"Error: {e}"
    
    def _check_inactivity_stop(self,position: Dict,current_price: float,age_hours: float) -> Tuple[bool, str]:
        """
        Check if position is inactive (not making progress).
        Args:
            position: Position dictionary
            current_price: Current market price
            age_hours: Position age in hours
        Returns:
            (should_close, reason) tuple
        """
        if age_hours < self.inactivity_hours:
            return False, "Too young for inactivity check"
        
        try:
            symbol = position.get("symbol")
            entry_price = position.get("entry_price", 0)
            side = position.get("side", "BUY").upper()
            
            if entry_price == 0:
                return False, "No entry price"
            
            # Calculate profit percentage
            if side == "BUY":
                profit_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                profit_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Check if profit is below threshold
            if profit_pct < self.inactivity_profit_threshold:
                logger.warning(
                    f"⏱️ {symbol} inactive for {age_hours:.1f}h with profit={profit_pct:.2f}% "
                    f"< {self.inactivity_profit_threshold}%"
                )
                return True, f"Inactivity stop ({age_hours:.1f}h, {profit_pct:.2f}% profit)"
            
            return False, "Making progress"
            
        except Exception as e:
            logger.error(f"❌ Inactivity check failed: {e}")
            return False, f"Error: {e}"
    
    def get_time_decay_multiplier(self, position: Dict) -> float:
        """
        Get time-decay multiplier for trailing stop.
        As position gets older, trailing stop gets tighter.
        Args:
            position: Position dictionary
        Returns:
            Multiplier (e.g., 0.7 means use 70% of original ATR distance)
        """
        if not self.enable_time_decay:
            return 1.0
        
        try:
            opened_at = position.get("opened_at", time.time())
            age_hours = (time.time() - opened_at) / 3600
            
            # Find applicable interval
            multiplier = 1.0
            for hours, mult in self.decay_intervals:
                if age_hours >= hours:
                    multiplier = mult
                else:
                    break
            
            return multiplier
            
        except Exception as e:
            logger.error(f"❌ Failed to get decay multiplier: {e}")
            return 1.0
    
    def apply_time_based_adjustments(self,position: Dict,current_price: float,order_manager) -> bool:
        """
        Apply time-based adjustments to position (tighten stops, etc.).
        Args:
            position: Position dictionary
            current_price: Current market price
            order_manager: OrderManager instance
        Returns:
            True if adjustments were made
        """
        if not self.enabled or not self.enable_time_decay:
            return False
        
        try:
            symbol = position.get("symbol")
            multiplier = self.get_time_decay_multiplier(position)
            
            if multiplier >= 1.0:
                return False  # No decay yet
            
            # Get current trailing stop config
            trail = position.get("trail", {})
            original_distance = trail.get("original_distance_pct")
            
            if not original_distance:
                # Store original distance on first decay
                original_distance = trail.get("distance_pct", 0.5)
                trail["original_distance_pct"] = original_distance
            
            # Apply decay
            new_distance = original_distance * multiplier
            
            if abs(new_distance - trail.get("distance_pct", 0)) > 0.05:  # 5% change threshold
                trail["distance_pct"] = new_distance
                position["trail"] = trail
                
                logger.info(
                    f"⏱️ {symbol} time-decay applied: "
                    f"trailing distance {original_distance:.2f}% → {new_distance:.2f}% "
                    f"(multiplier={multiplier:.2f})"
                )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to apply time adjustments: {e}")
            return False
    
    def get_time_report(self, positions: List[Dict]) -> Dict:
        """
        Generate time-based report for positions.
        Args:
            positions: List of position dictionaries
        Returns:
            Report dictionary
        """
        if not positions:
            return {"total_positions": 0}
        
        try:
            now = time.time()
            
            ages = []
            near_max = []
            inactive = []
            
            for pos in positions:
                symbol = pos.get("symbol")
                opened_at = pos.get("opened_at", now)
                age_hours = (now - opened_at) / 3600
                
                ages.append(age_hours)
                
                # Check if near max hold time
                max_hours = (
                    self.max_hold_hours_momentum 
                    if pos.get("is_momentum") 
                    else self.max_hold_hours
                )
                
                if age_hours >= max_hours * 0.8:  # 80% of max
                    near_max.append({
                        "symbol": symbol,
                        "age_hours": round(age_hours, 1),
                        "max_hours": max_hours,
                        "remaining_hours": round(max_hours - age_hours, 1)
                    })
                
                # Check inactivity (simplified)
                if age_hours >= self.inactivity_hours:
                    inactive.append({
                        "symbol": symbol,
                        "age_hours": round(age_hours, 1)
                    })
            
            return {
                "total_positions": len(positions),
                "avg_age_hours": round(sum(ages) / len(ages), 1) if ages else 0,
                "oldest_hours": round(max(ages), 1) if ages else 0,
                "near_max_hold": near_max,
                "potentially_inactive": inactive,
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate time report: {e}")
            return {"error": str(e)}
