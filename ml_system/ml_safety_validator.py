# ml_system/ml_safety_validator.py
"""
🛡️ Professional ML Safety Validator
Enforces strict safety boundaries on ML predictions with comprehensive validation.
Prevents ML from making dangerous decisions in live trading environments.
Thread-safe and production-ready.
"""
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from loguru import logger
from .config import ml_config


class MLSafetyValidator:
    """
    Professional validator that enforces safety boundaries on ML predictions
    Key Principles:
    1. Hard boundaries on SL/TP percentages
    2. Minimum R/R ratio enforcement
    3. ATR-based dynamic limits
    4. Position size constraints
    5. Correlation with market conditions
    6. Emergency stop mechanisms
    """
    
    def __init__(self):
        """Initialize safety validator with configuration from ml_config"""
        try:
            # Load boundaries from ml_config
            self.sl_min_pct = ml_config.sl_min_pct
            self.sl_max_pct = ml_config.sl_max_pct
            self.tp_min_pct = ml_config.tp_min_pct
            self.tp_max_pct = ml_config.tp_max_pct
            
            self.min_rr = ml_config.min_risk_reward
            self.max_rr = ml_config.max_risk_reward
            
            # ATR-based dynamic boundaries
            self.enable_atr_bounds = ml_config.enable_atr_bounds
            self.sl_min_atr = ml_config.sl_min_atr_multiplier
            self.sl_max_atr = ml_config.sl_max_atr_multiplier
            self.tp_min_atr = ml_config.tp_min_atr_multiplier
            self.tp_max_atr = ml_config.tp_max_atr_multiplier
            
            # Volatility adjustments (using existing config or defaults)
            self.max_volatility_multiplier = getattr(ml_config, 'max_volatility_multiplier', 1.5)
            
            # Position sizing limits
            self.max_balance_fraction = getattr(ml_config, 'max_position_size_pct', 0.05)  # 5% default
            self.min_position_usdt = getattr(ml_config, 'min_position_usdt', 10.0)
            self.max_position_usdt = getattr(ml_config, 'max_position_usdt', 100.0)
            
            logger.success("🛡️ Professional ML Safety Validator initialized")
            logger.info(f"   SL bounds: {self.sl_min_pct:.2f}% - {self.sl_max_pct:.2f}%")
            logger.info(f"   TP bounds: {self.tp_min_pct:.2f}% - {self.tp_max_pct:.2f}%")
            logger.info(f"   R/R bounds: {self.min_rr:.2f} - {self.max_rr:.2f}")
            logger.info(f"   ATR bounds enabled: {self.enable_atr_bounds}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Safety Validator: {e}")
            # Set safe defaults
            self._set_safe_defaults()
    
    def _set_safe_defaults(self):
        """Set safe default values if configuration fails"""
        self.sl_min_pct = 0.5
        self.sl_max_pct = 3.0
        self.tp_min_pct = 1.0
        self.tp_max_pct = 5.0
        self.min_rr = 1.5
        self.max_rr = 5.0
        self.enable_atr_bounds = True
        self.sl_min_atr = 0.5
        self.sl_max_atr = 2.5
        self.tp_min_atr = 1.0
        self.tp_max_atr = 4.0
        self.max_volatility_multiplier = 1.5
        self.max_balance_fraction = 0.05
        self.min_position_usdt = 10.0
        self.max_position_usdt = 100.0
    
    def validate_stops(
        self,
        ml_sl: float,
        ml_tp: float,
        entry_price: float,
        side: str,
        atr: Optional[float] = None,
        volatility: Optional[float] = None,
        symbol: str = "UNKNOWN"
    ) -> Tuple[float, float, bool, str]:
        """
        Validate and enforce safety boundaries on ML predictions
        
        Args:
            ml_sl: ML predicted stop loss price
            ml_tp: ML predicted take profit price
            entry_price: Trade entry price
            side: "BUY" or "SELL"
            atr: Average True Range (for dynamic bounds)
            volatility: Recent volatility measure
            symbol: Trading symbol for logging
        
        Returns:
            (validated_sl, validated_tp, is_safe, reason)
        """
        try:
            # Validate inputs
            if not self._validate_inputs(ml_sl, ml_tp, entry_price, side):
                return ml_sl, ml_tp, False, "Invalid input parameters"
            
            side_up = side.upper()
            violations = []
            
            # Calculate original percentages
            original_sl_pct, original_tp_pct = self._calculate_percentages(entry_price, ml_sl, ml_tp, side_up)
            
            # Initialize working percentages
            sl_pct = original_sl_pct
            tp_pct = original_tp_pct
            
            # ========================================
            # 1. HARD PERCENTAGE BOUNDARIES
            # ========================================
            sl_pct, tp_pct, hard_violations = self._apply_hard_boundaries(sl_pct, tp_pct)
            violations.extend(hard_violations)
            
            # ========================================
            # 2. ATR-BASED DYNAMIC BOUNDARIES
            # ========================================
            if self.enable_atr_bounds and atr and atr > 0:
                sl_pct, tp_pct, atr_violations = self._apply_atr_boundaries(sl_pct, tp_pct, atr, entry_price)
                violations.extend(atr_violations)
            
            # ========================================
            # 3. VOLATILITY ADJUSTMENT
            # ========================================
            if volatility and volatility > 0:
                sl_pct, tp_pct, vol_violations = self._apply_volatility_adjustment(sl_pct, tp_pct, volatility)
                violations.extend(vol_violations)
            
            # ========================================
            # 4. RISK/REWARD RATIO ENFORCEMENT
            # ========================================
            sl_pct, tp_pct, rr_violations = self._enforce_rr_ratio(sl_pct, tp_pct)
            violations.extend(rr_violations)
            
            # ========================================
            # 5. SANITY CHECKS
            # ========================================
            if not self._validate_percentages(sl_pct, tp_pct):
                return ml_sl, ml_tp, False, "Invalid stops after validation"
            
            # ========================================
            # 6. CALCULATE FINAL PRICES
            # ========================================
            validated_sl, validated_tp = self._calculate_final_prices(entry_price, sl_pct, tp_pct, side_up)
            
            # ========================================
            # 7. FINAL POSITION VALIDATION
            # ========================================
            position_valid, position_reason = self._validate_final_positions(validated_sl, validated_tp, entry_price, side_up)
            if not position_valid:
                return ml_sl, ml_tp, False, position_reason
            
            # ========================================
            # 8. GENERATE RESULT
            # ========================================
            is_safe = len(violations) == 0
            
            if violations:
                reason = f"ML predictions adjusted: {'; '.join(violations)}"
                logger.warning(f"⚠️ {symbol}: {reason}")
            else:
                reason = "ML predictions within safe boundaries"
                logger.debug(f"✅ {symbol}: {reason}")
            
            # Log significant adjustments
            self._log_significant_adjustments(original_sl_pct, original_tp_pct, sl_pct, tp_pct, symbol, violations)
            
            return validated_sl, validated_tp, is_safe, reason
            
        except Exception as e:
            logger.error(f"❌ Safety validation failed for {symbol}: {e}")
            return ml_sl, ml_tp, False, f"Validation error: {str(e)}"
    
    def _validate_inputs(self, ml_sl: float, ml_tp: float, entry_price: float, side: str) -> bool:
        """Validate input parameters"""
        try:
            if any(pd.isna([ml_sl, ml_tp, entry_price]) for pd in [np]):
                return False
            
            if entry_price <= 0:
                logger.error(f"❌ Invalid entry price: {entry_price}")
                return False
            
            if not isinstance(side, str) or side.upper() not in ['BUY', 'SELL']:
                logger.error(f"❌ Invalid side: {side}")
                return False
            
            return True
        except:
            return False
    
    def _calculate_percentages(self, entry_price: float, ml_sl: float, ml_tp: float, side: str) -> Tuple[float, float]:
        """Calculate SL and TP percentages from prices"""
        try:
            if side == "BUY":
                sl_pct = ((entry_price - ml_sl) / entry_price) * 100
                tp_pct = ((ml_tp - entry_price) / entry_price) * 100
            else:  # SELL
                sl_pct = ((ml_sl - entry_price) / entry_price) * 100
                tp_pct = ((entry_price - ml_tp) / entry_price) * 100
            
            return sl_pct, tp_pct
        except:
            return 0.0, 0.0
    
    def _apply_hard_boundaries(self, sl_pct: float, tp_pct: float) -> Tuple[float, float, List[str]]:
        """Apply hard percentage boundaries"""
        violations = []
        
        if sl_pct < self.sl_min_pct:
            violations.append(f"SL {sl_pct:.2f}% < min {self.sl_min_pct}%")
            sl_pct = self.sl_min_pct
        
        if sl_pct > self.sl_max_pct:
            violations.append(f"SL {sl_pct:.2f}% > max {self.sl_max_pct}%")
            sl_pct = self.sl_max_pct
        
        if tp_pct < self.tp_min_pct:
            violations.append(f"TP {tp_pct:.2f}% < min {self.tp_min_pct}%")
            tp_pct = self.tp_min_pct
        
        if tp_pct > self.tp_max_pct:
            violations.append(f"TP {tp_pct:.2f}% > max {self.tp_max_pct}%")
            tp_pct = self.tp_max_pct
        
        return sl_pct, tp_pct, violations
    
    def _apply_atr_boundaries(self, sl_pct: float, tp_pct: float, atr: float, entry_price: float) -> Tuple[float, float, List[str]]:
        """Apply ATR-based dynamic boundaries"""
        violations = []
        
        try:
            atr_pct = (atr / entry_price) * 100
            
            # Calculate ATR-based limits
            sl_min_atr_pct = atr_pct * self.sl_min_atr
            sl_max_atr_pct = atr_pct * self.sl_max_atr
            tp_min_atr_pct = atr_pct * self.tp_min_atr
            tp_max_atr_pct = atr_pct * self.tp_max_atr
            
            # Apply ATR boundaries (only if they're more restrictive than hard limits)
            if sl_min_atr_pct > self.sl_min_pct and sl_pct < sl_min_atr_pct:
                violations.append(f"SL {sl_pct:.2f}% < ATR-based min {sl_min_atr_pct:.2f}%")
                sl_pct = sl_min_atr_pct
            
            if sl_max_atr_pct < self.sl_max_pct and sl_pct > sl_max_atr_pct:
                violations.append(f"SL {sl_pct:.2f}% > ATR-based max {sl_max_atr_pct:.2f}%")
                sl_pct = sl_max_atr_pct
            
            if tp_max_atr_pct < self.tp_max_pct and tp_pct > tp_max_atr_pct:
                violations.append(f"TP {tp_pct:.2f}% > ATR-based max {tp_max_atr_pct:.2f}%")
                tp_pct = tp_max_atr_pct
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply ATR boundaries: {e}")
        
        return sl_pct, tp_pct, violations
    
    def _apply_volatility_adjustment(self, sl_pct: float, tp_pct: float, volatility: float) -> Tuple[float, float, List[str]]:
        """Apply volatility-based adjustments"""
        violations = []
        
        try:
            # In high volatility, widen stops
            if volatility > 3.0:  # High volatility threshold
                vol_multiplier = min(volatility / 3.0, self.max_volatility_multiplier)
                original_sl = sl_pct
                original_tp = tp_pct
                
                sl_pct *= vol_multiplier
                tp_pct *= vol_multiplier
                
                # Ensure we don't exceed maximum bounds after adjustment
                sl_pct = min(sl_pct, self.sl_max_pct)
                tp_pct = min(tp_pct, self.tp_max_pct)
                
                if abs(sl_pct - original_sl) > 0.1 or abs(tp_pct - original_tp) > 0.1:
                    violations.append(f"High volatility ({volatility:.2f}%): stops widened by {vol_multiplier:.2f}x")
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply volatility adjustment: {e}")
        
        return sl_pct, tp_pct, violations
    
    def _enforce_rr_ratio(self, sl_pct: float, tp_pct: float) -> Tuple[float, float, List[str]]:
        """Enforce Risk/Reward ratio boundaries"""
        violations = []
        
        try:
            if sl_pct <= 0:
                return sl_pct, tp_pct, violations
            
            rr_ratio = tp_pct / sl_pct
            
            if rr_ratio < self.min_rr:
                # Widen TP to meet minimum R/R
                new_tp_pct = sl_pct * self.min_rr
                violations.append(f"R/R {rr_ratio:.2f} < min {self.min_rr}, adjusted TP to {new_tp_pct:.2f}%")
                tp_pct = new_tp_pct
                rr_ratio = self.min_rr
            
            elif rr_ratio > self.max_rr:
                # Tighten SL to meet maximum R/R
                new_sl_pct = tp_pct / self.max_rr
                # But don't go below minimum SL
                if new_sl_pct >= self.sl_min_pct:
                    violations.append(f"R/R {rr_ratio:.2f} > max {self.max_rr}, adjusted SL to {new_sl_pct:.2f}%")
                    sl_pct = new_sl_pct
                    rr_ratio = self.max_rr
                else:
                    # If tightening SL would violate minimum, adjust TP instead
                    new_tp_pct = self.sl_min_pct * self.max_rr
                    violations.append(f"R/R {rr_ratio:.2f} > max {self.max_rr}, adjusted TP to {new_tp_pct:.2f}%")
                    tp_pct = new_tp_pct
                    rr_ratio = self.max_rr
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to enforce R/R ratio: {e}")
        
        return sl_pct, tp_pct, violations
    
    def _validate_percentages(self, sl_pct: float, tp_pct: float) -> bool:
        """Validate that percentages are positive and reasonable"""
        try:
            if sl_pct <= 0 or tp_pct <= 0:
                logger.error(f"❌ Invalid stops after validation (SL={sl_pct}, TP={tp_pct})")
                return False
            
            if sl_pct > 100 or tp_pct > 100:  # Unrealistic percentages
                logger.error(f"❌ Unrealistic stops after validation (SL={sl_pct}%, TP={tp_pct}%)")
                return False
            
            return True
        except:
            return False
    
    def _calculate_final_prices(self, entry_price: float, sl_pct: float, tp_pct: float, side: str) -> Tuple[float, float]:
        """Calculate final SL and TP prices from percentages"""
        try:
            if side == "BUY":
                validated_sl = entry_price * (1 - sl_pct / 100)
                validated_tp = entry_price * (1 + tp_pct / 100)
            else:  # SELL
                validated_sl = entry_price * (1 + sl_pct / 100)
                validated_tp = entry_price * (1 - tp_pct / 100)
            
            return validated_sl, validated_tp
        except:
            return 0.0, 0.0
    
    def _validate_final_positions(self, sl_price: float, tp_price: float, entry_price: float, side: str) -> Tuple[bool, str]:
        """Validate final positions relative to entry price"""
        try:
            if side == "BUY":
                if sl_price >= entry_price:
                    return False, "Invalid BUY: SL must be below entry"
                if tp_price <= entry_price:
                    return False, "Invalid BUY: TP must be above entry"
            else:  # SELL
                if sl_price <= entry_price:
                    return False, "Invalid SELL: SL must be above entry"
                if tp_price >= entry_price:
                    return False, "Invalid SELL: TP must be below entry"
            
            return True, "Valid positions"
        except Exception as e:
            return False, f"Position validation error: {str(e)}"
    
    def _log_significant_adjustments(self, orig_sl: float, orig_tp: float, final_sl: float, final_tp: float, 
                                   symbol: str, violations: List[str]):
        """Log significant adjustments to stops"""
        try:
            if (abs(orig_sl - final_sl) > 0.1 or abs(orig_tp - final_tp) > 0.1) and violations:
                rr_ratio = final_tp / final_sl if final_sl > 0 else 0
                logger.info(
                    f"🛡️ {symbol} Safety Adjustment:\n"
                    f"   ML Original:  SL={orig_sl:.2f}%, TP={orig_tp:.2f}%\n"
                    f"   Validated:    SL={final_sl:.2f}%, TP={final_tp:.2f}%\n"
                    f"   R/R: {rr_ratio:.2f}\n"
                    f"   Adjustments: {len(violations)} changes applied"
                )
        except:
            pass
    
    def validate_position_size(
        self,
        ml_size: float,
        balance: float,
        symbol: str = "UNKNOWN"
    ) -> Tuple[float, bool, str]:
        """
        Validate ML-suggested position size with comprehensive checks
        
        Returns: (validated_size, is_safe, reason)
        """
        try:
            # Validate inputs
            if ml_size <= 0 or balance <= 0:
                return ml_size, False, "Invalid size or balance"
            
            violations = []
            validated_size = ml_size
            
            # Check against balance percentage
            max_allowed_by_balance = balance * self.max_balance_fraction
            if validated_size > max_allowed_by_balance:
                violations.append(
                    f"Size ${validated_size:.2f} > max allowed ${max_allowed_by_balance:.2f} "
                    f"({self.max_balance_fraction:.1%} of balance)"
                )
                validated_size = max_allowed_by_balance
            
            # Check absolute limits
            if validated_size < self.min_position_usdt:
                violations.append(f"Size ${validated_size:.2f} < min ${self.min_position_usdt:.2f}")
                validated_size = self.min_position_usdt
            
            if validated_size > self.max_position_usdt:
                violations.append(f"Size ${validated_size:.2f} > max ${self.max_position_usdt:.2f}")
                validated_size = self.max_position_usdt
            
            # Final sanity check
            if validated_size <= 0:
                return ml_size, False, "Invalid size after validation"
            
            is_safe = len(violations) == 0
            reason = "; ".join(violations) if violations else "Size within safe limits"
            
            if violations:
                logger.warning(f"⚠️ {symbol}: Position size adjusted - {reason}")
            
            return validated_size, is_safe, reason
            
        except Exception as e:
            logger.error(f"❌ Position size validation failed for {symbol}: {e}")
            return ml_size, False, f"Validation error: {str(e)}"
    
    def emergency_stop_check(
        self,
        current_price: float,
        entry_price: float,
        side: str,
        max_loss_pct: float = 5.0
    ) -> bool:
        """
        Emergency check: has loss exceeded absolute maximum?
        
        Returns True if emergency stop needed
        """
        try:
            if current_price <= 0 or entry_price <= 0:
                return False
            
            side_up = side.upper()
            
            if side_up == "BUY":
                loss_pct = ((entry_price - current_price) / entry_price) * 100
            else:  # SELL
                loss_pct = ((current_price - entry_price) / entry_price) * 100
            
            if loss_pct > max_loss_pct:
                logger.critical(
                    f"🚨 EMERGENCY STOP TRIGGERED: {side_up} position in {symbol} "
                    f"loss {loss_pct:.2f}% exceeds max {max_loss_pct:.2f}%"
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Emergency stop check failed: {e}")
            return False
    
    def get_safety_stats(self) -> Dict[str, Any]:
        """Get current safety configuration and statistics"""
        try:
            return {
                'configuration': {
                    'sl_bounds_pct': (self.sl_min_pct, self.sl_max_pct),
                    'tp_bounds_pct': (self.tp_min_pct, self.tp_max_pct),
                    'rr_bounds': (self.min_rr, self.max_rr),
                    'atr_enabled': self.enable_atr_bounds,
                    'sl_atr_bounds': (self.sl_min_atr, self.sl_max_atr),
                    'tp_atr_bounds': (self.tp_min_atr, self.tp_max_atr),
                    'max_volatility_multiplier': self.max_volatility_multiplier,
                    'position_limits': {
                        'max_balance_fraction': self.max_balance_fraction,
                        'min_position_usdt': self.min_position_usdt,
                        'max_position_usdt': self.max_position_usdt
                    }
                },
                'status': 'operational',
                'last_updated': datetime.now().isoformat() if 'datetime' in globals() else None
            }
        except Exception as e:
            logger.error(f"❌ Failed to get safety stats: {e}")
            return {'error': str(e), 'status': 'error'}
    
    def validate_prediction_confidence(self, confidence: float, min_threshold: float = None) -> bool:
        """
        Validate prediction confidence level
        """
        try:
            if min_threshold is None:
                min_threshold = ml_config.confidence_threshold
            
            return confidence >= min_threshold and 0.0 <= confidence <= 1.0
        except:
            return False
    
    def validate_trade_context(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate overall trade context before execution
        """
        try:
            if not isinstance(context, dict):
                return False, "Invalid context format"
            
            # Check required fields
            required_fields = ['symbol', 'side', 'entry_price']
            missing_fields = [field for field in required_fields if field not in context]
            if missing_fields:
                return False, f"Missing required fields: {missing_fields}"
            
            # Validate price
            entry_price = context.get('entry_price', 0)
            if entry_price <= 0:
                return False, "Invalid entry price"
            
            # Validate side
            side = str(context.get('side', '')).upper()
            if side not in ['BUY', 'SELL']:
                return False, f"Invalid side: {side}"
            
            # Check daily trade limit (if configured)
            if hasattr(ml_config, 'max_daily_trades'):
                # This would require tracking - implement in your system
                pass
            
            return True, "Trade context valid"
            
        except Exception as e:
            return False, f"Context validation error: {str(e)}"

# Global safety validator instance
try:
    safety_validator = MLSafetyValidator()
    logger.success("✅ ML Safety Validator initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize ML Safety Validator: {e}")
    safety_validator = None