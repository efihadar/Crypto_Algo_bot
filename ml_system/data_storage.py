# ml_system/data_storage.py
"""
🗄️ Professional ML Data Storage
Handles storage and retrieval of trade data for ML training with comprehensive monitoring.
Production-ready with error handling and integration with ML system.
"""
import numpy as np
import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import threading
import time
from loguru import logger

# Try to import pandas (optional dependency)
PANDAS_AVAILABLE = False
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
    logger.info("✅ pandas loaded successfully")
except ImportError:
    logger.warning("⚠️ pandas not available - CSV export disabled")

# Import local components
from .config import ml_config

class DataStorage:
    """
    Professional ML Data Storage with comprehensive error handling and monitoring
    Features:
    - Thread-safe operations
    - Data validation and sanitization
    - Automatic backup and recovery
    - Performance monitoring
    - Flexible storage formats
    """
    
    def __init__(self):
        """Initialize data storage"""
        self.data_dir = Path(ml_config.data_dir) if hasattr(ml_config, 'data_dir') else Path("data")
        self.trades_file = self.data_dir / "trade_history.json"
        self.backup_dir = self.data_dir / "backups"
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize trades file if needed
        if not self.trades_file.exists():
            self._save_trades([])
            logger.info(f"🆕 Created new trades file: {self.trades_file}")
        
        # Thread safety
        self._lock = threading.RLock()
        self._cache = {
            'trades': None,
            'last_modified': 0.0,
            'cache_time': 0.0
        }
        
        # Performance tracking
        self.stats = {
            'total_reads': 0,
            'total_writes': 0,
            'failed_operations': 0,
            'last_operation_time': 0.0
        }
        
        logger.success(f"✅ ML Data Storage initialized at: {self.data_dir}")
    
    def save_trade(self, trade_record: Dict) -> bool:
        """
        Save a trade record to storage with comprehensive validation
        
        Args:
            trade_record: Dictionary containing trade data and features
        
        Returns:
            True if saved successfully
        """
        start_time = time.time()
        try:
            # Validate input
            if not isinstance(trade_record, dict):
                logger.error("❌ Trade record must be a dictionary")
                return False
            
            # Validate required fields
            required_fields = ['symbol', 'side', 'entry_price', 'exit_price', 'pnl']
            missing_fields = [field for field in required_fields if field not in trade_record]
            if missing_fields:
                logger.error(f"❌ Missing required fields: {missing_fields}")
                return False
            
            # Sanitize and validate numeric fields
            numeric_fields = ['entry_price', 'exit_price', 'pnl', 'sl_used', 'tp_used']
            for field in numeric_fields:
                if field in trade_record:
                    try:
                        trade_record[field] = float(trade_record[field])
                    except (ValueError, TypeError):
                        logger.warning(f"⚠️ Invalid value for {field}: {trade_record[field]}")
                        trade_record[field] = 0.0
            
            # Add metadata
            with self._lock:
                # Load existing trades
                trades = self.load_trades()
                
                # Add timestamp if not present
                if 'recorded_at' not in trade_record:
                    trade_record['recorded_at'] = datetime.now(timezone.utc).isoformat()
                
                # Add unique ID
                trade_id = f"trade_{len(trades)}_{int(time.time())}"
                trade_record['trade_id'] = trade_id
                
                # Add version info
                trade_record['data_version'] = getattr(ml_config, '__version__', '1.0.0')
                
                # Validate features
                if 'features' in trade_record and not isinstance(trade_record['features'], dict):
                    logger.warning("⚠️ Features must be a dictionary - removing invalid features")
                    del trade_record['features']
                
                # Append new trade
                trades.append(trade_record)
                
                # Save back with backup
                if self._save_trades_with_backup(trades):
                    # Clear cache
                    self._cache['trades'] = None
                    self._cache['last_modified'] = time.time()
                    
                    # Update statistics
                    self.stats['total_writes'] += 1
                    self.stats['last_operation_time'] = time.time() - start_time
                    
                    logger.debug(
                        f"📝 Saved trade {trade_id}: {trade_record.get('symbol')} - "
                        f"${trade_record.get('pnl', 0):.2f} ({trade_record.get('outcome', 'unknown')})"
                    )
                    return True
                else:
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Failed to save trade record: {e}")
            self.stats['failed_operations'] += 1
            return False
    
    def _save_trades_with_backup(self, trades: List[Dict]) -> bool:
        """Save trades with automatic backup"""
        try:
            # Create backup if file exists
            if self.trades_file.exists():
                backup_path = self.backup_dir / f"trade_history_{int(time.time())}.json"
                try:
                    self.trades_file.rename(backup_path)
                    logger.debug(f"💾 Created backup: {backup_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to create backup: {e}")
            
            # Save new file
            self._save_trades(trades)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save trades with backup: {e}")
            return False
    
    def load_trades(self) -> List[Dict]:
        """
        Load all trade records with caching and error recovery
        
        Returns:
            List of trade records
        """
        start_time = time.time()
        try:
            with self._lock:
                # Check cache first
                current_mtime = self.trades_file.stat().st_mtime if self.trades_file.exists() else 0
                cache_age = time.time() - self._cache['cache_time']
                
                if (self._cache['trades'] is not None and 
                    current_mtime == self._cache['last_modified'] and 
                    cache_age < 60):  # Cache valid for 60 seconds
                    logger.debug("🎯 Using cached trades data")
                    return self._cache['trades'].copy()
                
                # Load from file
                if not self.trades_file.exists():
                    trades = []
                else:
                    with open(self.trades_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if not content:
                            trades = []
                        else:
                            trades = json.loads(content)
                
                # Validate data structure
                if not isinstance(trades, list):
                    logger.error("❌ Trades file contains invalid data structure")
                    trades = []
                
                # Validate each trade record
                valid_trades = []
                for i, trade in enumerate(trades):
                    try:
                        if isinstance(trade, dict):
                            # Ensure required fields
                            if 'trade_id' not in trade:
                                trade['trade_id'] = f"trade_legacy_{i}"
                            
                            if 'recorded_at' not in trade:
                                trade['recorded_at'] = "2024-01-01T00:00:00Z"
                            
                            # Convert numeric fields
                            numeric_fields = ['entry_price', 'exit_price', 'pnl', 'sl_used', 'tp_used']
                            for field in numeric_fields:
                                if field in trade and trade[field] is not None:
                                    try:
                                        trade[field] = float(trade[field])
                                    except (ValueError, TypeError):
                                        trade[field] = 0.0
                            
                            valid_trades.append(trade)
                        else:
                            logger.warning(f"⚠️ Skipping invalid trade record at index {i}")
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing trade {i}: {e}")
                        continue
                
                # Update cache
                self._cache['trades'] = valid_trades.copy()
                self._cache['last_modified'] = current_mtime
                self._cache['cache_time'] = time.time()
                
                # Update statistics
                self.stats['total_reads'] += 1
                self.stats['last_operation_time'] = time.time() - start_time
                
                logger.debug(f"📚 Loaded {len(valid_trades)} trade records")
                return valid_trades
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ Corrupted trades file: {e}")
            # Backup corrupted file
            backup_path = self.trades_file.with_suffix('.corrupted')
            try:
                if self.trades_file.exists():
                    self.trades_file.rename(backup_path)
                    logger.info(f"🔄 Backed up corrupted file to: {backup_path}")
            except Exception as backup_error:
                logger.error(f"❌ Failed to backup corrupted file: {backup_error}")
            
            # Return empty list and create new file
            self._save_trades([])
            self._cache['trades'] = []
            return []
            
        except Exception as e:
            logger.error(f"❌ Failed to load trades: {e}")
            self.stats['failed_operations'] += 1
            return []
    
    def _save_trades(self, trades: List[Dict]):
        """Save trades to file with comprehensive error handling"""
        try:
            # Create backup of current file if it exists
            if self.trades_file.exists():
                backup_path = self.backup_dir / f"trade_history_{int(time.time())}.json"
                try:
                    import shutil
                    shutil.copy2(self.trades_file, backup_path)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to create backup: {e}")
            
            # Save new data
            with open(self.trades_file, 'w', encoding='utf-8') as f:
                json.dump(trades, f, indent=2, default=str, ensure_ascii=False)
                
            logger.debug(f"💾 Saved {len(trades)} trades to {self.trades_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save trades: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about stored trades"""
        try:
            trades = self.load_trades()
            
            if not trades:
                return {
                    'total_trades': 0,
                    'profitable': 0,
                    'losing': 0,
                    'breakeven': 0,
                    'win_rate': 0.0,
                    'avg_pnl': 0.0,
                    'total_pnl': 0.0,
                    'best_trade': 0.0,
                    'worst_trade': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 0.0,
                    'sharpe_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'calmar_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'symbols': [],
                    'first_trade': None,
                    'last_trade': None,
                    'storage_stats': self.stats
                }
            
            pnls = [float(t.get('pnl', 0)) for t in trades]
            profitable = sum(1 for pnl in pnls if pnl > 0)
            losing = sum(1 for pnl in pnls if pnl < 0)
            breakeven = sum(1 for pnl in pnls if pnl == 0)
            
            wins = [pnl for pnl in pnls if pnl > 0]
            losses = [abs(pnl) for pnl in pnls if pnl < 0]
            
            avg_win = sum(wins) / len(wins) if wins else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            profit_factor = (sum(wins) / sum(losses)) if losses and sum(losses) > 0 else float('inf')
            
            # Calculate risk metrics
            cumulative_pnl = np.cumsum(pnls) if len(pnls) > 0 else [0]
            running_max = np.maximum.accumulate(cumulative_pnl) if len(cumulative_pnl) > 0 else [0]
            drawdown = running_max - cumulative_pnl
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
            
            # Calculate ratios
            std_pnl = np.std(pnls) if len(pnls) > 1 else 0.0
            sharpe_ratio = (np.mean(pnls) / std_pnl) if std_pnl > 0 else 0.0
            
            downside_returns = [r for r in pnls if r < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 1 else std_pnl
            sortino_ratio = (np.mean(pnls) / downside_std) if downside_std > 0 else 0.0
            
            calmar_ratio = (np.sum(pnls) / max_drawdown) if max_drawdown > 0 else 0.0
            
            # Get symbols and dates
            symbols = list(set(t.get('symbol', 'UNKNOWN') for t in trades))
            dates = [t.get('recorded_at') for t in trades if t.get('recorded_at')]
            first_trade = min(dates) if dates else None
            last_trade = max(dates) if dates else None
            
            stats = {
                'total_trades': len(trades),
                'profitable': profitable,
                'losing': losing,
                'breakeven': breakeven,
                'win_rate': (profitable / len(trades) * 100) if trades else 0.0,
                'avg_pnl': sum(pnls) / len(pnls) if pnls else 0.0,
                'total_pnl': sum(pnls),
                'best_trade': max(pnls) if pnls else 0.0,
                'worst_trade': min(pnls) if pnls else 0.0,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'symbols': symbols,
                'first_trade': first_trade,
                'last_trade': last_trade,
                'storage_stats': self.stats.copy()
            }
            
            logger.debug(f"📊 Generated statistics for {len(trades)} trades")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get statistics: {e}")
            return {'error': str(e)}
    
    def get_trades_for_training(self, min_samples: Optional[int] = None) -> List[Dict]:
        """
        Get trades suitable for training with comprehensive filtering
        Args: min_samples: Minimum samples required (uses config if not specified)
        Returns: List of trade records with features
        """
        try:
            trades = self.load_trades()
            
            # Filter trades with features and valid PnL
            valid_trades = [
                t for t in trades 
                if t.get('features') and 
                isinstance(t.get('features'), dict) and
                t.get('pnl') is not None
            ]
            
            min_required = min_samples or getattr(ml_config, 'min_samples_to_train', 50)
            
            if len(valid_trades) < min_required:
                logger.warning(
                    f"⚠️ Insufficient training data: {len(valid_trades)} < {min_required} "
                    f"(total trades: {len(trades)})"
                )
            
            logger.debug(f"🎯 Selected {len(valid_trades)} valid trades for training")
            return valid_trades
            
        except Exception as e:
            logger.error(f"❌ Failed to get trades for training: {e}")
            return []
    
    def get_recent_trades(self, n: int = 100, symbol: Optional[str] = None) -> List[Dict]:
        """Get N most recent trades, optionally filtered by symbol"""
        try:
            trades = self.load_trades()
            
            # Filter by symbol if specified
            if symbol:
                trades = [t for t in trades if t.get('symbol') == symbol]
            
            # Sort by recorded_at (most recent first)
            trades.sort(
                key=lambda x: x.get('recorded_at', ''), 
                reverse=True
            )
            
            result = trades[:n]
            logger.debug(f"🕒 Retrieved {len(result)} recent trades" + (f" for {symbol}" if symbol else ""))
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent trades: {e}")
            return []
    
    def clear_old_trades(self, keep_n: int = 1000, backup: bool = True) -> bool:
        """Keep only the N most recent trades with optional backup"""
        try:
            trades = self.load_trades()
            
            if len(trades) <= keep_n:
                logger.info(f"ℹ️ No trades to clean (current: {len(trades)}, keep: {keep_n})")
                return True
            
            # Sort by recorded_at (oldest first)
            trades.sort(
                key=lambda x: x.get('recorded_at', ''), 
                reverse=False
            )
            
            # Keep most recent
            trades_to_keep = trades[-keep_n:]
            
            # Create backup if requested
            if backup:
                backup_path = self.backup_dir / f"trades_cleared_{int(time.time())}.json"
                try:
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        json.dump(trades, f, indent=2, default=str, ensure_ascii=False)
                    logger.info(f"💾 Backed up {len(trades)} trades before cleaning")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to create backup: {e}")
            
            # Save cleaned data
            self._save_trades(trades_to_keep)
            
            # Clear cache
            with self._lock:
                self._cache['trades'] = None
                self._cache['last_modified'] = time.time()
            
            removed_count = len(trades) - len(trades_to_keep)
            logger.success(f"🧹 Cleaned {removed_count} old trades, kept {keep_n} most recent")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to clean old trades: {e}")
            return False
    
    def export_to_csv(self, filepath: str, include_features: bool = True) -> bool:
        """Export trades to CSV for analysis"""
        try:
            if not PANDAS_AVAILABLE:
                logger.error("❌ Cannot export to CSV - pandas not available")
                return False
            
            trades = self.load_trades()
            if not trades:
                logger.warning("⚠️ No trades to export")
                return False
            
            # Create DataFrame
            df_data = []
            for trade in trades:
                row = {}
                
                # Add basic trade info
                basic_fields = [
                    'trade_id', 'symbol', 'side', 'entry_price', 'exit_price', 
                    'pnl', 'sl_used', 'tp_used', 'recorded_at', 'outcome',
                    'signal_strength', 'model_version'
                ]
                
                for field in basic_fields:
                    row[field] = trade.get(field)
                
                # Add features if requested
                if include_features and 'features' in trade:
                    features = trade['features']
                    if isinstance(features, dict):
                        for feat_name, feat_value in features.items():
                            # Prefix feature names to avoid conflicts
                            row[f'feat_{feat_name}'] = feat_value
                
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Export to CSV
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            logger.success(f"✅ Exported {len(trades)} trades to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            return False
    
    def get_symbol_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get statistics for a specific symbol"""
        try:
            trades = self.get_recent_trades(n=1000, symbol=symbol)
            
            if not trades:
                return {'symbol': symbol, 'message': 'No trades found'}
            
            pnls = [float(t.get('pnl', 0)) for t in trades]
            profitable = sum(1 for pnl in pnls if pnl > 0)
            losing = sum(1 for pnl in pnls if pnl < 0)
            
            return {
                'symbol': symbol,
                'total_trades': len(trades),
                'profitable': profitable,
                'losing': losing,
                'win_rate': (profitable / len(trades) * 100) if trades else 0.0,
                'avg_pnl': sum(pnls) / len(pnls) if pnls else 0.0,
                'total_pnl': sum(pnls),
                'best_trade': max(pnls) if pnls else 0.0,
                'worst_trade': min(pnls) if pnls else 0.0,
                'last_trade_date': trades[0].get('recorded_at') if trades else None,
                'avg_holding_time': self._calculate_avg_holding_time(trades),
                'risk_reward_ratio': self._calculate_avg_risk_reward(trades)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get symbol statistics: {e}")
            return {'error': str(e)}
    
    def _calculate_avg_holding_time(self, trades: List[Dict]) -> Optional[float]:
        """Calculate average holding time in minutes"""
        try:
            holding_times = []
            for trade in trades:
                duration = trade.get('trade_duration_minutes')
                if duration is not None:
                    holding_times.append(float(duration))
            
            return sum(holding_times) / len(holding_times) if holding_times else None
        except:
            return None
    
    def _calculate_avg_risk_reward(self, trades: List[Dict]) -> Optional[float]:
        """Calculate average risk/reward ratio"""
        try:
            rr_ratios = []
            for trade in trades:
                sl_pct = trade.get('sl_pct_used')
                tp_pct = trade.get('tp_pct_used')
                if sl_pct and tp_pct and float(sl_pct) > 0:
                    rr_ratio = float(tp_pct) / float(sl_pct)
                    rr_ratios.append(rr_ratio)
            
            return sum(rr_ratios) / len(rr_ratios) if rr_ratios else None
        except:
            return None
    
    def get_storage_health(self) -> Dict[str, Any]:
        """Get health status of data storage"""
        try:
            trades = self.load_trades()
            file_size = self.trades_file.stat().st_size if self.trades_file.exists() else 0
            
            return {
                'status': 'healthy' if len(trades) > 0 else 'warning',
                'total_trades': len(trades),
                'file_size_bytes': file_size,
                'file_size_mb': file_size / (1024 * 1024),
                'last_modified': self.trades_file.stat().st_mtime if self.trades_file.exists() else None,
                'backup_count': len(list(self.backup_dir.glob("*.json"))),
                'cache_status': {
                    'cached': self._cache['trades'] is not None,
                    'cache_size': len(self._cache['trades']) if self._cache['trades'] else 0,
                    'cache_age_seconds': time.time() - self._cache['cache_time']
                },
                'performance_stats': self.stats.copy(),
                'directory_exists': self.data_dir.exists(),
                'file_exists': self.trades_file.exists()
            }
        except Exception as e:
            logger.error(f"❌ Failed to get storage health: {e}")
            return {'error': str(e)}

# Global storage instance with thread safety
_ml_storage_instance: Optional[DataStorage] = None
_ml_storage_lock = threading.Lock()

def get_ml_storage() -> DataStorage:
    """Get global ML storage instance with thread safety"""
    global _ml_storage_instance
    with _ml_storage_lock:
        if _ml_storage_instance is None:
            _ml_storage_instance = DataStorage()
        return _ml_storage_instance

def reset_ml_storage():
    """Reset global ML storage instance"""
    global _ml_storage_instance
    with _ml_storage_lock:
        if _ml_storage_instance:
            # Backup current data before resetting
            try:
                backup_path = f"data/trade_history_backup_{int(time.time())}.json"
                _ml_storage_instance.export_to_csv(backup_path.replace('.json', '.csv'))
            except:
                pass
        _ml_storage_instance = None
        logger.info("🔄 ML Storage instance reset")