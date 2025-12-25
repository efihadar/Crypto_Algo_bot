# walk_forward_tester.py
"""
Walk-Forward Testing System
Tests strategy on rolling time windows to validate robustness
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import json
from pathlib import Path


class WalkForwardTester:
    """Performs walk-forward analysis on trading strategy."""
    
    def __init__(self, session, strategy, config_loader):
        """
        Args:
            session: BybitSession instance
            strategy: Strategy instance to test
            config_loader: ConfigLoader instance
        """
        self.session = session
        self.strategy = strategy
        self.cfg = config_loader
        
        # Load settings
        section = "walk_forward"
        self.enabled = self.cfg.get(section, "ENABLED", False, bool)
        self.train_days = self.cfg.get(section, "TRAIN_DAYS", 60, int)
        self.test_days = self.cfg.get(section, "TEST_DAYS", 30, int)
        self.step_days = self.cfg.get(section, "STEP_DAYS", 7, int)
        self.min_trades = self.cfg.get(section, "MIN_TRADES", 20, int)
        
        # Results storage
        self.results_dir = Path("walk_forward_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(
            f"📊 WalkForwardTester initialized: "
            f"train={self.train_days}d, test={self.test_days}d, step={self.step_days}d"
        )
    
    def run_walk_forward_test(
        self,
        symbols: List[str],
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Run walk-forward test on given symbols.
        
        Args:
            symbols: List of symbols to test
            end_date: End date for testing (default: today)
            
        Returns:
            Results dictionary
        """
        if not self.enabled:
            logger.info("ℹ️ Walk-forward testing disabled")
            return {"enabled": False}
        
        try:
            logger.info(f"🧪 Starting walk-forward test on {len(symbols)} symbols...")
            
            end_date = end_date or datetime.now()
            
            # Calculate test windows
            windows = self._generate_windows(end_date)
            
            logger.info(f"📊 Testing {len(windows)} windows")
            
            # Run test on each window
            all_results = []
            
            for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
                logger.info(
                    f"\n{'='*60}\n"
                    f"Window {i+1}/{len(windows)}\n"
                    f"Train: {train_start.date()} to {train_end.date()}\n"
                    f"Test:  {test_start.date()} to {test_end.date()}\n"
                    f"{'='*60}"
                )
                
                window_results = self._test_window(
                    symbols, train_start, train_end, test_start, test_end
                )
                
                window_results["window_id"] = i + 1
                all_results.append(window_results)
            
            # Aggregate results
            summary = self._aggregate_results(all_results)
            
            # Save results
            self._save_results(all_results, summary)
            
            logger.success(f"✅ Walk-forward test completed!")
            logger.info(f"📊 Results saved to {self.results_dir}")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Walk-forward test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def _generate_windows(self, end_date: datetime) -> List[Tuple]:
        """
        Generate train/test windows for walk-forward analysis.
        
        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        windows = []
        
        # Start from earliest possible date
        total_days = self.train_days + self.test_days
        current_end = end_date
        
        while True:
            test_end = current_end
            test_start = test_end - timedelta(days=self.test_days)
            train_end = test_start - timedelta(days=1)
            train_start = train_end - timedelta(days=self.train_days)
            
            # Check if we have enough history
            if (end_date - train_start).days > 365:  # Max 1 year back
                break
            
            windows.insert(0, (train_start, train_end, test_start, test_end))
            
            # Step backward
            current_end = current_end - timedelta(days=self.step_days)
            
            if len(windows) >= 20:  # Max 20 windows
                break
        
        return windows
    
    def _test_window(
        self,
        symbols: List[str],
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime
    ) -> Dict:
        """
        Test strategy on a single window.
        
        Returns:
            Results dictionary for this window
        """
        try:
            # Fetch data for training period
            train_data = self._fetch_historical_data(
                symbols, train_start, train_end
            )
            
            # Optimize strategy on training data (optional)
            # best_params = self._optimize_on_train(train_data)
            
            # Fetch data for testing period
            test_data = self._fetch_historical_data(
                symbols, test_start, test_end
            )
            
            # Backtest on test period
            trades = self._backtest_on_test(test_data)
            
            # Calculate metrics
            metrics = self._calculate_metrics(trades)
            
            return {
                "train_period": {
                    "start": train_start.isoformat(),
                    "end": train_end.isoformat()
                },
                "test_period": {
                    "start": test_start.isoformat(),
                    "end": test_end.isoformat()
                },
                "trades": len(trades),
                "metrics": metrics,
            }
            
        except Exception as e:
            logger.error(f"❌ Window test failed: {e}")
            return {"error": str(e)}
    
    def _fetch_historical_data(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical kline data for symbols."""
        data = {}
        
        for symbol in symbols:
            try:
                # Calculate required klines
                days = (end - start).days
                limit = min(1000, days * 24)  # Assuming 1h candles
                
                klines = self.session.get_klines(
                    symbol=symbol,
                    interval="1h",
                    limit=limit
                )
                
                if not klines:
                    continue
                
                df = pd.DataFrame(klines, columns=[
                    "open_time", "open", "high", "low", "close", "volume", "turnover"
                ])
                
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df["timestamp"] = pd.to_datetime(df["open_time"], unit='ms')
                
                # Filter by date range
                df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
                
                data[symbol] = df
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch data for {symbol}: {e}")
                continue
        
        logger.info(f"📊 Fetched data for {len(data)} symbols")
        return data
    
    def _backtest_on_test(self, test_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Run backtest on test data.
        
        Returns:
            List of simulated trades
        """
        trades = []
        
        for symbol, df in test_data.items():
            try:
                # Calculate indicators
                df = self.strategy.calculate_indicators(df)
                
                # Generate signals
                for idx in range(len(df) - 1):
                    row = df.iloc[idx]
                    
                    # Simple simulation: check if signal would be generated
                    signal = self._simulate_signal_check(symbol, df.iloc[:idx+1])
                    
                    if signal:
                        # Simulate trade execution
                        entry_price = df.iloc[idx + 1]["close"]
                        
                        # Find exit (simplified: next 24 candles or SL/TP hit)
                        exit_idx, exit_price, exit_reason = self._simulate_exit(
                            df.iloc[idx+1:idx+25],
                            entry_price,
                            signal
                        )
                        
                        if exit_idx is not None:
                            trade = {
                                "symbol": symbol,
                                "entry_time": df.iloc[idx + 1]["timestamp"],
                                "exit_time": df.iloc[idx + 1 + exit_idx]["timestamp"],
                                "entry_price": entry_price,
                                "exit_price": exit_price,
                                "side": signal["side"],
                                "pnl_pct": self._calc_pnl_pct(
                                    signal["side"], entry_price, exit_price
                                ),
                                "exit_reason": exit_reason
                            }
                            trades.append(trade)
                
            except Exception as e:
                logger.warning(f"⚠️ Backtest failed for {symbol}: {e}")
                continue
        
        return trades
    
    def _simulate_signal_check(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> Optional[Dict]:
        """Check if strategy would generate signal at this point."""
        try:
            if len(df) < 50:
                return None
            
            # Use strategy's analyze_symbol method
            signal = self.strategy.analyze_symbol(symbol, df)
            
            return signal
            
        except Exception as e:
            return None
    
    def _simulate_exit(
        self,
        future_data: pd.DataFrame,
        entry_price: float,
        signal: Dict
    ) -> Tuple[Optional[int], Optional[float], str]:
        """
        Simulate exit conditions.
        
        Returns:
            (exit_index, exit_price, exit_reason) tuple
        """
        sl = signal.get("stop_loss", entry_price * 0.95)
        tp = signal.get("take_profit", entry_price * 1.10)
        side = signal.get("side", "BUY").upper()
        
        for idx, row in future_data.iterrows():
            if side == "BUY":
                if row["low"] <= sl:
                    return idx, sl, "SL"
                if row["high"] >= tp:
                    return idx, tp, "TP"
            else:  # SELL
                if row["high"] >= sl:
                    return idx, sl, "SL"
                if row["low"] <= tp:
                    return idx, tp, "TP"
        
        # Exit at last candle (time stop)
        return len(future_data) - 1, future_data.iloc[-1]["close"], "Time"
    
    def _calc_pnl_pct(self, side: str, entry: float, exit: float) -> float:
        """Calculate P&L percentage."""
        if side == "BUY":
            return ((exit - entry) / entry) * 100
        else:
            return ((entry - exit) / entry) * 100
    
    def _calculate_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate performance metrics from trades."""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_pnl_pct": 0,
                "sharpe": 0,
            }
        
        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        # Calculate metrics
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        sharpe = (avg_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0
        
        max_dd = self._calculate_max_drawdown(pnls)
        
        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 2),
            "avg_pnl_pct": round(avg_pnl, 3),
            "avg_win_pct": round(np.mean(wins), 3) if wins else 0,
            "avg_loss_pct": round(np.mean(losses), 3) if losses else 0,
            "sharpe": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses else 0,
        }
    
    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """Calculate maximum drawdown from P&L series."""
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max)
        return abs(drawdown.min()) if len(drawdown) > 0 else 0
    
    def _aggregate_results(self, all_results: List[Dict]) -> Dict:
        """Aggregate results across all windows."""
        valid_results = [r for r in all_results if "error" not in r]
        
        if not valid_results:
            return {"error": "No valid results"}
        
        # Extract metrics
        all_metrics = [r["metrics"] for r in valid_results]
        
        # Aggregate
        summary = {
            "total_windows": len(all_results),
            "valid_windows": len(valid_results),
            "avg_win_rate": np.mean([m["win_rate"] for m in all_metrics]),
            "avg_sharpe": np.mean([m["sharpe"] for m in all_metrics]),
            "avg_trades_per_window": np.mean([m["total_trades"] for m in all_metrics]),
            "avg_max_drawdown": np.mean([m["max_drawdown_pct"] for m in all_metrics]),
            "best_window": max(all_metrics, key=lambda x: x["sharpe"]),
            "worst_window": min(all_metrics, key=lambda x: x["sharpe"]),
            "consistency": self._calculate_consistency(all_metrics),
        }
        
        return summary
    
    def _calculate_consistency(self, metrics: List[Dict]) -> float:
        """Calculate strategy consistency (% of profitable windows)."""
        profitable = sum(1 for m in metrics if m["avg_pnl_pct"] > 0)
        return (profitable / len(metrics)) * 100 if metrics else 0
    
    def _save_results(self, all_results: List[Dict], summary: Dict):
        """Save results to JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detail_file = self.results_dir / f"wf_detailed_{timestamp}.json"
        with open(detail_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Save summary
        summary_file = self.results_dir / f"wf_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {detail_file} and {summary_file}")
