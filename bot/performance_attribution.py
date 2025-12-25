# performance_attribution.py
"""
Performance Attribution System - Elite Full Version
Analyzes WHERE profits/losses come from (strategy, timing, sizing, market, etc.)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger
from collections import defaultdict
import json
from pathlib import Path
import traceback

class PerformanceAttribution:
    """
    Comprehensive Performance Attribution System.
    Analyzes trading results across multiple dimensions: Strategy, Symbol, Time, and Execution.
    """
    
    def __init__(self, config_loader):
        """
        Initializes the attribution system and ensures persistence directory exists.
        Args:
            config_loader: ConfigLoader instance for accessing bot settings.
        """
        self.cfg = config_loader
        
        # Internal Storage (Memory)
        self.trades: List[Dict[str, Any]] = []
        
        # Persistence Settings
        self.results_dir = Path("performance_reports")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.success("📊 PerformanceAttribution system fully initialized")

    def record_trade(self, trade: Dict[str, Any]):
        """
        Records a completed trade for analysis and ensures data types are normalized.
        Args:
            trade: Dictionary containing trade metadata (symbol, pnl, fees, timestamps, etc.)
        """
        try:
            # 1. Normalize Timestamps (Crucial for time-based analysis)
            for key in ["opened_at", "closed_at"]:
                if key in trade and trade[key] is not None:
                    if isinstance(trade[key], (int, float)):
                        trade[key] = datetime.fromtimestamp(trade[key])
                    elif isinstance(trade[key], str):
                        try:
                            trade[key] = pd.to_datetime(trade[key])
                        except:
                            logger.error(f"Failed to parse timestamp {key}: {trade[key]}")

            if "closed_at" not in trade or trade["closed_at"] is None:
                trade["closed_at"] = datetime.now()
            
            # 2. Add to history
            self.trades.append(trade)
            
            # 3. Limit memory usage (Keep last 5,000 trades for statistical significance)
            if len(self.trades) > 2000:
                self.trades = self.trades[-2000:]
                
            logger.debug(f"Trade recorded for {trade.get('symbol', 'UNKNOWN')}. History size: {len(self.trades)}")
                
        except Exception as e:
            logger.error(f"❌ Failed to record trade: {str(e)}")

    def generate_attribution_report(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Main entry point for generating a full-scale performance report.
        """
        try:
            # Filter trades by date
            cutoff = datetime.now() - timedelta(days=lookback_days)
            recent_trades = [
                t for t in self.trades 
                if t.get("closed_at", datetime.now()) >= cutoff
            ]
            
            if not recent_trades:
                logger.warning(f"⚠️ No trades found in the last {lookback_days} days")
                return {"error": "No trades found in lookback period"}
            
            logger.info(f"📊 Generating Elite Attribution Report for {len(recent_trades)} trades...")
            
            # Construct the comprehensive report dictionary
            report = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "lookback_days": lookback_days,
                    "trade_count": len(recent_trades)
                },
                "overview": self._calculate_overview(recent_trades),
                "by_symbol": self._attribute_by_symbol(recent_trades),
                "by_strategy": self._attribute_by_strategy(recent_trades),
                "by_time": self._attribute_by_time(recent_trades),
                "by_market_condition": self._attribute_by_market(recent_trades),
                "by_trade_duration": self._attribute_by_duration(recent_trades),
                "sizing_analysis": self._analyze_sizing(recent_trades),
                "execution_analysis": self._analyze_execution(recent_trades),
            }
            
            # Save the report for persistence
            self._save_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Elite Attribution Report generation failed: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # INTERNAL ATTRIBUTION MODULES
    # ------------------------------------------------------------------
    def _calculate_overview(self, trades: List[Dict]) -> Dict:
        """Calculate high-level summary and risk-adjusted metrics."""
        pnls = np.array([float(t.get("net_pnl", 0)) for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        
        total_pnl = np.sum(pnls)
        win_rate = (len(wins) / len(pnls) * 100) if len(pnls) > 0 else 0
        
        # Risk Metrics (Standard Deviation, Sharpe, Expectancy)
        std_dev = np.std(pnls) if len(pnls) > 1 else 0
        sharpe = (np.mean(pnls) / std_dev) * np.sqrt(252) if std_dev > 0 else 0
        
        # Max Drawdown Analysis
        cum_pnl = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum_pnl)
        drawdowns = peak - cum_pnl
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

        return {
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(win_rate, 2),
            "avg_win": round(np.mean(wins), 2) if len(wins) > 0 else 0,
            "avg_loss": round(np.mean(losses), 2) if len(losses) > 0 else 0,
            "profit_factor": round(np.sum(wins) / abs(np.sum(losses)), 2) if len(losses) > 0 else 0,
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pnl": round(max_dd, 2),
            "expectancy": round(np.mean(pnls), 2) if len(pnls) > 0 else 0,
            "best_trade": round(np.max(pnls), 2) if len(pnls) > 0 else 0,
            "worst_trade": round(np.min(pnls), 2) if len(pnls) > 0 else 0
        }

    def _attribute_by_symbol(self, trades: List[Dict]) -> Dict:
        """Analyze performance per trading pair."""
        by_symbol = defaultdict(lambda: {"trades": 0, "pnl": 0, "wins": 0})
        
        for t in trades:
            symbol = t.get("symbol", "UNKNOWN")
            pnl = float(t.get("net_pnl", 0))
            
            by_symbol[symbol]["trades"] += 1
            by_symbol[symbol]["pnl"] += pnl
            if pnl > 0:
                by_symbol[symbol]["wins"] += 1
        
        processed_symbols = {}
        for sym, data in by_symbol.items():
            processed_symbols[sym] = {
                "trades": data["trades"],
                "pnl": round(data["pnl"], 2),
                "win_rate": round((data["wins"] / data["trades"]) * 100, 2)
            }
        
        # Sorting
        sorted_syms = dict(sorted(processed_symbols.items(), key=lambda x: x[1]["pnl"], reverse=True))
        
        return {
            "top_performers": dict(list(sorted_syms.items())[:5]),
            "worst_performers": dict(list(sorted_syms.items())[-5:]),
            "full_list": sorted_syms
        }

    def _attribute_by_strategy(self, trades: List[Dict]) -> Dict:
        """Analyze results based on strategy name or signal source."""
        by_strategy = defaultdict(lambda: {"trades": 0, "pnl": 0, "wins": 0})
        
        for t in trades:
            # Check multiple potential keys for strategy identification
            strategy = t.get("strategy_name") or t.get("signal_type") or "REGULAR"
            pnl = float(t.get("net_pnl", 0))
            
            by_strategy[strategy]["trades"] += 1
            by_strategy[strategy]["pnl"] += pnl
            if pnl > 0:
                by_strategy[strategy]["wins"] += 1
        
        return {
            strat: {
                "trades": d["trades"],
                "pnl": round(d["pnl"], 2),
                "win_rate": round((d["wins"] / d["trades"]) * 100, 2)
            } for strat, d in by_strategy.items()
        }

    def _attribute_by_time(self, trades: List[Dict]) -> Dict:
        """Find the most profitable hours and days of the week."""
        by_hour = defaultdict(lambda: {"trades": 0, "pnl": 0})
        by_day = defaultdict(lambda: {"trades": 0, "pnl": 0})
        
        for t in trades:
            dt = t.get("closed_at")
            if not isinstance(dt, datetime): continue
            
            hour = dt.hour
            day = dt.strftime("%A")
            pnl = float(t.get("net_pnl", 0))
            
            by_hour[hour]["trades"] += 1
            by_hour[hour]["pnl"] += pnl
            by_day[day]["trades"] += 1
            by_day[day]["pnl"] += pnl
            
        return {
            "by_hour": {h: {"pnl": round(d["pnl"], 2), "trades": d["trades"]} for h, d in sorted(by_hour.items())},
            "by_day": {d: {"pnl": round(v["pnl"], 2), "trades": v["trades"]} for d, v in by_day.items()}
        }

    def _attribute_by_market(self, trades: List[Dict]) -> Dict:
        """Analyze performance against market trends recorded during entry."""
        by_trend = defaultdict(lambda: {"trades": 0, "pnl": 0})
        
        for t in trades:
            trend = t.get("trend") or t.get("market_condition") or "UNKNOWN"
            pnl = float(t.get("net_pnl", 0))
            
            by_trend[trend]["trades"] += 1
            by_trend[trend]["pnl"] += pnl
            
        return {k: {"pnl": round(v["pnl"], 2), "trades": v["trades"]} for k, v in by_trend.items()}

    def _attribute_by_duration(self, trades: List[Dict]) -> Dict:
        """Segment P&L by how long the trade remained open."""
        by_duration = {
            "< 1h": {"trades": 0, "pnl": 0},
            "1-4h": {"trades": 0, "pnl": 0},
            "4-12h": {"trades": 0, "pnl": 0},
            "12-24h": {"trades": 0, "pnl": 0},
            "> 24h": {"trades": 0, "pnl": 0},
        }
        
        for t in trades:
            opened, closed = t.get("opened_at"), t.get("closed_at")
            if not (isinstance(opened, datetime) and isinstance(closed, datetime)): continue
            
            duration_hrs = (closed - opened).total_seconds() / 3600
            pnl = float(t.get("net_pnl", 0))
            
            if duration_hrs < 1: bucket = "< 1h"
            elif duration_hrs < 4: bucket = "1-4h"
            elif duration_hrs < 12: bucket = "4-12h"
            elif duration_hrs < 24: bucket = "12-24h"
            else: bucket = "> 24h"
            
            by_duration[bucket]["trades"] += 1
            by_duration[bucket]["pnl"] += pnl
            
        return {k: {"pnl": round(v["pnl"], 2), "trades": v["trades"]} for k, v in by_duration.items()}

    def _analyze_sizing(self, trades: List[Dict]) -> Dict:
        """Correlation analysis between position size and success."""
        sizes = [float(t.get("position_size", 0)) for t in trades if t.get("position_size")]
        pnls = [float(t.get("net_pnl", 0)) for t in trades]
        
        if len(sizes) < 2: return {"status": "insufficient sizing data"}
        
        correlation = np.corrcoef(sizes, pnls)[0, 1]
        median_size = np.median(sizes)
        
        large_trades = [t for t in trades if float(t.get("position_size", 0)) > median_size]
        small_trades = [t for t in trades if float(t.get("position_size", 0)) <= median_size]
        
        return {
            "avg_position_size": round(np.mean(sizes), 2),
            "size_pnl_correlation": round(correlation, 3),
            "large_positions": {
                "trades": len(large_trades),
                "total_pnl": round(sum(t.get("net_pnl", 0) for t in large_trades), 2)
            },
            "small_positions": {
                "trades": len(small_trades),
                "total_pnl": round(sum(t.get("net_pnl", 0) for t in small_trades), 2)
            }
        }

    def _analyze_execution(self, trades: List[Dict]) -> Dict:
        """Analyze fees, slippage, and overall execution cost."""
        total_fees = sum(float(t.get("fees", 0)) for t in trades)
        total_net_pnl = sum(float(t.get("net_pnl", 0)) for t in trades)
        # Slippage can be derived or passed directly from the client wrapper
        total_slippage = sum(float(t.get("slippage", 0)) for t in trades)
        
        execution_drag = total_fees + total_slippage
        
        return {
            "total_fees": round(total_fees, 2),
            "total_slippage": round(total_slippage, 2),
            "total_execution_cost": round(execution_drag, 2),
            "drag_on_pnl_pct": round((abs(execution_drag) / abs(total_net_pnl)) * 100, 2) if total_net_pnl != 0 else 0
        }

    def _save_report(self, report: Dict):
        """Saves report as JSON and exports a comprehensive trade audit CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_path = self.results_dir / f"report_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save Audit CSV (Elite Feature)
        if self.trades:
            csv_path = self.results_dir / f"audit_log_{timestamp}.csv"
            pd.DataFrame(self.trades).to_csv(csv_path, index=False)
            
        logger.info(f"💾 Elite Attribution Report saved to {json_path.name}")

    def print_summary(self, report: Dict):
        """Displays high-level stats in the console."""
        ov = report.get("overview", {})
        print("\n" + "="*60)
        print(f" ELITE TRADING SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*60)
        print(f"Total P&L:       ${ov.get('total_pnl', 0):<10} | Win Rate:    {ov.get('win_rate', 0)}%")
        print(f"Sharpe Ratio:    {ov.get('sharpe_ratio', 0):<10} | Profit Factor: {ov.get('profit_factor', 0)}")
        print(f"Max Drawdown:    ${ov.get('max_drawdown_pnl', 0):<10} | Expectancy:   ${ov.get('expectancy', 0)}")
        
        # Quick view of strategy performance
        strat_perf = report.get("by_strategy", {})
        if strat_perf:
            print("-" * 60)
            print(" PERFORMANCE BY STRATEGY:")
            for s, d in strat_perf.items():
                print(f" > {s:<15}: ${d['pnl']:<8} (WR: {d['win_rate']}%)")
        print("="*60 + "\n")