# ml_system/ml_monitor.py
"""
🎯 Professional ML Performance Monitor
Advanced analytics for ML trading system performance with drift detection and market regime analysis.
Production-ready for live trading environments.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import warnings
warnings.filterwarnings('ignore')

class MLPerformanceMonitor:
    """Professional monitor and analyzer for ML trading performance"""
    
    def __init__(self, data_dir: str = "ml_data"):
        self.data_dir = Path(data_dir)
        self.training_data_path = self.data_dir / "training_data.csv"
        
        # Create directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Color scheme for consistent visualization
        self.colors = {
            'profit': '#2ecc71',  # Green
            'loss': '#e74c3c',    # Red
            'neutral': '#3498db', # Blue
            'background': '#f8f9fa',
            'text': '#2c3e50'
        }
        
        logger.info(f"📊 ML Performance Monitor initialized with data directory: {self.data_dir}")
    
    def load_data(self) -> Optional[pd.DataFrame]:
        """Load and validate training data with comprehensive error handling"""
        try:
            if not self.training_data_path.exists():
                logger.warning("⚠️ No training data found at path")
                return None
            
            df = pd.read_csv(self.training_data_path)
            
            # Validate required columns
            required_cols = ['timestamp', 'pnl', 'symbol']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"❌ Missing required columns: {missing_cols}")
                return None
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Remove rows with NaN timestamps or PnL
            initial_len = len(df)
            df = df.dropna(subset=['timestamp', 'pnl'])
            if len(df) < initial_len:
                logger.warning(f"⚠️ Removed {initial_len - len(df)} rows with NaN timestamps or PnL")
            
            # Validate data range
            if len(df) == 0:
                logger.error("❌ No valid data after cleaning")
                return None
            
            logger.success(f"✅ Loaded {len(df)} trades from {self.training_data_path}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            return None
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics with risk-adjusted ratios"""
        try:
            if len(df) == 0:
                return self._get_empty_metrics()
            
            # Basic metrics
            total_trades = len(df)
            profitable_trades = len(df[df['pnl'] > 0])
            losing_trades = len(df[df['pnl'] < 0])
            win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0
            total_pnl = df['pnl'].sum()
            avg_pnl = df['pnl'].mean()
            
            # Win/Loss metrics
            wins = df[df['pnl'] > 0]['pnl']
            losses = df[df['pnl'] < 0]['pnl']
            
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = losses.mean() if len(losses) > 0 else 0
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            # Risk metrics
            max_drawdown = self._calculate_max_drawdown(df)
            calmar_ratio = total_pnl / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Volatility-adjusted metrics
            if len(df) > 1:
                returns_std = df['pnl'].std()
                sharpe_ratio = avg_pnl / returns_std if returns_std > 0 else 0
                
                # Sortino ratio (only penalizes downside volatility)
                downside_returns = df[df['pnl'] < 0]['pnl']
                downside_std = downside_returns.std() if len(downside_returns) > 0 else returns_std
                sortino_ratio = avg_pnl / downside_std if downside_std > 0 else 0
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
            
            # Profit factor
            gross_profit = wins.sum() if len(wins) > 0 else 0
            gross_loss = abs(losses.sum()) if len(losses) > 0 else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Expectancy
            expectancy = (win_rate/100 * avg_win) - ((1-win_rate/100) * abs(avg_loss)) if total_trades > 0 else 0
            
            # SL/TP metrics (if available)
            avg_sl_pct = df['sl_pct_used'].mean() if 'sl_pct_used' in df.columns and not df['sl_pct_used'].isna().all() else 0
            avg_tp_pct = df['tp_pct_used'].mean() if 'tp_pct_used' in df.columns and not df['tp_pct_used'].isna().all() else 0
            avg_rr_ratio = (df['tp_pct_used'] / df['sl_pct_used']).mean() if 'tp_pct_used' in df.columns and 'sl_pct_used' in df.columns else 0
            
            metrics = {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'win_loss_ratio': win_loss_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'profit_factor': profit_factor,
                'expectancy': expectancy,
                'avg_sl_pct': avg_sl_pct,
                'avg_tp_pct': avg_tp_pct,
                'avg_rr_ratio': avg_rr_ratio,
                'data_start': df['timestamp'].min(),
                'data_end': df['timestamp'].max(),
                'trading_days': (df['timestamp'].max() - df['timestamp'].min()).days if len(df) > 1 else 1,
            }
            
            # Add derived metrics
            metrics['pnl_per_trade'] = total_pnl / total_trades if total_trades > 0 else 0
            metrics['pnl_per_day'] = total_pnl / metrics['trading_days'] if metrics['trading_days'] > 0 else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate metrics: {e}")
            return self._get_empty_metrics()
    
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            'total_trades': 0,
            'profitable_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_pnl': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_loss_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'avg_sl_pct': 0.0,
            'avg_tp_pct': 0.0,
            'avg_rr_ratio': 0.0,
            'data_start': None,
            'data_end': None,
            'trading_days': 0,
            'pnl_per_trade': 0.0,
            'pnl_per_day': 0.0,
        }
    
    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown from cumulative PnL"""
        try:
            if len(df) == 0:
                return 0.0
            
            df_sorted = df.sort_values('timestamp').copy()
            df_sorted['cumulative_pnl'] = df_sorted['pnl'].cumsum()
            
            running_max = df_sorted['cumulative_pnl'].cummax()
            drawdown = df_sorted['cumulative_pnl'] - running_max
            max_drawdown = drawdown.min()
            
            return float(max_drawdown)
        except Exception:
            return 0.0
    
    def compare_by_symbol(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compare performance across symbols with statistical significance"""
        try:
            if len(df) == 0 or 'symbol' not in df.columns:
                return pd.DataFrame()
            
            symbol_stats = []
            
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol]
                
                if len(symbol_df) < 5:  # Skip symbols with too few trades
                    continue
                
                metrics = self.calculate_metrics(symbol_df)
                metrics['symbol'] = symbol
                metrics['trades_count'] = len(symbol_df)
                
                # Add statistical significance measures
                if len(symbol_df) >= 10:
                    # T-test against zero (is performance significantly different from random?)
                    from scipy import stats
                    t_stat, p_value = stats.ttest_1samp(symbol_df['pnl'], 0)
                    metrics['p_value'] = p_value
                    metrics['statistically_significant'] = p_value < 0.05
                else:
                    metrics['p_value'] = 1.0
                    metrics['statistically_significant'] = False
                
                symbol_stats.append(metrics)
            
            if not symbol_stats:
                return pd.DataFrame()
            
            result_df = pd.DataFrame(symbol_stats)
            
            # Calculate relative performance
            if len(result_df) > 0:
                avg_pnl_per_trade = result_df['pnl_per_trade'].mean()
                result_df['relative_performance'] = result_df['pnl_per_trade'] / avg_pnl_per_trade if avg_pnl_per_trade != 0 else 1.0
            
            return result_df.sort_values('total_pnl', ascending=False)
            
        except Exception as e:
            logger.error(f"❌ Failed to compare by symbol: {e}")
            return pd.DataFrame()
    
    def analyze_sl_tp_effectiveness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze if SL/TP levels are well-calibrated with advanced statistics"""
        try:
            if len(df) == 0:
                return {}
            
            analysis = {}
            
            # Winning trades analysis
            winners = df[df['pnl'] > 0]
            if len(winners) > 0:
                analysis['winners'] = {
                    'count': len(winners),
                    'avg_tp_used': winners['tp_pct_used'].mean() if 'tp_pct_used' in winners.columns else 0,
                    'tp_hit_rate': len(winners) / len(df) * 100,
                    'avg_pnl': winners['pnl'].mean(),
                    'median_pnl': winners['pnl'].median(),
                }
            
            # Losing trades analysis
            losers = df[df['pnl'] < 0]
            if len(losers) > 0:
                analysis['losers'] = {
                    'count': len(losers),
                    'avg_sl_used': losers['sl_pct_used'].mean() if 'sl_pct_used' in losers.columns else 0,
                    'sl_hit_rate': len(losers) / len(df) * 100,
                    'avg_pnl': losers['pnl'].mean(),
                    'median_pnl': losers['pnl'].median(),
                }
            
            # SL/TP range analysis
            if 'sl_pct_used' in df.columns and not df['sl_pct_used'].isna().all():
                analysis['sl_range'] = {
                    'min': df['sl_pct_used'].min(),
                    'max': df['sl_pct_used'].max(),
                    'mean': df['sl_pct_used'].mean(),
                    'std': df['sl_pct_used'].std(),
                    'q25': df['sl_pct_used'].quantile(0.25),
                    'q75': df['sl_pct_used'].quantile(0.75),
                }
            
            if 'tp_pct_used' in df.columns and not df['tp_pct_used'].isna().all():
                analysis['tp_range'] = {
                    'min': df['tp_pct_used'].min(),
                    'max': df['tp_pct_used'].max(),
                    'mean': df['tp_pct_used'].mean(),
                    'std': df['tp_pct_used'].std(),
                    'q25': df['tp_pct_used'].quantile(0.25),
                    'q75': df['tp_pct_used'].quantile(0.75),
                }
            
            # Risk/Reward analysis
            if 'sl_pct_used' in df.columns and 'tp_pct_used' in df.columns:
                valid_rr = df[(df['sl_pct_used'] > 0) & (df['tp_pct_used'] > 0)]
                if len(valid_rr) > 0:
                    rr_ratios = valid_rr['tp_pct_used'] / valid_rr['sl_pct_used']
                    analysis['risk_reward'] = {
                        'mean': rr_ratios.mean(),
                        'median': rr_ratios.median(),
                        'std': rr_ratios.std(),
                        'optimal_rr': self._find_optimal_rr(df),
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze SL/TP effectiveness: {e}")
            return {}
    
    def _find_optimal_rr(self, df: pd.DataFrame) -> float:
        """Find optimal risk/reward ratio based on historical data"""
        try:
            if len(df) < 10:
                return 2.0
            
            # Test different RR ratios
            rr_test_values = np.arange(1.0, 5.1, 0.5)
            best_rr = 2.0
            best_score = float('-inf')
            
            for rr in rr_test_values:
                # Simulate using this RR ratio
                simulated_pnl = []
                for _, row in df.iterrows():
                    if 'sl_pct_used' in row and 'tp_pct_used' in row and row['sl_pct_used'] > 0:
                        # Calculate what PnL would be with this RR
                        simulated_tp = row['sl_pct_used'] * rr
                        if row['pnl'] > 0:  # Winner
                            simulated_pnl.append(simulated_tp)
                        else:  # Loser
                            simulated_pnl.append(-row['sl_pct_used'])
                
                if len(simulated_pnl) > 0:
                    simulated_df = pd.DataFrame({'pnl': simulated_pnl})
                    metrics = self.calculate_metrics(simulated_df)
                    # Score based on profit factor and win rate
                    score = metrics['profit_factor'] * (metrics['win_rate'] / 100)
                    if score > best_score:
                        best_score = score
                        best_rr = rr
            
            return best_rr
            
        except Exception:
            return 2.0
    
    def feature_correlation_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze which features correlate most with profitability"""
        try:
            if len(df) == 0:
                return pd.DataFrame()
            
            # Define potential feature columns
            feature_cols = [
                'rsi', 'atr', 'atr_pct', 'trend_strength', 'volume_ratio',
                'ema_distance', 'momentum_24h', 'volatility', 'signal_strength',
                'bb_position', 'sr_position_in_range', 'mtf_trend_alignment',
                'candle_body_ratio', 'volume_zscore', 'macd_value'
            ]
            
            # Filter existing columns
            existing_features = [col for col in feature_cols if col in df.columns]
            
            if len(existing_features) == 0:
                logger.warning("⚠️ No feature columns found for correlation analysis")
                return pd.DataFrame()
            
            # Add binary outcome
            df['is_profitable'] = (df['pnl'] > 0).astype(int)
            
            # Calculate correlations
            correlations = []
            for feat in existing_features:
                # Remove NaN values for this feature
                clean_df = df[[feat, 'is_profitable']].dropna()
                if len(clean_df) < 10:
                    continue
                
                corr = clean_df[feat].corr(clean_df['is_profitable'])
                # Calculate p-value for statistical significance
                from scipy.stats import pearsonr
                try:
                    _, p_value = pearsonr(clean_df[feat], clean_df['is_profitable'])
                except:
                    p_value = 1.0
                
                correlations.append({
                    'Feature': feat,
                    'Correlation': corr,
                    'Absolute_Correlation': abs(corr),
                    'P_Value': p_value,
                    'Statistically_Significant': p_value < 0.05,
                    'Sample_Size': len(clean_df)
                })
            
            if not correlations:
                return pd.DataFrame()
            
            corr_df = pd.DataFrame(correlations)
            return corr_df.sort_values('Absolute_Correlation', ascending=False)
            
        except Exception as e:
            logger.error(f"❌ Failed feature correlation analysis: {e}")
            return pd.DataFrame()
    
    def detect_performance_drift(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect if model performance is drifting over time"""
        try:
            if len(df) < 20:
                return {"status": "insufficient_data"}
            
            # Split data into recent (last 30%) and historical (first 70%)
            df_sorted = df.sort_values('timestamp').reset_index(drop=True)
            split_point = int(len(df_sorted) * 0.7)
            
            historical = df_sorted.iloc[:split_point]
            recent = df_sorted.iloc[split_point:]
            
            hist_metrics = self.calculate_metrics(historical)
            recent_metrics = self.calculate_metrics(recent)
            
            # Compare key metrics
            drift_detected = False
            drift_details = {}
            
            # Win rate drift
            win_rate_change = recent_metrics['win_rate'] - hist_metrics['win_rate']
            if abs(win_rate_change) > 10:  # More than 10% change
                drift_detected = True
                drift_details['win_rate_drift'] = win_rate_change
            
            # Profit factor drift
            pf_change = recent_metrics['profit_factor'] - hist_metrics['profit_factor']
            if abs(pf_change) > 0.5:
                drift_detected = True
                drift_details['profit_factor_drift'] = pf_change
            
            # PnL per trade drift
            pnl_change = recent_metrics['pnl_per_trade'] - hist_metrics['pnl_per_trade']
            if abs(pnl_change) > abs(hist_metrics['pnl_per_trade']) * 0.5:
                drift_detected = True
                drift_details['pnl_per_trade_drift'] = pnl_change
            
            return {
                'drift_detected': drift_detected,
                'drift_details': drift_details,
                'historical_period': f"{historical['timestamp'].min()} to {historical['timestamp'].max()}",
                'recent_period': f"{recent['timestamp'].min()} to {recent['timestamp'].max()}",
                'historical_metrics': hist_metrics,
                'recent_metrics': recent_metrics,
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to detect performance drift: {e}")
            return {"status": "error", "message": str(e)}
    
    def analyze_by_market_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by market regime (trending vs ranging)"""
        try:
            if len(df) == 0:
                return {}
            
            # Create market regime based on volatility and trend
            if 'volatility' in df.columns and 'trend_strength' in df.columns:
                df_regime = df.copy()
                df_regime['volatility_quintile'] = pd.qcut(df_regime['volatility'], 5, labels=False)
                df_regime['trend_quintile'] = pd.qcut(abs(df_regime['trend_strength']), 5, labels=False)
                
                # Define regimes
                df_regime['regime'] = 'RANGING_LOW_VOL'
                df_regime.loc[(df_regime['trend_quintile'] >= 3) & (df_regime['volatility_quintile'] < 3), 'regime'] = 'TRENDING_LOW_VOL'
                df_regime.loc[(df_regime['trend_quintile'] < 3) & (df_regime['volatility_quintile'] >= 3), 'regime'] = 'RANGING_HIGH_VOL'
                df_regime.loc[(df_regime['trend_quintile'] >= 3) & (df_regime['volatility_quintile'] >= 3), 'regime'] = 'TRENDING_HIGH_VOL'
                
                # Analyze by regime
                regime_analysis = {}
                for regime in df_regime['regime'].unique():
                    regime_df = df_regime[df_regime['regime'] == regime]
                    if len(regime_df) >= 5:
                        regime_analysis[regime] = self.calculate_metrics(regime_df)
                        regime_analysis[regime]['sample_size'] = len(regime_df)
                
                return regime_analysis
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze by market regime: {e}")
            return {}
    
    def generate_report(self) -> str:
        """Generate comprehensive professional performance report"""
        df = self.load_data()
        if df is None or len(df) == 0:
            report = "❌ NO DATA AVAILABLE FOR ANALYSIS"
            print(report)
            return report
        
        # Build report
        report_lines = [
            "=" * 80,
            "🤖 PROFESSIONAL ML TRADING PERFORMANCE REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data Period: {df['timestamp'].min()} to {df['timestamp'].max()}",
            f"Total Trading Days: {(df['timestamp'].max() - df['timestamp'].min()).days}",
            "=" * 80,
        ]
        
        # Overall metrics
        metrics = self.calculate_metrics(df)
        report_lines.extend([
            "\n📊 COMPREHENSIVE PERFORMANCE METRICS",
            "-" * 80,
            f"Total Trades:      {metrics['total_trades']:>8d}",
            f"Win Rate:          {metrics['win_rate']:>8.2f}%",
            f"Total P&L:         ${metrics['total_pnl']:>8.2f}",
            f"P&L Per Trade:     ${metrics['pnl_per_trade']:>8.2f}",
            f"P&L Per Day:       ${metrics['pnl_per_day']:>8.2f}",
            f"Profit Factor:     {metrics['profit_factor']:>8.2f}",
            f"Expectancy:        ${metrics['expectancy']:>8.2f}",
            f"Sharpe Ratio:      {metrics['sharpe_ratio']:>8.2f}",
            f"Sortino Ratio:     {metrics['sortino_ratio']:>8.2f}",
            f"Calmar Ratio:      {metrics['calmar_ratio']:>8.2f}",
            f"Max Drawdown:      ${metrics['max_drawdown']:>8.2f}",
            f"Avg Win/Loss:      ${metrics['avg_win']:>6.2f}/${abs(metrics['avg_loss']):<6.2f}",
            f"Win/Loss Ratio:    {metrics['win_loss_ratio']:>8.2f}",
        ])
        
        # Symbol breakdown
        symbol_df = self.compare_by_symbol(df)
        if not symbol_df.empty:
            report_lines.extend([
                "\n📈 PER-SYMBOL PERFORMANCE (Top 10)",
                "-" * 80,
                f"{'Symbol':<12} {'Trades':>6} {'WinRate':>7} {'TotalPnL':>10} {'PnL/Trade':>9} {'Significant':>12}"
            ])
            
            for _, row in symbol_df.head(10).iterrows():
                significant = "✓" if row.get('statistically_significant', False) else "✗"
                report_lines.append(
                    f"{row['symbol']:<12} {int(row['trades_count']):>6d} "
                    f"{row['win_rate']:>6.1f}% ${row['total_pnl']:>9.2f} "
                    f"${row['pnl_per_trade']:>8.2f} {significant:>12}"
                )
        
        # SL/TP analysis
        sl_tp_analysis = self.analyze_sl_tp_effectiveness(df)
        if sl_tp_analysis:
            report_lines.extend([
                "\n🎯 SL/TP CALIBRATION ANALYSIS",
                "-" * 80,
            ])
            
            if 'sl_range' in sl_tp_analysis:
                sl = sl_tp_analysis['sl_range']
                report_lines.append(
                    f"SL Range: {sl['min']:.2f}% - {sl['max']:.2f}% "
                    f"(Mean: {sl['mean']:.2f}%, Std: {sl['std']:.2f}%)"
                )
            
            if 'tp_range' in sl_tp_analysis:
                tp = sl_tp_analysis['tp_range']
                report_lines.append(
                    f"TP Range: {tp['min']:.2f}% - {tp['max']:.2f}% "
                    f"(Mean: {tp['mean']:.2f}%, Std: {tp['std']:.2f}%)"
                )
            
            if 'risk_reward' in sl_tp_analysis:
                rr = sl_tp_analysis['risk_reward']
                report_lines.append(
                    f"Optimal R/R: {rr['optimal_rr']:.2f} (Current Avg: {rr['mean']:.2f})"
                )
        
        # Performance drift detection
        drift_analysis = self.detect_performance_drift(df)
        if drift_analysis.get('drift_detected', False):
            report_lines.extend([
                "\n⚠️ PERFORMANCE DRIFT DETECTED",
                "-" * 80,
                f"Recent Period: {drift_analysis['recent_period']}",
                f"Win Rate Change: {drift_analysis['drift_details'].get('win_rate_drift', 0):+.2f}%",
                f"Profit Factor Change: {drift_analysis['drift_details'].get('profit_factor_drift', 0):+.2f}",
                f"PnL/Trade Change: ${drift_analysis['drift_details'].get('pnl_per_trade_drift', 0):+.2f}",
                "→ Consider model retraining or adjustment",
            ])
        
        # Market regime analysis
        regime_analysis = self.analyze_by_market_regime(df)
        if regime_analysis:
            report_lines.extend([
                "\n🌍 PERFORMANCE BY MARKET REGIME",
                "-" * 80,
            ])
            
            for regime, reg_metrics in regime_analysis.items():
                report_lines.append(
                    f"{regime:<20} | Trades: {reg_metrics['sample_size']:>3d} | "
                    f"Win Rate: {reg_metrics['win_rate']:>5.1f}% | "
                    f"PnL/Trade: ${reg_metrics['pnl_per_trade']:>7.2f}"
                )
        
        # Feature importance
        corr_df = self.feature_correlation_analysis(df)
        if not corr_df.empty:
            report_lines.extend([
                "\n🔍 FEATURE CORRELATIONS WITH PROFITABILITY",
                "-" * 80,
                f"{'Feature':<25} {'Correlation':>10} {'P-Value':>8} {'Significant':>12}"
            ])
            
            for _, row in corr_df.head(10).iterrows():
                significant = "✓" if row['Statistically_Significant'] else "✗"
                sign = "+" if row['Correlation'] > 0 else "-"
                bar_length = int(abs(row['Correlation']) * 20)
                bar = "█" * bar_length
                report_lines.append(
                    f"{row['Feature']:<25} {sign}{abs(row['Correlation']):>9.3f} "
                    f"{row['P_Value']:>8.3f} {significant:>12}"
                )
        
        # Time-based analysis
        report_lines.extend([
            "\n📅 RECENT PERFORMANCE",
            "-" * 80,
        ])
        
        # Last 7 days
        last_week = df[df['timestamp'] >= datetime.now() - timedelta(days=7)]
        if len(last_week) > 0:
            week_metrics = self.calculate_metrics(last_week)
            report_lines.append(
                f"Last 7 Days:  {len(last_week):>3d} trades, "
                f"{week_metrics['win_rate']:>5.1f}% win rate, "
                f"${week_metrics['total_pnl']:>8.2f} P&L"
            )
        
        # Last 24 hours
        last_day = df[df['timestamp'] >= datetime.now() - timedelta(days=1)]
        if len(last_day) > 0:
            day_metrics = self.calculate_metrics(last_day)
            report_lines.append(
                f"Last 24 Hours: {len(last_day):>3d} trades, "
                f"{day_metrics['win_rate']:>5.1f}% win rate, "
                f"${day_metrics['total_pnl']:>8.2f} P&L"
            )
        
        report_lines.append("\n" + "=" * 80)
        full_report = "\n".join(report_lines)
        
        print(full_report)
        return full_report
    
    def plot_performance(self, save_path: str = "ml_performance.png") -> bool:
        """Generate professional performance visualization"""
        df = self.load_data()
        if df is None or len(df) == 0:
            return False
        
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            fig = plt.figure(figsize=(20, 15))
            fig.patch.set_facecolor(self.colors['background'])
            
            # Create subplots
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Cumulative P&L with drawdown
            ax1 = fig.add_subplot(gs[0, :])
            self._plot_cumulative_pnl(ax1, df)
            
            # 2. Monthly Returns Heatmap
            ax2 = fig.add_subplot(gs[1, 0])
            self._plot_monthly_returns_heatmap(ax2, df)
            
            # 3. Win/Loss Distribution
            ax3 = fig.add_subplot(gs[1, 1])
            self._plot_pnl_distribution(ax3, df)
            
            # 4. Win Rate by Hour
            ax4 = fig.add_subplot(gs[1, 2])
            self._plot_win_rate_by_hour(ax4, df)
            
            # 5. SL vs TP Scatter
            ax5 = fig.add_subplot(gs[2, 0])
            self._plot_sl_tp_scatter(ax5, df)
            
            # 6. Performance by Symbol
            ax6 = fig.add_subplot(gs[2, 1])
            self._plot_performance_by_symbol(ax6, df)
            
            # 7. Feature Importance
            ax7 = fig.add_subplot(gs[2, 2])
            self._plot_feature_importance(ax7, df)
            
            # Title and layout
            fig.suptitle('🤖 PROFESSIONAL ML TRADING PERFORMANCE DASHBOARD', 
                        fontsize=20, fontweight='bold', color=self.colors['text'])
            
            plt.tight_layout()
            
            # Save with high quality
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=self.colors['background'])
            plt.close()
            
            logger.success(f"📊 Professional performance chart saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Plotting failed: {e}")
            return False
    
    def _plot_cumulative_pnl(self, ax, df):
        """Plot cumulative P&L with drawdown"""
        df_sorted = df.sort_values('timestamp').copy()
        df_sorted['cumulative_pnl'] = df_sorted['pnl'].cumsum()
        
        # Calculate running max and drawdown
        running_max = df_sorted['cumulative_pnl'].cummax()
        drawdown = df_sorted['cumulative_pnl'] - running_max
        
        ax.plot(df_sorted['timestamp'], df_sorted['cumulative_pnl'], 
                linewidth=2, color=self.colors['profit'], label='Cumulative P&L')
        ax.fill_between(df_sorted['timestamp'], df_sorted['cumulative_pnl'], 
                       running_max, where=(drawdown < 0), 
                       color=self.colors['loss'], alpha=0.3, label='Drawdown')
        
        ax.set_title('Cumulative P&L with Drawdown', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative P&L ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_monthly_returns_heatmap(self, ax, df):
        """Plot monthly returns heatmap"""
        df_copy = df.copy()
        df_copy['month'] = df_copy['timestamp'].dt.month
        df_copy['year'] = df_copy['timestamp'].dt.year
        
        monthly_returns = df_copy.groupby(['year', 'month'])['pnl'].sum().unstack()
        
        if not monthly_returns.empty:
            sns.heatmap(monthly_returns, annot=True, fmt='.0f', cmap='RdYlGn', 
                       center=0, ax=ax, cbar_kws={'label': 'P&L ($)'})
            ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
    
    def _plot_pnl_distribution(self, ax, df):
        """Plot P&L distribution"""
        wins = df[df['pnl'] > 0]['pnl']
        losses = df[df['pnl'] < 0]['pnl']
        
        ax.hist([wins, losses], bins=20, label=['Wins', 'Losses'], 
                color=[self.colors['profit'], self.colors['loss']], alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.set_title('P&L Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('P&L ($)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_win_rate_by_hour(self, ax, df):
        """Plot win rate by hour of day"""
        df_copy = df.copy()
        df_copy['hour'] = df_copy['timestamp'].dt.hour
        hourly_data = df_copy.groupby('hour').agg({
            'pnl': lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0,
            'timestamp': 'count'
        }).rename(columns={'pnl': 'win_rate', 'timestamp': 'trade_count'})
        
        bars = ax.bar(hourly_data.index, hourly_data['win_rate'], 
                     color=[self.colors['profit'] if x >= 50 else self.colors['loss'] 
                           for x in hourly_data['win_rate']])
        
        # Add trade count annotations
        for i, (win_rate, count) in enumerate(zip(hourly_data['win_rate'], hourly_data['trade_count'])):
            ax.text(i, win_rate + 1, str(count), ha='center', va='bottom', fontsize=8)
        
        ax.set_title('Win Rate by Hour of Day\n(Number = Trade Count)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hour (UTC)', fontsize=12)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 2))
    
    def _plot_sl_tp_scatter(self, ax, df):
        """Plot SL vs TP scatter"""
        if 'sl_pct_used' in df.columns and 'tp_pct_used' in df.columns:
            scatter = ax.scatter(df['sl_pct_used'], df['tp_pct_used'], 
                               c=df['pnl'], cmap='RdYlGn', alpha=0.6, s=50)
            plt.colorbar(scatter, ax=ax, label='P&L ($)')
            ax.set_title('SL vs TP Usage\n(Color = P&L)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Stop Loss %', fontsize=12)
            ax.set_ylabel('Take Profit %', fontsize=12)
            ax.grid(True, alpha=0.3)
    
    def _plot_performance_by_symbol(self, ax, df):
        """Plot performance by symbol"""
        symbol_df = self.compare_by_symbol(df)
        if not symbol_df.empty:
            top_symbols = symbol_df.head(10)
            bars = ax.barh(range(len(top_symbols)), top_symbols['pnl_per_trade'],
                          color=[self.colors['profit'] if x > 0 else self.colors['loss'] 
                                for x in top_symbols['pnl_per_trade']])
            
            ax.set_title('P&L Per Trade by Symbol', fontsize=14, fontweight='bold')
            ax.set_xlabel('P&L Per Trade ($)', fontsize=12)
            ax.set_ylabel('Symbol', fontsize=12)
            ax.set_yticks(range(len(top_symbols)))
            ax.set_yticklabels(top_symbols['symbol'])
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    def _plot_feature_importance(self, ax, df):
        """Plot feature importance"""
        corr_df = self.feature_correlation_analysis(df)
        if not corr_df.empty:
            top_features = corr_df.head(10)
            colors = [self.colors['profit'] if x > 0 else self.colors['loss'] 
                     for x in top_features['Correlation']]
            
            bars = ax.barh(range(len(top_features)), top_features['Correlation'], color=colors)
            ax.set_title('Feature Correlation with Profitability', fontsize=14, fontweight='bold')
            ax.set_xlabel('Correlation Coefficient', fontsize=12)
            ax.set_ylabel('Feature', fontsize=12)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['Feature'])
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

def main():
    """Run ML performance analysis"""
    monitor = MLPerformanceMonitor()
    
    # Generate text report
    monitor.generate_report()
    
    # Generate visual report
    try:
        monitor.plot_performance()
    except ImportError:
        logger.info("ℹ️ Install matplotlib and seaborn for visualization: pip install matplotlib seaborn")


if __name__ == "__main__":
    main()