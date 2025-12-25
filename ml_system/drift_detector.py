# ml_system/drift_detector.py
"""
Professional Data Drift Detector for Trading ML Systems
Detects statistical distribution shifts in feature space.
Uses Kolmogorov-Smirnov test + configurable thresholds.
Thread-safe and production-ready.
"""

import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
import numpy as np
from .config import ml_config


class DataDriftDetector:
    """
    Detect when data distribution changes significantly.
    Critical for model reliability in live trading.
    """

    def __init__(self, threshold: float = 0.05):
        """
        Initialize drift detector.
        
        Args:
            threshold: P-value threshold for drift detection (lower = more sensitive)
        """
        if not (0.0 < threshold <= 1.0):
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        self.threshold = threshold
        self.reference_data: Optional[pd.DataFrame] = None
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
    
    def fit(self, data: pd.DataFrame) -> bool:
        """
        Store reference distribution for future comparison.
        
        Args:
            data: Reference dataset (usually training set or recent stable period)
        
        Returns:
            bool: True if fitting succeeded
        """
        try:
            if not isinstance(data, pd.DataFrame) or data.empty:
                logger.error("❌ Invalid reference data provided")
                return False
            
            # Identify numeric columns only (KS test works on continuous distributions)
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) == 0:
                logger.error("❌ No numeric columns found in reference data")
                return False
            
            self.reference_data = data.copy()
            self.numeric_columns = numeric_cols
            self.categorical_columns = [
                col for col in data.columns if col not in numeric_cols
            ]
            
            logger.info(
                f"✅ Drift detector fitted with {len(numeric_cols)} numeric features: "
                f"{', '.join(numeric_cols[:3])}{'...' if len(numeric_cols) > 3 else ''}"
            )
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to fit drift detector: {e}")
            return False
    
    def detect_drift(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for drift in new data against reference distribution.
        
        Args:
            new_data: New data to compare against reference
        
        Returns:
            Dict with overall drift status, per-column results, and summary metrics
        """
        try:
            if self.reference_data is None:
                logger.error("❌ Drift detector not fitted - call fit() first")
                return {"error": "Not fitted", "drift_detected": False}
            
            if not isinstance(new_data, pd.DataFrame) or new_data.empty:
                logger.error("❌ Invalid new data provided")
                return {"error": "Invalid input", "drift_detected": False}
            
            # Ensure we have overlapping numeric columns
            common_numeric = list(set(self.numeric_columns) & set(new_data.columns))
            if len(common_numeric) == 0:
                logger.warning("⚠️ No overlapping numeric columns for drift detection")
                return {
                    "drift_detected": False,
                    "message": "No common numeric columns",
                    "tested_columns": [],
                    "drifts": {}
                }
            
            drifts = {}
            total_tests = 0
            drifted_columns = 0
            max_drift_severity = 0  # 0=none, 1=medium, 2=high
            
            for column in common_numeric:
                try:
                    ref_col = self.reference_data[column].dropna()
                    new_col = new_data[column].dropna()
                    
                    if len(ref_col) < 10 or len(new_col) < 10:
                        logger.debug(f"Skipping {column} - insufficient samples")
                        continue
                    
                    statistic, p_value = ks_2samp(ref_col, new_col)
                    total_tests += 1
                    
                    if p_value < self.threshold:
                        drifted_columns += 1
                        severity = 'high' if p_value < 0.01 else 'medium'
                        severity_level = 2 if p_value < 0.01 else 1
                        max_drift_severity = max(max_drift_severity, severity_level)
                        
                        drifts[column] = {
                            'statistic': float(statistic),
                            'p_value': float(p_value),
                            'severity': severity,
                            'sample_size_ref': len(ref_col),
                            'sample_size_new': len(new_col)
                        }
                        
                        logger.debug(
                            f"📊 Drift detected in {column}: p={p_value:.4f}, "
                            f"statistic={statistic:.4f}, severity={severity}"
                        )
                
                except Exception as e:
                    logger.warning(f"⚠️ Failed to test drift for column {column}: {e}")
                    continue
            
            # Calculate overall drift score (0-1)
            drift_score = drifted_columns / total_tests if total_tests > 0 else 0.0
            
            # Determine if action required based on config
            require_action = (
                drifted_columns > 0 and 
                drift_score >= getattr(ml_config, 'max_feature_drift_threshold', 0.2)
            )
            
            result = {
                "drift_detected": drifted_columns > 0,
                "require_retrain": require_action,
                "drift_score": drift_score,
                "total_columns_tested": total_tests,
                "drifted_columns_count": drifted_columns,
                "max_severity": ['none', 'medium', 'high'][max_drift_severity],
                "tested_columns": common_numeric,
                "drifts": drifts,
                "timestamp": pd.Timestamp.now().isoformat(),
                "threshold_used": self.threshold
            }
            
            if drifted_columns > 0:
                logger.warning(
                    f"⚠️ DRIFT DETECTED: {drifted_columns}/{total_tests} features drifted. "
                    f"Score: {drift_score:.2%}. Max severity: {result['max_severity']}"
                )
            else:
                logger.debug("✅ No significant drift detected")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Drift detection failed: {e}")
            return {
                "error": str(e),
                "drift_detected": False,
                "require_retrain": False,
                "timestamp": pd.Timestamp.now().isoformat()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current detector status"""
        return {
            "is_fitted": self.reference_data is not None,
            "numeric_columns": len(self.numeric_columns),
            "categorical_columns": len(self.categorical_columns),
            "threshold": self.threshold
        }
    
    def reset(self):
        """Reset detector state"""
        self.reference_data = None
        self.numeric_columns = []
        self.categorical_columns = []
        logger.info("🔄 Drift detector reset")