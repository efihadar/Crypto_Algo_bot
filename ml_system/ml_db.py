# ml_system/ml_db.py
"""
🗄️ Professional ML Database Manager
Handles all ML database operations with connection pooling, migrations, and comprehensive monitoring.
Production-ready for live trading environments.
"""

import os
import json
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import time
from loguru import logger

# Try to import PostgreSQL dependencies
PSYCOPG2_AVAILABLE = False
try:
    import psycopg2
    import psycopg2.pool
    import psycopg2.extras
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    PSYCOPG2_AVAILABLE = True
    logger.info("✅ psycopg2 loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ psycopg2 not available: {e}")

class MLDatabase:
    """
    Professional ML Database Manager with comprehensive error handling and monitoring
    
    Features:
    - Connection pooling with health checks
    - Automatic schema creation and migrations
    - Safe transaction management
    - Performance monitoring
    - Comprehensive error handling
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        database: str = None,
        user: str = None,
        password: str = None,
        min_conn: int = 2,
        max_conn: int = 10
    ):
        """Initialize database connection pool"""
        # Load from environment if not provided
        self.host = host or os.getenv("ML_DB_HOST", "localhost")
        self.port = port or int(os.getenv("ML_DB_PORT", "5432"))
        self.database = database or os.getenv("ML_DB_NAME", "ml_trading_db")
        self.user = user or os.getenv("ML_DB_USER", "ml_user")
        self.password = password or os.getenv("ML_DB_PASS", "")
        
        # Validate critical parameters
        if not self.password:
            logger.warning("⚠️ Database password not set - using empty password")
        
        self.pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self.min_conn = min_conn
        self.max_conn = max_conn
        self._is_initialized = False
        
        # Initialize if psycopg2 is available
        if PSYCOPG2_AVAILABLE:
            self._init_pool()
            if self.pool:
                self._init_schema()
                self._is_initialized = True
                logger.success(f"✅ ML Database initialized: {self.host}:{self.port}/{self.database}")
        else:
            logger.error("❌ Cannot initialize ML database - psycopg2 not available")
    
    def _init_pool(self) -> bool:
        """Initialize connection pool with comprehensive error handling"""
        try:
            # Test connection first
            test_conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                connect_timeout=5
            )
            test_conn.close()
            
            # Create connection pool
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.min_conn,
                maxconn=self.max_conn,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                connect_timeout=10,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5
            )
            
            logger.success(f"✅ Database connection pool initialized ({self.min_conn}-{self.max_conn} connections)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize connection pool: {e}")
            self.pool = None
            return False
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with comprehensive error handling"""
        conn = None
        start_time = time.time()
        try:
            if not self.pool:
                logger.error("❌ No connection pool available")
                yield None
                return
            
            conn = self.pool.getconn()
            
            # Set session parameters
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout = 30000")  # 30 seconds
                cur.execute("SET lock_timeout = 10000")       # 10 seconds
            
            yield conn
            
        except psycopg2.OperationalError as e:
            logger.error(f"❌ Database operational error: {e}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            yield None
        except Exception as e:
            logger.error(f"❌ Database connection error: {e}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            yield None
        finally:
            if conn and self.pool:
                try:
                    self.pool.putconn(conn)
                except Exception as e:
                    logger.error(f"❌ Failed to return connection to pool: {e}")
            
            duration = (time.time() - start_time) * 1000
            if duration > 1000:  # Log slow operations
                logger.warning(f"⚠️ Slow database operation: {duration:.2f}ms")
    
    def _init_schema(self) -> bool:
        """Initialize database schema with comprehensive error handling"""
        try:
            with self.get_connection() as conn:
                if not conn:
                    logger.error("❌ Cannot initialize schema - no connection")
                    return False
                
                with conn.cursor() as cur:
                    # Enable required extensions
                    cur.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
                    
                    # ML Training Data
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS ml_training_data (
                            id BIGSERIAL PRIMARY KEY,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            symbol VARCHAR(20) NOT NULL,
                            signal_id BIGINT,
                            
                            -- Features (JSONB for flexibility)
                            features JSONB NOT NULL DEFAULT '{}',
                            
                            -- Labels (actual outcomes)
                            sl_pct_used NUMERIC(8,4),
                            tp_pct_used NUMERIC(8,4),
                            entry_price NUMERIC(12,6),
                            exit_price NUMERIC(12,6),
                            pnl NUMERIC(12,4),
                            outcome VARCHAR(10) CHECK (outcome IN ('profit', 'loss', 'breakeven')),
                            
                            -- Trade metadata
                            trade_duration_minutes INTEGER CHECK (trade_duration_minutes >= 0),
                            market_conditions JSONB NOT NULL DEFAULT '{}',
                            side VARCHAR(10) CHECK (side IN ('BUY', 'SELL', 'LONG', 'SHORT')),
                            
                            -- Versioning
                            feature_version VARCHAR(20) DEFAULT '1.0',
                            model_version VARCHAR(20) DEFAULT '1.0'
                        );
                        
                        -- Create indexes for common queries
                        CREATE INDEX IF NOT EXISTS idx_ml_training_symbol 
                            ON ml_training_data(symbol);
                        CREATE INDEX IF NOT EXISTS idx_ml_training_created 
                            ON ml_training_data(created_at DESC);
                        CREATE INDEX IF NOT EXISTS idx_ml_training_outcome 
                            ON ml_training_data(outcome);
                        CREATE INDEX IF NOT EXISTS idx_ml_training_side 
                            ON ml_training_data(side);
                        CREATE INDEX IF NOT EXISTS idx_ml_training_symbol_created 
                            ON ml_training_data(symbol, created_at DESC);
                    """)
                    
                    # ML Models Registry
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS ml_models (
                            id BIGSERIAL PRIMARY KEY,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            model_type VARCHAR(50) NOT NULL CHECK (model_type != ''),
                            model_name VARCHAR(100) NOT NULL CHECK (model_name != ''),
                            symbol VARCHAR(20),
                            version VARCHAR(20) NOT NULL CHECK (version != ''),
                            
                            -- Training metadata
                            training_samples INTEGER CHECK (training_samples >= 0),
                            validation_score NUMERIC(8,6) CHECK (validation_score >= 0 AND validation_score <= 1),
                            test_score NUMERIC(8,6) CHECK (test_score >= 0 AND test_score <= 1),
                            feature_importance JSONB NOT NULL DEFAULT '{}',
                            hyperparameters JSONB NOT NULL DEFAULT '{}',
                            
                            -- Storage
                            model_path TEXT NOT NULL CHECK (model_path != ''),
                            model_size_mb NUMERIC(10,2) CHECK (model_size_mb >= 0),
                            
                            -- Status
                            is_active BOOLEAN DEFAULT false,
                            deployed_at TIMESTAMPTZ,
                            deactivated_at TIMESTAMPTZ,
                            
                            -- Performance tracking
                            total_predictions INTEGER DEFAULT 0 CHECK (total_predictions >= 0),
                            successful_predictions INTEGER DEFAULT 0 CHECK (successful_predictions >= 0),
                            
                            -- Additional metadata
                            training_duration_seconds INTEGER CHECK (training_duration_seconds >= 0),
                            data_version VARCHAR(20) DEFAULT '1.0',
                            
                            UNIQUE(model_type, model_name, symbol, version)
                        );
                        
                        -- Create indexes for model queries
                        CREATE INDEX IF NOT EXISTS idx_ml_models_active 
                            ON ml_models(is_active) WHERE is_active = true;
                        CREATE INDEX IF NOT EXISTS idx_ml_models_symbol 
                            ON ml_models(symbol);
                        CREATE INDEX IF NOT EXISTS idx_ml_models_type 
                            ON ml_models(model_type);
                        CREATE INDEX IF NOT EXISTS idx_ml_models_created 
                            ON ml_models(created_at DESC);
                        CREATE INDEX IF NOT EXISTS idx_ml_models_symbol_active 
                            ON ml_models(symbol, is_active) WHERE is_active = true;
                    """)
                    
                    # ML Predictions Log
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS ml_predictions (
                            id BIGSERIAL PRIMARY KEY,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            model_id BIGINT REFERENCES ml_models(id) ON DELETE SET NULL,
                            
                            symbol VARCHAR(20) NOT NULL,
                            signal_id BIGINT,
                            
                            -- Predictions
                            predicted_sl_pct NUMERIC(8,4) CHECK (predicted_sl_pct >= 0),
                            predicted_tp_pct NUMERIC(8,4) CHECK (predicted_tp_pct >= 0),
                            predicted_rr NUMERIC(6,3) CHECK (predicted_rr >= 0),
                            
                            -- Actual values used (after safety validation)
                            actual_sl_pct NUMERIC(8,4) CHECK (actual_sl_pct >= 0),
                            actual_tp_pct NUMERIC(8,4) CHECK (actual_tp_pct >= 0),
                            
                            -- Confidence & blending
                            model_confidence NUMERIC(5,4) CHECK (model_confidence >= 0 AND model_confidence <= 1),
                            blend_weight NUMERIC(5,4) CHECK (blend_weight >= 0 AND blend_weight <= 1),
                            
                            -- Features snapshot
                            features JSONB NOT NULL DEFAULT '{}',
                            
                            -- Validation results
                            safety_adjusted BOOLEAN DEFAULT false,
                            adjustment_reason TEXT,
                            
                            -- Outcome (filled after trade closes)
                            actual_pnl NUMERIC(12,4),
                            prediction_error_sl NUMERIC(8,4),
                            prediction_error_tp NUMERIC(8,4),
                            
                            -- Additional metadata
                            processing_time_ms INTEGER CHECK (processing_time_ms >= 0),
                            feature_count INTEGER CHECK (feature_count >= 0)
                        );
                        
                        -- Create indexes for prediction queries
                        CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol 
                            ON ml_predictions(symbol);
                        CREATE INDEX IF NOT EXISTS idx_ml_predictions_created 
                            ON ml_predictions(created_at DESC);
                        CREATE INDEX IF NOT EXISTS idx_ml_predictions_model 
                            ON ml_predictions(model_id);
                        CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol_created 
                            ON ml_predictions(symbol, created_at DESC);
                        CREATE INDEX IF NOT EXISTS idx_ml_predictions_outcome 
                            ON ml_predictions(actual_pnl) WHERE actual_pnl IS NOT NULL;
                    """)
                    
                    # ML Performance Metrics
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS ml_performance (
                            id BIGSERIAL PRIMARY KEY,
                            evaluated_at TIMESTAMPTZ DEFAULT NOW(),
                            model_id BIGINT REFERENCES ml_models(id) ON DELETE CASCADE,
                            
                            -- Evaluation period
                            eval_start_date DATE NOT NULL,
                            eval_end_date DATE NOT NULL,
                            CHECK (eval_end_date >= eval_start_date),
                            
                            -- Prediction accuracy
                            total_predictions INTEGER CHECK (total_predictions >= 0),
                            mae_sl NUMERIC(8,6) CHECK (mae_sl >= 0),
                            mae_tp NUMERIC(8,6) CHECK (mae_tp >= 0),
                            rmse_sl NUMERIC(8,6) CHECK (rmse_sl >= 0),
                            rmse_tp NUMERIC(8,6) CHECK (rmse_tp >= 0),
                            
                            -- Trading performance
                            win_rate NUMERIC(5,4) CHECK (win_rate >= 0 AND win_rate <= 1),
                            avg_rr NUMERIC(6,3) CHECK (avg_rr >= 0),
                            total_pnl NUMERIC(12,4),
                            sharpe_ratio NUMERIC(8,4),
                            
                            -- Comparison with traditional
                            traditional_win_rate NUMERIC(5,4) CHECK (traditional_win_rate >= 0 AND traditional_win_rate <= 1),
                            traditional_pnl NUMERIC(12,4),
                            improvement_pct NUMERIC(6,3),
                            
                            -- Per-symbol breakdown
                            symbol_metrics JSONB NOT NULL DEFAULT '{}',
                            
                            -- Additional metrics
                            calmar_ratio NUMERIC(8,4),
                            sortino_ratio NUMERIC(8,4),
                            max_drawdown NUMERIC(12,4)
                        );
                        
                        -- Create indexes for performance queries
                        CREATE INDEX IF NOT EXISTS idx_ml_performance_model 
                            ON ml_performance(model_id);
                        CREATE INDEX IF NOT EXISTS idx_ml_performance_date 
                            ON ml_performance(eval_end_date DESC);
                        CREATE INDEX IF NOT EXISTS idx_ml_performance_symbol 
                            ON ml_performance USING GIN (symbol_metrics);
                    """)
                    
                    # Feature Store
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS ml_feature_store (
                            symbol VARCHAR(20) PRIMARY KEY,
                            updated_at TIMESTAMPTZ DEFAULT NOW(),
                            
                            -- Latest features
                            features JSONB NOT NULL DEFAULT '{}',
                            
                            -- Metadata
                            feature_version VARCHAR(20) DEFAULT '1.0',
                            calculation_time_ms INTEGER CHECK (calculation_time_ms >= 0),
                            
                            -- Indicators snapshot
                            rsi NUMERIC(6,3) CHECK (rsi >= 0 AND rsi <= 100),
                            atr NUMERIC(12,6) CHECK (atr >= 0),
                            volume_ratio NUMERIC(8,4) CHECK (volume_ratio >= 0),
                            trend_strength NUMERIC(6,3),
                            
                            -- Additional indicators
                            volatility NUMERIC(8,4) CHECK (volatility >= 0),
                            momentum NUMERIC(8,4),
                            bb_position NUMERIC(8,4)
                        );
                        
                        -- Create indexes for feature store
                        CREATE INDEX IF NOT EXISTS idx_ml_features_updated 
                            ON ml_feature_store(updated_at DESC);
                        CREATE INDEX IF NOT EXISTS idx_ml_features_rsi 
                            ON ml_feature_store(rsi) WHERE rsi IS NOT NULL;
                        CREATE INDEX IF NOT EXISTS idx_ml_features_volatility 
                            ON ml_feature_store(volatility) WHERE volatility IS NOT NULL;
                    """)
                    
                    # RL Agent State
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS rl_agent_state (
                            agent_name VARCHAR(100) PRIMARY KEY CHECK (agent_name != ''),
                            updated_at TIMESTAMPTZ DEFAULT NOW(),
                            
                            -- Agent state
                            state JSONB NOT NULL DEFAULT '{}',
                            episode INTEGER DEFAULT 0 CHECK (episode >= 0),
                            total_reward NUMERIC(12,4),
                            
                            -- Exploration parameters
                            epsilon NUMERIC(5,4) CHECK (epsilon >= 0 AND epsilon <= 1),
                            learning_rate NUMERIC(8,6) CHECK (learning_rate >= 0),
                            
                            -- Performance
                            avg_reward NUMERIC(12,4),
                            best_reward NUMERIC(12,4),
                            
                            -- Model path
                            model_path TEXT,
                            last_training TIMESTAMP WITH TIME ZONE,
                            
                            -- Additional metadata
                            training_steps BIGINT DEFAULT 0 CHECK (training_steps >= 0),
                            exploration_decay NUMERIC(8,6) CHECK (exploration_decay >= 0 AND exploration_decay <= 1)
                        );
                    """)
                    
                    # Data Drift Detection
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS ml_drift_detection (
                            id BIGSERIAL PRIMARY KEY,
                            detected_at TIMESTAMPTZ DEFAULT NOW(),
                            
                            drift_type VARCHAR(50) NOT NULL CHECK (drift_type != ''),
                            symbol VARCHAR(20),
                            
                            -- Drift metrics
                            drift_score NUMERIC(8,6) CHECK (drift_score >= 0),
                            threshold NUMERIC(8,6) CHECK (threshold >= 0),
                            
                            -- Affected features
                            affected_features JSONB NOT NULL DEFAULT '[]',
                            
                            -- Actions taken
                            action_taken VARCHAR(100),
                            retraining_triggered BOOLEAN DEFAULT false,
                            model_id BIGINT REFERENCES ml_models(id) ON DELETE SET NULL,
                            
                            -- Additional metrics
                            sample_size INTEGER CHECK (sample_size >= 0),
                            detection_duration_ms INTEGER CHECK (detection_duration_ms >= 0)
                        );
                        
                        -- Create indexes for drift detection
                        CREATE INDEX IF NOT EXISTS idx_ml_drift_detected 
                            ON ml_drift_detection(detected_at DESC);
                        CREATE INDEX IF NOT EXISTS idx_ml_drift_symbol 
                            ON ml_drift_detection(symbol);
                        CREATE INDEX IF NOT EXISTS idx_ml_drift_score 
                            ON ml_drift_detection(drift_score DESC);
                        CREATE INDEX IF NOT EXISTS idx_ml_drift_model 
                            ON ml_drift_detection(model_id);
                    """)
                    
                    # Health Monitoring Table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS ml_system_health (
                            id BIGSERIAL PRIMARY KEY,
                            recorded_at TIMESTAMPTZ DEFAULT NOW(),
                            component VARCHAR(50) NOT NULL,
                            status VARCHAR(20) NOT NULL CHECK (status IN ('HEALTHY', 'DEGRADED', 'UNHEALTHY', 'CRITICAL')),
                            score INTEGER CHECK (score >= 0 AND score <= 100),
                            issues JSONB NOT NULL DEFAULT '[]',
                            metrics JSONB NOT NULL DEFAULT '{}'
                        );
                        
                        CREATE INDEX IF NOT EXISTS idx_ml_health_component 
                            ON ml_system_health(component);
                        CREATE INDEX IF NOT EXISTS idx_ml_health_status 
                            ON ml_system_health(status);
                        CREATE INDEX IF NOT EXISTS idx_ml_health_recorded 
                            ON ml_system_health(recorded_at DESC);
                    """)
                    
                    conn.commit()
                    logger.success("✅ ML database schema initialized with all tables and indexes")
                    return True
                    
        except Exception as e:
            logger.error(f"❌ Failed to initialize schema: {e}")
            return False
    
    # ========================================
    # Training Data Operations
    # ========================================
    
    def insert_training_data(
        self,
        symbol: str,
        features: Dict,
        sl_pct: float,
        tp_pct: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        outcome: str,
        trade_duration_minutes: int = None,
        market_conditions: Dict = None,
        side: str = None,
        signal_id: int = None,
        feature_version: str = "1.0",
        model_version: str = "1.0"
    ) -> Optional[int]:
        """Insert training data record with comprehensive validation"""
        try:
            # Validate inputs
            if not symbol or len(symbol) > 20:
                logger.error(f"❌ Invalid symbol: {symbol}")
                return None
            
            if not isinstance(features, dict):
                logger.error("❌ Features must be a dictionary")
                return None
            
            if outcome not in ['profit', 'loss', 'breakeven']:
                logger.error(f"❌ Invalid outcome: {outcome}")
                return None
            
            if side and side.upper() not in ['BUY', 'SELL', 'LONG', 'SHORT']:
                logger.error(f"❌ Invalid side: {side}")
                return None
            
            with self.get_connection() as conn:
                if not conn:
                    return None
                
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO ml_training_data (
                            symbol, features, sl_pct_used, tp_pct_used,
                            entry_price, exit_price, pnl, outcome,
                            trade_duration_minutes, market_conditions, side, signal_id,
                            feature_version, model_version
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        symbol[:20], json.dumps(features), 
                        float(sl_pct) if sl_pct is not None else None,
                        float(tp_pct) if tp_pct is not None else None,
                        float(entry_price) if entry_price is not None else None,
                        float(exit_price) if exit_price is not None else None,
                        float(pnl) if pnl is not None else None,
                        outcome,
                        int(trade_duration_minutes) if trade_duration_minutes is not None else None,
                        json.dumps(market_conditions or {}),
                        side.upper() if side else None,
                        int(signal_id) if signal_id is not None else None,
                        feature_version,
                        model_version
                    ))
                    
                    record_id = cur.fetchone()[0]
                    conn.commit()
                    
                    logger.debug(f"📝 Inserted training data record {record_id} for {symbol}")
                    return record_id
                    
        except Exception as e:
            logger.error(f"❌ Failed to insert training data: {e}")
            return None
    
    def get_training_data(
        self,
        symbol: str = None,
        min_samples: int = 100,
        since_date: datetime = None,
        limit: int = None
    ) -> List[Dict]:
        """Get training data for model training with flexible filtering"""
        try:
            with self.get_connection() as conn:
                if not conn:
                    return []
                
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    query = """
                        SELECT 
                            id, created_at, symbol, features, 
                            sl_pct_used, tp_pct_used, entry_price, exit_price, pnl, outcome,
                            trade_duration_minutes, market_conditions, side, signal_id,
                            feature_version, model_version
                        FROM ml_training_data 
                        WHERE 1=1
                    """
                    params = []
                    
                    if symbol:
                        query += " AND symbol = %s"
                        params.append(symbol)
                    
                    if since_date:
                        query += " AND created_at >= %s"
                        params.append(since_date)
                    
                    query += " ORDER BY created_at DESC"
                    
                    if limit:
                        query += f" LIMIT {int(limit)}"
                    elif not symbol:  # Global model
                        query += f" LIMIT {int(min_samples * 10)}"  # Get more samples
                    
                    cur.execute(query, params)
                    rows = cur.fetchall()
                    
                    result = []
                    for row in rows:
                        row_dict = dict(row)
                        # Convert JSON fields back to Python objects
                        if row_dict.get('features'):
                            try:
                                row_dict['features'] = json.loads(row_dict['features'])
                            except:
                                row_dict['features'] = {}
                        if row_dict.get('market_conditions'):
                            try:
                                row_dict['market_conditions'] = json.loads(row_dict['market_conditions'])
                            except:
                                row_dict['market_conditions'] = {}
                        result.append(row_dict)
                    
                    logger.debug(f"📊 Retrieved {len(result)} training records")
                    return result
                    
        except Exception as e:
            logger.error(f"❌ Failed to get training data: {e}")
            return []
    
    def get_training_statistics(self, symbol: str = None) -> Dict[str, Any]:
        """Get comprehensive statistics about training data"""
        try:
            with self.get_connection() as conn:
                if not conn:
                    return {}
                
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    if symbol:
                        cur.execute("""
                            SELECT 
                                COUNT(*) as total_trades,
                                COUNT(*) FILTER (WHERE outcome = 'profit') as profitable_trades,
                                COUNT(*) FILTER (WHERE outcome = 'loss') as losing_trades,
                                AVG(pnl) as avg_pnl,
                                AVG(sl_pct_used) as avg_sl_pct,
                                AVG(tp_pct_used) as avg_tp_pct,
                                MIN(created_at) as earliest_trade,
                                MAX(created_at) as latest_trade
                            FROM ml_training_data 
                            WHERE symbol = %s
                        """, (symbol,))
                    else:
                        cur.execute("""
                            SELECT 
                                COUNT(*) as total_trades,
                                COUNT(*) FILTER (WHERE outcome = 'profit') as profitable_trades,
                                COUNT(*) FILTER (WHERE outcome = 'loss') as losing_trades,
                                AVG(pnl) as avg_pnl,
                                AVG(sl_pct_used) as avg_sl_pct,
                                AVG(tp_pct_used) as avg_tp_pct,
                                MIN(created_at) as earliest_trade,
                                MAX(created_at) as latest_trade
                            FROM ml_training_data
                        """)
                    
                    row = cur.fetchone()
                    if not row:
                        return {}
                    
                    stats = dict(row)
                    if stats['total_trades'] > 0:
                        stats['win_rate'] = stats['profitable_trades'] / stats['total_trades'] * 100
                    else:
                        stats['win_rate'] = 0.0
                    
                    logger.debug(f"📈 Training statistics for {symbol or 'all symbols'}: {stats}")
                    return stats
                    
        except Exception as e:
            logger.error(f"❌ Failed to get training statistics: {e}")
            return {}
    
    # ========================================
    # Model Registry Operations
    # ========================================
    
    def register_model(
        self,
        model_type: str,
        model_name: str,
        version: str,
        model_path: str,
        training_samples: int,
        validation_score: float,
        hyperparameters: Dict,
        feature_importance: Dict = None,
        symbol: str = None,
        test_score: float = None,
        model_size_mb: float = None,
        training_duration_seconds: int = None,
        data_version: str = "1.0"
    ) -> Optional[int]:
        """Register a trained model with comprehensive validation"""
        try:
            # Validate inputs
            if not model_type or not model_name or not version or not model_path:
                logger.error("❌ Missing required model parameters")
                return None
            
            if training_samples < 0:
                logger.error(f"❌ Invalid training samples: {training_samples}")
                return None
            
            if validation_score < 0 or validation_score > 1:
                logger.error(f"❌ Invalid validation score: {validation_score}")
                return None
            
            with self.get_connection() as conn:
                if not conn:
                    return None
                
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO ml_models (
                            model_type, model_name, symbol, version,
                            training_samples, validation_score, test_score,
                            hyperparameters, feature_importance, model_path,
                            model_size_mb, training_duration_seconds, data_version
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        model_type, model_name, symbol, version,
                        int(training_samples), float(validation_score), 
                        float(test_score) if test_score is not None else None,
                        json.dumps(hyperparameters), 
                        json.dumps(feature_importance or {}),
                        model_path,
                        float(model_size_mb) if model_size_mb is not None else None,
                        int(training_duration_seconds) if training_duration_seconds is not None else None,
                        data_version
                    ))
                    
                    model_id = cur.fetchone()[0]
                    conn.commit()
                    
                    logger.success(f"✅ Model registered: {model_name} v{version} (ID: {model_id})")
                    return model_id
                    
        except Exception as e:
            logger.error(f"❌ Failed to register model: {e}")
            return None
    
    def activate_model(self, model_id: int) -> bool:
        """Activate a model for production use with comprehensive error handling"""
        try:
            if model_id <= 0:
                logger.error(f"❌ Invalid model ID: {model_id}")
                return False
            
            with self.get_connection() as conn:
                if not conn:
                    return False
                
                with conn.cursor() as cur:
                    # First, get the model details
                    cur.execute("""
                        SELECT model_type, symbol FROM ml_models WHERE id = %s
                    """, (model_id,))
                    model_info = cur.fetchone()
                    
                    if not model_info:
                        logger.error(f"❌ Model {model_id} not found")
                        return False
                    
                    model_type, symbol = model_info
                    
                    # Deactivate all other models of the same type and symbol
                    if symbol:
                        cur.execute("""
                            UPDATE ml_models
                            SET is_active = false, deactivated_at = NOW()
                            WHERE id != %s
                            AND model_type = %s
                            AND symbol = %s
                            AND is_active = true
                        """, (model_id, model_type, symbol))
                    else:
                        cur.execute("""
                            UPDATE ml_models
                            SET is_active = false, deactivated_at = NOW()
                            WHERE id != %s
                            AND model_type = %s
                            AND symbol IS NULL
                            AND is_active = true
                        """, (model_id, model_type))
                    
                    # Activate the new model
                    cur.execute("""
                        UPDATE ml_models
                        SET is_active = true, deployed_at = NOW()
                        WHERE id = %s
                        RETURNING id
                    """, (model_id,))
                    
                    result = cur.fetchone()
                    if not result:
                        logger.error(f"❌ Failed to activate model {model_id}")
                        return False
                    
                    conn.commit()
                    logger.success(f"✅ Model {model_id} activated successfully")
                    return True
                    
        except Exception as e:
            logger.error(f"❌ Failed to activate model: {e}")
            return False
    
    def get_active_model(self, model_type: str, symbol: str = None) -> Optional[Dict]:
        """Get active model for prediction with comprehensive error handling"""
        try:
            if not model_type:
                logger.error("❌ Model type is required")
                return None
            
            with self.get_connection() as conn:
                if not conn:
                    return None
                
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    if symbol:
                        cur.execute("""
                            SELECT * FROM ml_models
                            WHERE is_active = true
                            AND model_type = %s
                            AND symbol = %s
                            ORDER BY deployed_at DESC
                            LIMIT 1
                        """, (model_type, symbol))
                    else:
                        cur.execute("""
                            SELECT * FROM ml_models
                            WHERE is_active = true
                            AND model_type = %s
                            AND symbol IS NULL
                            ORDER BY deployed_at DESC
                            LIMIT 1
                        """, (model_type,))
                    
                    row = cur.fetchone()
                    if not row:
                        logger.debug(f"🔍 No active model found for {model_type} {symbol or '(global)'}")
                        return None
                    
                    model_dict = dict(row)
                    # Convert JSON fields back to Python objects
                    if model_dict.get('hyperparameters'):
                        try:
                            model_dict['hyperparameters'] = json.loads(model_dict['hyperparameters'])
                        except:
                            model_dict['hyperparameters'] = {}
                    if model_dict.get('feature_importance'):
                        try:
                            model_dict['feature_importance'] = json.loads(model_dict['feature_importance'])
                        except:
                            model_dict['feature_importance'] = {}
                    
                    logger.debug(f"🔍 Found active model: {model_dict.get('model_name')} v{model_dict.get('version')}")
                    return model_dict
                    
        except Exception as e:
            logger.error(f"❌ Failed to get active model: {e}")
            return None
    
    def get_model_performance(self, model_id: int) -> Optional[Dict]:
        """Get comprehensive performance metrics for a model"""
        try:
            if model_id <= 0:
                return None
            
            with self.get_connection() as conn:
                if not conn:
                    return None
                
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT 
                            mp.*,
                            mm.model_name,
                            mm.version,
                            mm.symbol
                        FROM ml_performance mp
                        JOIN ml_models mm ON mp.model_id = mm.id
                        WHERE mp.model_id = %s
                        ORDER BY mp.eval_end_date DESC
                        LIMIT 1
                    """, (model_id,))
                    
                    row = cur.fetchone()
                    if not row:
                        return None
                    
                    perf_dict = dict(row)
                    if perf_dict.get('symbol_metrics'):
                        try:
                            perf_dict['symbol_metrics'] = json.loads(perf_dict['symbol_metrics'])
                        except:
                            perf_dict['symbol_metrics'] = {}
                    
                    return perf_dict
                    
        except Exception as e:
            logger.error(f"❌ Failed to get model performance: {e}")
            return None
    
    # ========================================
    # Prediction Logging
    # ========================================
    
    def log_prediction(
        self,
        model_id: int,
        symbol: str,
        predicted_sl_pct: float,
        predicted_tp_pct: float,
        predicted_rr: float,
        actual_sl_pct: float,
        actual_tp_pct: float,
        model_confidence: float,
        blend_weight: float,
        features: Dict,
        safety_adjusted: bool = False,
        adjustment_reason: str = None,
        signal_id: int = None,
        processing_time_ms: int = None,
        feature_count: int = None
    ) -> Optional[int]:
        """Log a prediction for tracking with comprehensive validation"""
        try:
            # Validate inputs
            if model_id <= 0:
                logger.error(f"❌ Invalid model ID: {model_id}")
                return None
            
            if not symbol or len(symbol) > 20:
                logger.error(f"❌ Invalid symbol: {symbol}")
                return None
            
            if not isinstance(features, dict):
                logger.error("❌ Features must be a dictionary")
                return None
            
            with self.get_connection() as conn:
                if not conn:
                    return None
                
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO ml_predictions (
                            model_id, symbol, signal_id,
                            predicted_sl_pct, predicted_tp_pct, predicted_rr,
                            actual_sl_pct, actual_tp_pct,
                            model_confidence, blend_weight,
                            features, safety_adjusted, adjustment_reason,
                            processing_time_ms, feature_count
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        int(model_id), symbol[:20], 
                        int(signal_id) if signal_id is not None else None,
                        float(predicted_sl_pct) if predicted_sl_pct is not None else None,
                        float(predicted_tp_pct) if predicted_tp_pct is not None else None,
                        float(predicted_rr) if predicted_rr is not None else None,
                        float(actual_sl_pct) if actual_sl_pct is not None else None,
                        float(actual_tp_pct) if actual_tp_pct is not None else None,
                        float(model_confidence) if model_confidence is not None else None,
                        float(blend_weight) if blend_weight is not None else None,
                        json.dumps(features),
                        bool(safety_adjusted),
                        adjustment_reason[:500] if adjustment_reason else None,
                        int(processing_time_ms) if processing_time_ms is not None else None,
                        int(feature_count) if feature_count is not None else None
                    ))
                    
                    pred_id = cur.fetchone()[0]
                    conn.commit()
                    
                    # Update model's prediction count
                    cur.execute("""
                        UPDATE ml_models
                        SET total_predictions = total_predictions + 1
                        WHERE id = %s
                    """, (model_id,))
                    conn.commit()
                    
                    logger.debug(f"📝 Logged prediction {pred_id} for model {model_id}")
                    return pred_id
                    
        except Exception as e:
            logger.error(f"❌ Failed to log prediction: {e}")
            return None
    
    def update_prediction_outcome(
        self,
        prediction_id: int,
        actual_pnl: float,
        prediction_error_sl: float = None,
        prediction_error_tp: float = None
    ) -> bool:
        """Update prediction with actual outcome and calculate success"""
        try:
            if prediction_id <= 0:
                logger.error(f"❌ Invalid prediction ID: {prediction_id}")
                return False
            
            with self.get_connection() as conn:
                if not conn:
                    return False
                
                with conn.cursor() as cur:
                    # First, get the model_id for this prediction
                    cur.execute("""
                        SELECT model_id FROM ml_predictions WHERE id = %s
                    """, (prediction_id,))
                    result = cur.fetchone()
                    
                    if not result:
                        logger.error(f"❌ Prediction {prediction_id} not found")
                        return False
                    
                    model_id = result[0]
                    
                    # Update the prediction
                    cur.execute("""
                        UPDATE ml_predictions
                        SET actual_pnl = %s,
                            prediction_error_sl = %s,
                            prediction_error_tp = %s
                        WHERE id = %s
                        RETURNING id
                    """, (
                        float(actual_pnl) if actual_pnl is not None else None,
                        float(prediction_error_sl) if prediction_error_sl is not None else None,
                        float(prediction_error_tp) if prediction_error_tp is not None else None,
                        prediction_id
                    ))
                    
                    if not cur.fetchone():
                        logger.error(f"❌ Failed to update prediction {prediction_id}")
                        return False
                    
                    # If prediction was successful (positive PnL), update model's successful predictions
                    if actual_pnl is not None and actual_pnl > 0:
                        cur.execute("""
                            UPDATE ml_models
                            SET successful_predictions = successful_predictions + 1
                            WHERE id = %s
                        """, (model_id,))
                    
                    conn.commit()
                    logger.debug(f"✅ Updated prediction {prediction_id} with outcome")
                    return True
                    
        except Exception as e:
            logger.error(f"❌ Failed to update prediction outcome: {e}")
            return False
    
    def get_prediction_statistics(self, model_id: int = None, symbol: str = None) -> Dict[str, Any]:
        """Get comprehensive statistics about predictions"""
        try:
            with self.get_connection() as conn:
                if not conn:
                    return {}
                
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    query = """
                        SELECT 
                            COUNT(*) as total_predictions,
                            COUNT(*) FILTER (WHERE actual_pnl > 0) as successful_predictions,
                            COUNT(*) FILTER (WHERE actual_pnl < 0) as failed_predictions,
                            AVG(actual_pnl) as avg_pnl,
                            AVG(predicted_sl_pct) as avg_predicted_sl,
                            AVG(actual_sl_pct) as avg_actual_sl,
                            AVG(predicted_tp_pct) as avg_predicted_tp,
                            AVG(actual_tp_pct) as avg_actual_tp,
                            AVG(model_confidence) as avg_confidence,
                            AVG(processing_time_ms) as avg_processing_time
                        FROM ml_predictions 
                        WHERE 1=1
                    """
                    params = []
                    
                    if model_id:
                        query += " AND model_id = %s"
                        params.append(model_id)
                    
                    if symbol:
                        query += " AND symbol = %s"
                        params.append(symbol)
                    
                    cur.execute(query, params)
                    row = cur.fetchone()
                    
                    if not row:
                        return {}
                    
                    stats = dict(row)
                    if stats['total_predictions'] > 0:
                        stats['success_rate'] = stats['successful_predictions'] / stats['total_predictions'] * 100
                    else:
                        stats['success_rate'] = 0.0
                    
                    logger.debug(f"📈 Prediction statistics for model {model_id or 'all'}: {stats}")
                    return stats
                    
        except Exception as e:
            logger.error(f"❌ Failed to get prediction statistics: {e}")
            return {}
    
    # ========================================
    # Feature Store Operations
    # ========================================
    
    def update_feature_store(
        self,
        symbol: str,
        features: Dict,
        rsi: float = None,
        atr: float = None,
        volume_ratio: float = None,
        trend_strength: float = None,
        volatility: float = None,
        momentum: float = None,
        bb_position: float = None,
        calculation_time_ms: int = None,
        feature_version: str = "1.0"
    ) -> bool:
        """Update feature store with latest calculated features"""
        try:
            if not symbol or len(symbol) > 20:
                logger.error(f"❌ Invalid symbol: {symbol}")
                return False
            
            if not isinstance(features, dict):
                logger.error("❌ Features must be a dictionary")
                return False
            
            with self.get_connection() as conn:
                if not conn:
                    return False
                
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO ml_feature_store (
                            symbol, features, rsi, atr, volume_ratio, trend_strength,
                            volatility, momentum, bb_position, calculation_time_ms, feature_version
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol) DO UPDATE SET
                            features = EXCLUDED.features,
                            rsi = EXCLUDED.rsi,
                            atr = EXCLUDED.atr,
                            volume_ratio = EXCLUDED.volume_ratio,
                            trend_strength = EXCLUDED.trend_strength,
                            volatility = EXCLUDED.volatility,
                            momentum = EXCLUDED.momentum,
                            bb_position = EXCLUDED.bb_position,
                            calculation_time_ms = EXCLUDED.calculation_time_ms,
                            feature_version = EXCLUDED.feature_version,
                            updated_at = NOW()
                    """, (
                        symbol[:20], json.dumps(features),
                        float(rsi) if rsi is not None else None,
                        float(atr) if atr is not None else None,
                        float(volume_ratio) if volume_ratio is not None else None,
                        float(trend_strength) if trend_strength is not None else None,
                        float(volatility) if volatility is not None else None,
                        float(momentum) if momentum is not None else None,
                        float(bb_position) if bb_position is not None else None,
                        int(calculation_time_ms) if calculation_time_ms is not None else None,
                        feature_version
                    ))
                    
                    conn.commit()
                    logger.debug(f"🔄 Updated feature store for {symbol}")
                    return True
                    
        except Exception as e:
            logger.error(f"❌ Failed to update feature store: {e}")
            return False
    
    def get_latest_features(self, symbol: str) -> Optional[Dict]:
        """Get latest calculated features for a symbol"""
        try:
            if not symbol:
                return None
            
            with self.get_connection() as conn:
                if not conn:
                    return None
                
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM ml_feature_store WHERE symbol = %s
                    """, (symbol,))
                    
                    row = cur.fetchone()
                    if not row:
                        return None
                    
                    features_dict = dict(row)
                    if features_dict.get('features'):
                        try:
                            features_dict['features'] = json.loads(features_dict['features'])
                        except:
                            features_dict['features'] = {}
                    
                    logger.debug(f"🔍 Retrieved latest features for {symbol}")
                    return features_dict
                    
        except Exception as e:
            logger.error(f"❌ Failed to get latest features: {e}")
            return None
    
    # ========================================
    # System Health Monitoring
    # ========================================
    
    def log_system_health(
        self,
        component: str,
        status: str,
        score: int,
        issues: List[str] = None,
        metrics: Dict = None
    ) -> bool:
        """Log system health status"""
        try:
            if not component or not status or score < 0 or score > 100:
                return False
            
            with self.get_connection() as conn:
                if not conn:
                    return False
                
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO ml_system_health (
                            component, status, score, issues, metrics
                        ) VALUES (%s, %s, %s, %s, %s)
                    """, (
                        component,
                        status,
                        score,
                        json.dumps(issues or []),
                        json.dumps(metrics or {})
                    ))
                    
                    conn.commit()
                    return True
                    
        except Exception as e:
            logger.error(f"❌ Failed to log system health: {e}")
            return False
    
    def get_system_health(self, component: str = None, limit: int = 10) -> List[Dict]:
        """Get system health logs"""
        try:
            with self.get_connection() as conn:
                if not conn:
                    return []
                
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    query = "SELECT * FROM ml_system_health WHERE 1=1"
                    params = []
                    
                    if component:
                        query += " AND component = %s"
                        params.append(component)
                    
                    query += " ORDER BY recorded_at DESC LIMIT %s"
                    params.append(limit)
                    
                    cur.execute(query, params)
                    rows = cur.fetchall()
                    
                    result = []
                    for row in rows:
                        row_dict = dict(row)
                        if row_dict.get('issues'):
                            try:
                                row_dict['issues'] = json.loads(row_dict['issues'])
                            except:
                                row_dict['issues'] = []
                        if row_dict.get('metrics'):
                            try:
                                row_dict['metrics'] = json.loads(row_dict['metrics'])
                            except:
                                row_dict['metrics'] = {}
                        result.append(row_dict)
                    
                    return result
                    
        except Exception as e:
            logger.error(f"❌ Failed to get system health: {e}")
            return []
    
    def close(self) -> None:
        """Close connection pool gracefully"""
        try:
            if self.pool:
                self.pool.closeall()
                logger.info("🔒 ML Database connection pool closed gracefully")
        except Exception as e:
            logger.error(f"❌ Failed to close database connection pool: {e}")

# Global instance with thread safety
import threading
_ml_db_instance: Optional[MLDatabase] = None
_ml_db_lock = threading.Lock()

def get_ml_db() -> MLDatabase:
    """Get global ML database instance with thread safety"""
    global _ml_db_instance
    with _ml_db_lock:
        if _ml_db_instance is None:
            _ml_db_instance = MLDatabase()
        return _ml_db_instance

def reset_ml_db():
    """Reset global ML database instance (for testing or reconfiguration)"""
    global _ml_db_instance
    with _ml_db_lock:
        if _ml_db_instance:
            _ml_db_instance.close()
        _ml_db_instance = None
        logger.info("🔄 ML Database instance reset")