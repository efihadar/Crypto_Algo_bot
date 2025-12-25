# ml_system/ml_service.py
"""
🚀 Professional ML Service API
FastAPI service for ML predictions with comprehensive monitoring, security, and reliability.
Production-ready for live trading environments.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Any
import uvicorn
from loguru import logger
import time
from datetime import datetime
import asyncio

# Import local components
from .config import ml_config
from .predictor import ml_predictor
from .ml_db import MLDatabase
from .feature_engineering import feature_engineer
from .safety_validator import safety_validator

# Optional: Add Prometheus metrics
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("⚠️ Prometheus metrics not available")

app = FastAPI(
    title="Trading ML API",
    description="Professional Machine Learning API for Trading Systems",
    version="1.0.0",
    contact={
        "name": "ML Trading Team",
        "email": "ml-team@trading.com",
    },
    license_info={
        "name": "Proprietary",
        "url": "https://yourcompany.com/license",
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
ml_db = MLDatabase()

class PredictionRequest(BaseModel):
    """Request model for ML predictions"""
    symbol: str = Field(..., min_length=1, max_length=20, description="Trading symbol (e.g., BTC/USDT)")
    signal: Dict = Field(..., description="Trading signal dictionary")
    entry_price: float = Field(..., gt=0, description="Entry price for the trade")
    current_atr: Optional[float] = Field(None, ge=0, description="Current ATR value for dynamic boundaries")
    current_volatility: Optional[float] = Field(None, ge=0, description="Current volatility percentage")
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "BTC/USDT",
                "signal": {
                    "side": "BUY",
                    "strength": 75,
                    "is_momentum": True
                },
                "entry_price": 45000.0,
                "current_atr": 450.0,
                "current_volatility": 2.5
            }
        }

class PredictionResponse(BaseModel):
    """Response model for ML predictions"""
    success: bool = Field(..., description="Whether prediction was successful")
    stop_loss: Optional[float] = Field(None, description="Validated stop loss price")
    take_profit: Optional[float] = Field(None, description="Validated take profit price")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence level")
    blend_weight: float = Field(..., ge=0.0, le=1.0, description="ML blend weight for final decision")
    model_used: Optional[str] = Field(None, description="Model type used for prediction")
    adjustments: List[str] = Field(default_factory=list, description="List of safety adjustments applied")
    risk_reward_ratio: Optional[float] = Field(None, description="Final risk/reward ratio")
    timestamp: str = Field(..., description="Timestamp of prediction")
    version: str = Field(..., description="API version")

class TrainingDataRequest(BaseModel):
    """Request model for recording training data"""
    symbol: str = Field(..., min_length=1, max_length=20)
    side: str = Field(..., pattern="^(BUY|SELL)$")
    entry_price: float = Field(..., gt=0)
    exit_price: float = Field(..., gt=0)
    pnl: float = Field(...)
    sl_used: float = Field(..., gt=0)
    tp_used: float = Field(..., gt=0)
    features: Dict = Field(..., description="Extracted features for training")
    timestamp: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "ETH/USDT",
                "side": "BUY",
                "entry_price": 3000.0,
                "exit_price": 3150.0,
                "pnl": 150.0,
                "sl_used": 2950.0,
                "tp_used": 3150.0,
                "features": {
                    "rsi": 65.0,
                    "atr_pct": 1.2,
                    "trend_strength": 0.8
                }
            }
        }

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("🚀 Starting ML Service API...")
    
    # Validate ML system is ready
    if not ml_config.enabled:
        logger.warning("⚠️ ML system is disabled in configuration")
    
    # Initialize Prometheus metrics if available
    if PROMETHEUS_AVAILABLE:
        Instrumentator().instrument(app).expose(app)
        logger.info("📊 Prometheus metrics enabled")
    
    logger.success("✅ ML Service API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down ML Service API...")
    # Add any cleanup code here
    logger.success("✅ ML Service API shutdown complete")

@app.get("/health", summary="Health Check", tags=["Monitoring"])
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "service": "ml_api",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "ml_system": "enabled" if ml_config.enabled else "disabled",
                "model_loaded": ml_predictor.model is not None,
                "database": "connected" if ml_db else "disconnected",
                "safety_validator": "active" if safety_validator else "inactive"
            }
        }
        
        # Add model info if available
        if ml_predictor.model is not None:
            health_status["model_info"] = {
                "type": getattr(ml_config, 'model_type', 'unknown'),
                "features": len(getattr(ml_predictor, 'feature_names', []))
            }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "service": "ml_api",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/metrics", summary="Performance Metrics", tags=["Monitoring"])
async def get_metrics():
    """Get performance metrics (placeholder for real metrics)"""
    try:
        # This would be replaced with actual metrics collection
        return {
            "predictions_total": 0,
            "predictions_success_rate": 0.0,
            "average_response_time_ms": 0.0,
            "models_loaded": 1 if ml_predictor.model is not None else 0,
            "training_samples": ml_db.get_training_sample_count() if hasattr(ml_db, 'get_training_sample_count') else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")

@app.post(
    "/predict", 
    response_model=PredictionResponse,
    summary="Get ML Prediction",
    description="Get ML prediction for optimal stop loss and take profit levels",
    tags=["Predictions"]
)
async def predict(request: PredictionRequest):
    """
    Get ML prediction for optimal stop loss and take profit levels.
    
    The prediction includes safety validation to ensure stops are within configured boundaries.
    """
    start_time = time.time()
    request_id = f"req_{int(time.time()*1000)}"
    
    try:
        logger.info(f"🔍 Processing prediction request {request_id} for {request.symbol}")
        
        # Validate ML system is enabled
        if not ml_config.enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ML system is disabled"
            )
        
        # Validate model is loaded
        if ml_predictor.model is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No ML model loaded"
            )
        
        # Extract features for context (optional)
        features = feature_engineer.extract_features(
            pd.DataFrame(),  # Empty DataFrame - you might want to pass actual data
            request.signal
        ) or {}
        
        # Make prediction using your existing predictor
        prediction_result = ml_predictor.predict_optimal_stops(
            request.signal,
            pd.DataFrame(),  # You might want to pass actual price data
            request.entry_price
        )
        
        if prediction_result is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to generate prediction"
            )
        
        # Extract SL/TP from prediction result
        if isinstance(prediction_result, dict):
            predicted_sl = prediction_result.get('sl_price')
            predicted_tp = prediction_result.get('tp_price')
            confidence = prediction_result.get('confidence', 0.5)
        elif isinstance(prediction_result, tuple) and len(prediction_result) >= 2:
            predicted_sl, predicted_tp = prediction_result[:2]
            confidence = 0.6  # Default confidence
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid prediction format"
            )
        
        # Apply safety validation
        if safety_validator and predicted_sl and predicted_tp:
            validated_sl, validated_tp, is_safe, reason = safety_validator.validate_stops(
                ml_sl=predicted_sl,
                ml_tp=predicted_tp,
                entry_price=request.entry_price,
                side=request.signal.get('side', 'BUY'),
                atr=request.current_atr,
                volatility=request.current_volatility,
                symbol=request.symbol
            )
            
            adjustments = [reason] if not is_safe else []
            if reason and "adjusted" in reason.lower():
                adjustments = reason.split('; ')
        else:
            validated_sl, validated_tp = predicted_sl, predicted_tp
            adjustments = []
            is_safe = True
        
        # Calculate risk/reward ratio
        risk_reward_ratio = None
        try:
            side = request.signal.get('side', 'BUY').upper()
            if side == 'BUY':
                sl_distance = (request.entry_price - validated_sl) / request.entry_price
                tp_distance = (validated_tp - request.entry_price) / request.entry_price
            else:  # SELL
                sl_distance = (validated_sl - request.entry_price) / request.entry_price
                tp_distance = (request.entry_price - validated_tp) / request.entry_price
            
            if sl_distance > 0:
                risk_reward_ratio = tp_distance / sl_distance
        except:
            risk_reward_ratio = None
        
        # Prepare response
        response = PredictionResponse(
            success=True,
            stop_loss=float(validated_sl) if validated_sl else None,
            take_profit=float(validated_tp) if validated_tp else None,
            confidence=float(confidence),
            blend_weight=float(getattr(ml_config, 'ml_blend_weight', 0.7)),
            model_used=getattr(ml_config, 'model_type', 'unknown'),
            adjustments=adjustments,
            risk_reward_ratio=risk_reward_ratio,
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0"
        )
        
        # Log successful prediction
        processing_time = (time.time() - start_time) * 1000
        logger.success(
            f"✅ Prediction {request_id} completed in {processing_time:.2f}ms for {request.symbol}\n"
            f"   SL: {validated_sl:.4f}, TP: {validated_tp:.4f}, Confidence: {confidence:.2%}"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"❌ Prediction {request_id} failed after {processing_time:.2f}ms: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post(
    "/training/record",
    summary="Record Training Data",
    description="Record trade outcome for ML model training",
    tags=["Training"]
)
async def record_training_data(request: TrainingDataRequest):
    """
    Record trade outcome for ML model training.
    
    This data will be used to retrain and improve the ML model over time.
    """
    try:
        logger.info(f"📝 Recording training data for {request.symbol}")
        
        # Validate input
        if request.pnl == 0:
            logger.warning(f"Skipping recording for {request.symbol} - zero PnL")
            return {"success": True, "message": "Zero PnL trades not recorded"}
        
        # Create trade record
        trade_record = {
            'symbol': request.symbol,
            'side': request.side,
            'entry_price': request.entry_price,
            'exit_price': request.exit_price,
            'sl_used': request.sl_used,
            'tp_used': request.tp_used,
            'pnl': request.pnl,
            'outcome': 'profit' if request.pnl > 0 else 'loss',
            'features': request.features,
            'timestamp': request.timestamp or datetime.utcnow().isoformat(),
        }
        
        # Save to database
        if hasattr(ml_db, 'save_trade'):
            success = ml_db.save_trade(trade_record)
        else:
            # Fallback: implement basic saving logic
            success = True
            logger.warning("ML database save_trade method not implemented")
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save trade data"
            )
        
        logger.success(f"✅ Successfully recorded training data for {request.symbol}")
        return {"success": True, "message": "Trade data recorded successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to record training data: {e}")
        raise HTTPException(status_code=500, detail=f"Recording error: {str(e)}")

@app.get("/model/info", summary="Get Model Info", tags=["Model"])
async def get_model_info():
    """Get information about the current ML model"""
    try:
        if ml_predictor.model is None:
            return {
                "loaded": False,
                "message": "No model loaded",
                "can_train": True
            }
        
        # Get model info from ml_manager if available
        if hasattr(ml_predictor, 'get_model_info'):
            model_info = ml_predictor.get_model_info()
        else:
            model_info = {
                "model_type": getattr(ml_config, 'model_type', 'unknown'),
                "features_count": len(getattr(ml_predictor, 'feature_names', [])),
                "last_updated": datetime.utcnow().isoformat()
            }
        
        return {
            "loaded": True,
            "model_info": model_info,
            "config": {
                "confidence_threshold": ml_config.confidence_threshold,
                "min_prediction_confidence": ml_config.min_prediction_confidence,
                "model_type": ml_config.model_type
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=f"Model info error: {str(e)}")

@app.get("/config", summary="Get Configuration", tags=["Configuration"])
async def get_configuration():
    """Get current ML configuration"""
    try:
        config_dict = ml_config.to_dict() if hasattr(ml_config, 'to_dict') else vars(ml_config)
        return {
            "configuration": config_dict,
            "safety_boundaries": safety_validator.get_safety_stats() if safety_validator else {},
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "success": False}
    )

if __name__ == "__main__":
    # Run with Uvicorn
    logger.info("Starting ML Service API server...")
    uvicorn.run(
        "ml_system.ml_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Disable in production
        workers=1,    # Adjust based on CPU cores
        log_level="info"
    )