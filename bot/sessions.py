# sessions.py - ENHANCED PROFESSIONAL VERSION
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from loguru import logger

try:
    from pybit.unified_trading import HTTP
except ImportError as e:
    logger.error("❌ pybit.unified_trading not installed. Please add 'pybit' to requirements.")
    raise

BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "false").lower() in ("1", "true", "yes", "y")

class BybitSession:
    """
    Bybit API Session Wrapper for USDT Perpetual (Linear) trading.
    Provides a clean interface for common trading operations with caching and error handling.
    """
    _last_account_print_time = 0
    _ACCOUNT_PRINT_INTERVAL = 30
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None) -> None:
        """
        Initialize Bybit session with caching and retry logic.
        """
        api_key = api_key or os.getenv("BYBIT_API_KEY")
        api_secret = api_secret or os.getenv("BYBIT_API_SECRET")

        if not api_key or not api_secret:
            raise RuntimeError("BYBIT_API_KEY / BYBIT_API_SECRET not set")

        logger.info(f"🔌 Creating Bybit HTTP client (testnet={BYBIT_TESTNET})")

        self.client = HTTP(
            testnet=BYBIT_TESTNET,
            api_key=api_key,
            api_secret=api_secret,
            recv_window=20000,
            force_retry=True,
            retry_codes=[10002],
            max_retries=3,
        )

        self.category = "linear"

        # Caches with TTL
        self._limits_cache: Dict[str, Dict[str, float]] = {}
        self._limits_last_update: Dict[str, float] = {}
        self._limits_ttl_seconds: int = 300

        self._kline_cache: Dict[str, Dict[str, Any]] = {}
        self._kline_cache_ttl: int = 60

        self._ticker_cache: Dict[str, Dict[str, Any]] = {}
        self._ticker_cache_ttl: int = 10

        logger.success("✅ BybitSession initialized for USDT Perpetual (linear)")
        logger.info(f"  ⏰ recv_window: 20000ms (20s) - handles clock drift up to ±10s")

    # =====================================================
    # 🔢 HELPER: Format price as string for API
    # =====================================================
    def _format_price_string(self, symbol: str, price: float) -> str:
        """
        Format price as string with correct decimal places for API.
        Uses tick_size to determine precision.
        """
        try:
            limits = self.get_symbol_limits(symbol)
            tick_size = limits.get("tick_size", 0.0001) if limits else 0.0001
            
            if tick_size <= 0:
                tick_size = 0.0001

            # Calculate decimal places from tick size
            tick_str = f"{tick_size:.10f}".rstrip('0').rstrip('.')
            decimals = len(tick_str.split('.')[1]) if '.' in tick_str else 0

            # Normalize and format
            normalized = self.normalize_price(symbol, price)
            return f"{normalized:.{decimals}f}"
            
        except Exception as e:
            logger.warning(f"⚠️ _format_price_string failed for {symbol}: {e}")
            return f"{price:.8f}".rstrip('0').rstrip('.')

    # =====================================================
    # 💰 Balance / Account info (UNIFIED USDT)
    # =====================================================
    def _get_balance(self) -> Optional[float]:
        """
        Returns USDT balance in UNIFIED account.
        """
        try:
            res = self.client.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT",
            )
            if res.get("retCode") != 0:
                logger.error(f"❌ get_wallet_balance error: {res}")
                return None

            lst = res.get("result", {}).get("list", [])
            if not lst:
                return None

            coins = lst[0].get("coin", [])
            if not coins:
                return None

            usdt_info = coins[0]
            balance = float(usdt_info.get("walletBalance", 0))
            logger.success(f"💰 Successfully fetched UNIFIED balance: {balance}")
            return balance
        except Exception as e:
            logger.error(f"❌ _get_balance failed: {e}")
            return None

    def get_account_info(self) -> Optional[Dict[str, float]]:
        """
        Get comprehensive account information including balance, equity, margin info.
        Returns dict with: balance, equity, margin, free_margin, margin_ratio
        """
        try:
            res = self.client.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT",
            )
            if res.get("retCode") != 0:
                logger.error(f"❌ get_wallet_balance error: {res}")
                return None

            lst = res.get("result", {}).get("list", [])
            if not lst:
                logger.warning("⚠️ No account list returned")
                return None

            account_data = lst[0]
            coins = account_data.get("coin", [])
            
            if not coins:
                logger.warning("⚠️ No coin data in account")
                return None

            usdt_info = next((c for c in coins if c.get("coin") == "USDT"), coins[0])

            balance = float(usdt_info.get("walletBalance", 0) or 0)
            equity = float(usdt_info.get("equity", 0) or 0)
            if equity == 0 and balance > 0:
                equity = balance
            
            total_margin_balance = float(account_data.get("totalMarginBalance", 0) or 0)
            total_available_balance = float(account_data.get("totalAvailableBalance", 0) or 0)
            total_initial_margin = float(account_data.get("totalInitialMargin", 0) or 0)
            total_maintenance_margin = float(account_data.get("totalMaintenanceMargin", 0) or 0)
            
            used_margin = total_initial_margin
            free_margin = total_available_balance if total_available_balance > 0 else (equity - used_margin)
            margin_ratio = (used_margin / equity * 100) if equity > 0 else 0.0

            result = {
                "balance": balance,
                "equity": equity,
                "margin": used_margin,
                "used_margin": used_margin,
                "free_margin": free_margin,
                "available_balance": total_available_balance,
                "margin_ratio": margin_ratio,
                "total_margin_balance": total_margin_balance,
                "initial_margin": total_initial_margin,
                "maintenance_margin": total_maintenance_margin,
            }

            current_time = time.time()
            if current_time - self._last_account_print_time >= self._ACCOUNT_PRINT_INTERVAL:
                logger.success(
                    f"💰 Account info: Balance=${balance:.2f}, Equity=${equity:.2f}, "
                    f"Margin=${used_margin:.2f}, Free=${free_margin:.2f}"
                )
                self._last_account_print_time = current_time
            
            return result

        except Exception as e:
            logger.error(f"❌ get_account_info failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    # =====================================================
    # 📊 POSITIONS
    # =====================================================
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all open positions or positions for a specific symbol.
        Returns list of position dicts with normalized field names.
        """
        try:
            params: Dict[str, Any] = {"category": self.category, "settleCoin": "USDT"}
            if symbol:
                params["symbol"] = symbol.upper()
            
            res = self.client.get_positions(**params)
            if res.get("retCode") != 0:
                logger.error(f"❌ get_positions error: {res}")
                return []
            
            positions_raw = res.get("result", {}).get("list", [])
            positions = [
                {
                    "symbol": pos.get("symbol"),
                    "side": pos.get("side"),
                    "size": size,
                    "qty": size,
                    "avgPrice": float(pos.get("avgPrice", 0) or 0),
                    "entryPrice": float(pos.get("avgPrice", 0) or 0),
                    "unrealisedPnl": float(pos.get("unrealisedPnl", 0) or 0),
                    "leverage": float(pos.get("leverage", 1) or 1),
                    "positionValue": float(pos.get("positionValue", 0) or 0),
                    "liqPrice": float(pos.get("liqPrice", 0) or 0),
                    "stopLoss": pos.get("stopLoss", ""),
                    "takeProfit": pos.get("takeProfit", ""),
                    "trailingStop": pos.get("trailingStop", ""),
                    "positionIdx": int(pos.get("positionIdx", 0) or 0),
                    "createdTime": pos.get("createdTime", ""),
                    "updatedTime": pos.get("updatedTime", ""),
                }
                for pos in positions_raw
                if (size := float(pos.get("size", 0) or 0)) > 0
            ]
            
            if positions:
                logger.debug(f"📊 Found {len(positions)} open positions")
            
            return positions
            
        except Exception as e:
            logger.error(f"❌ get_positions failed: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get a single position for a specific symbol.
        Returns None if no position exists.
        """
        try:
            positions = self.get_positions(symbol=symbol)
            return positions[0] if positions else None
        except Exception as e:
            logger.error(f"❌ get_position failed for {symbol}: {e}")
            return None

    # =====================================================
    # 📈 Market Data - Klines (Candlesticks)
    # =====================================================
    def get_kline(
        self, symbol: str, interval: str = "15", limit: int = 100,
        category: Optional[str] = None, start: Optional[int] = None, end: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Fetch kline/candlestick data for a symbol.
        """
        try:
            category = category or self.category
            params: Dict[str, Any] = {
                "category": category,
                "symbol": symbol.upper(),
                "interval": str(interval),
                "limit": limit,
            }
            if start: params["start"] = start
            if end: params["end"] = end
            
            res = self.client.get_kline(**params)
            if res.get("retCode") != 0:
                logger.error(f"❌ get_kline error for {symbol}: {res}")
                return {"retCode": res.get("retCode", -1), "retMsg": res.get("retMsg", "Error"), "result": {}}
            
            return res
            
        except Exception as e:
            logger.error(f"❌ get_kline failed for {symbol}: {e}")
            return {"retCode": -1, "retMsg": str(e), "result": {}}

    def get_klines(self, symbol: str, interval: str = "15", limit: int = 100) -> List[List[str]]:
        """
        Simplified kline fetch that returns just the list of candles.
        Each candle: [timestamp, open, high, low, close, volume, turnover]
        """
        res = self.get_kline(symbol=symbol, interval=interval, limit=limit)
        return res.get("result", {}).get("list", [])

    # =====================================================
    # 📈 Market Data - Tickers (CRITICAL FIX)
    # =====================================================
    def get_tickers(self, category: Optional[str] = None, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Wrapper for pybit's get_tickers with caching.
        """
        category = category or self.category
        cache_key = f"{category}_{symbol or 'ALL'}"
        now = time.time()

        # Check cache
        if (
            cache_key in self._ticker_cache
            and (now - self._ticker_cache[cache_key]["timestamp"]) < self._ticker_cache_ttl
        ):
            return self._ticker_cache[cache_key]["data"]

        try:
            res = self.client.get_tickers(category=category, symbol=symbol)
            if res.get("retCode") == 0:
                self._ticker_cache[cache_key] = {
                    "data": res,
                    "timestamp": now
                }
            return res
        except Exception as e:
            logger.error(f"❌ get_tickers failed: {e}")
            return {}

    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time ticker data for a symbol. FIXED AND ADDED!
        Returns dict with at least: lastPrice, bid1Price, ask1Price, volume24h
        """
        try:
            symbol = symbol.upper()
            res = self.get_tickers(category=self.category, symbol=symbol)
            
            if res.get("retCode") != 0:
                logger.error(f"❌ get_ticker error for {symbol}: {res}")
                return None
                
            result = res.get("result", {})
            list_data = result.get("list", [])
            
            if not list_data:
                logger.warning(f"⚠️ No ticker data returned for {symbol}")
                return None
                
            ticker = list_data[0]
            
            # Ensure required fields exist
            for field in ["lastPrice", "bid1Price", "ask1Price", "volume24h"]:
                if field not in ticker:
                    ticker[field] = "0"
                    
            return ticker
            
        except Exception as e:
            logger.error(f"❌ get_ticker failed for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the last traded price for a symbol.
        Uses cached ticker data when available.
        """
        try:
            ticker = self.get_ticker(symbol)
            return float(ticker.get("lastPrice", 0)) if ticker else None
        except Exception as e:
            logger.error(f"❌ get_current_price failed: {e}")
            return None

    def get_orderbook(self, symbol: str, limit: int = 25) -> Optional[Dict[str, Any]]:
        """
        Get the orderbook for a symbol.
        """
        try:
            res = self.client.get_orderbook(category=self.category, symbol=symbol, limit=limit)
            return res.get("result", {}) if res.get("retCode") == 0 else None
        except Exception as e:
            logger.error(f"❌ get_orderbook failed: {e}")
            return None

    # =====================================================
    # 📏 EXCHANGE LIMITS
    # =====================================================
    def get_symbol_limits(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get trading limits for a symbol with caching.
        """
        symbol = symbol.upper()
        now = time.time()

        if (
            symbol in self._limits_cache
            and (now - self._limits_last_update.get(symbol, 0)) < self._limits_ttl_seconds
        ):
            return self._limits_cache[symbol]

        try:
            res = self.client.get_instruments_info(category=self.category, symbol=symbol)
            if res.get("retCode") != 0:
                logger.error(f"❌ get_instruments_info error for {symbol}: {res}")
                return None

            lst = res.get("result", {}).get("list", [])
            if not lst:
                logger.error(f"⚠️ No instruments-info for {symbol}")
                return None

            info = lst[0]
            lot_filter = info.get("lotSizeFilter", {})
            price_filter = info.get("priceFilter", {})
            leverage_filter = info.get("leverageFilter", {})

            limits = {
                "min_notional": float(lot_filter.get("minNotionalValue", 0) or 0.0),
                "min_qty": float(lot_filter.get("minOrderQty", 0) or 0.0),
                "max_qty": float(lot_filter.get("maxOrderQty", 0) or 0.0),
                "qty_step": float(lot_filter.get("qtyStep", 0) or 0.0),
                "tick_size": float(price_filter.get("tickSize", 0) or 0.0),
                "min_price": float(price_filter.get("minPrice", 0) or 0.0),
                "max_price": float(price_filter.get("maxPrice", 0) or 0.0),
                "max_leverage": float(leverage_filter.get("maxLeverage", 100) or 100.0),
            }

            self._limits_cache[symbol] = limits
            self._limits_last_update[symbol] = now
            logger.debug(f"ℹ️ Limits for {symbol}: {limits}")
            return limits

        except Exception as e:
            logger.error(f"❌ Failed to fetch symbol limits for {symbol}: {e}")
            return None

    # =====================================================
    # 🔢 NORMALIZE QTY / PRICE
    # =====================================================
    def normalize_qty(self, symbol: str, qty: float, round_up: bool = False) -> float:
        """
        Normalize quantity to match exchange lot size (qtyStep).
        """
        try:
            if qty <= 0:
                return 0.0
                
            limits = self.get_symbol_limits(symbol)
            if not limits:
                return qty

            step = limits.get("qty_step", 0.0)
            if step <= 0:
                return qty

            q_dec = Decimal(str(qty))
            step_dec = Decimal(str(step))
            rounding = ROUND_UP if round_up else ROUND_DOWN
            steps = (q_dec / step_dec).to_integral_value(rounding=rounding)
            normalized = steps * step_dec
            
            min_qty = Decimal(str(limits.get("min_qty", 0)))
            if normalized < min_qty and min_qty > 0:
                normalized = min_qty
            
            return float(normalized)

        except Exception as e:
            logger.error(f"❌ normalize_qty failed for {symbol}: {e}")
            return qty

    def normalize_price(self, symbol: str, price: float) -> float:
        """
        Normalize price to match exchange tick size.
        """
        try:
            if price <= 0:
                return 0.0
                
            limits = self.get_symbol_limits(symbol)
            if not limits:
                return round(price, 8)

            tick_size = limits.get("tick_size", 0.0)
            if tick_size <= 0:
                return round(price, 8)

            p_dec = Decimal(str(price))
            tick_dec = Decimal(str(tick_size))
            ticks = (p_dec / tick_dec).to_integral_value(rounding=ROUND_DOWN)
            normalized_dec = ticks * tick_dec
            
            tick_str = f"{tick_size:.10f}".rstrip('0').rstrip('.')
            decimals = len(tick_str.split('.')[1]) if '.' in tick_str else 0
            return round(float(normalized_dec), decimals)

        except Exception as e:
            logger.error(f"❌ normalize_price failed for {symbol}: {e}")
            return round(price, 4)

    # =====================================================
    # ⚙️ LEVERAGE
    # =====================================================
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol.
        """
        try:
            res = self.client.set_leverage(
                category=self.category,
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
            
            if res.get("retCode") == 0:
                logger.success(f"✅ Leverage set to {leverage}x for {symbol}")
                return True
            elif res.get("retCode") == 110043:
                logger.debug(f"ℹ️ Leverage already {leverage}x for {symbol}")
                return True
            else:
                logger.error(f"❌ set_leverage failed for {symbol}: {res}")
                return False
                
        except Exception as e:
            logger.error(f"❌ set_leverage exception for {symbol}: {e}")
            return False

    def get_leverage(self, symbol: str) -> Optional[float]:
        """
        Get current leverage for a symbol.
        """
        try:
            pos = self.get_position(symbol)
            return pos.get("leverage") if pos else None
        except Exception as e:
            logger.error(f"❌ get_leverage failed for {symbol}: {e}")
            return None

    # =====================================================
    # 🧾 Orders - Market Orders
    # =====================================================
    def place_market_order(self, symbol: str, side: str, qty: float, reduce_only: bool = False) -> Dict[str, Any]:
        """
        Place a Market order for linear (USDT Perp).
        """
        try:
            side_clean = str(side).strip().capitalize()
            if side_clean not in ("Buy", "Sell"):
                logger.error(f"❌ Invalid side '{side}' after normalize → '{side_clean}'")
                return {"status": "ERROR", "error": f"Invalid side '{side}'"}

            qty = self.normalize_qty(symbol, qty)
            if qty <= 0:
                logger.error(f"❌ Invalid quantity after normalization: {qty}")
                return {"status": "ERROR", "error": "Quantity too small"}

            logger.info(f"🧾 Sending MARKET order: {symbol} {side_clean} qty={qty}")

            res = self.client.place_order(
                category=self.category,
                symbol=symbol,
                side=side_clean,
                orderType="Market",
                qty=str(qty),
                timeInForce="GoodTillCancel",
                reduceOnly=reduce_only,
                closeOnTrigger=False,
            )

            if res.get("retCode") != 0:
                err = res.get("retMsg")
                logger.error(f"❌ Failed to place market order for {symbol}: {err}")
                return {"status": "ERROR", "error": err, "raw": res}

            result = res.get("result", {})
            order_id = result.get("orderId")
            fill_price = self.get_current_price(symbol) or 0.0

            logger.success(f"✅ Market order placed: {symbol} {side_clean} qty={qty} (order_id={order_id}, price={fill_price})")

            return {
                "status": "OK",
                "orderId": order_id,
                "price": fill_price,
                "qty": qty,
                "side": side_clean,
                "raw": res,
            }

        except Exception as e:
            logger.error(f"❌ Exception in place_market_order for {symbol}: {e}")
            return {"status": "ERROR", "error": str(e)}

    # =====================================================
    # 🧾 Orders - Limit Orders
    # =====================================================
    def place_limit_order(
        self, symbol: str, side: str, qty: float, price: float,
        reduce_only: bool = False, time_in_force: str = "GoodTillCancel"
    ) -> Dict[str, Any]:
        """
        Place a Limit order for linear (USDT Perp).
        """
        try:
            side_clean = str(side).strip().capitalize()
            if side_clean not in ("Buy", "Sell"):
                logger.error(f"❌ Invalid side '{side}'")
                return {"status": "ERROR", "error": f"Invalid side '{side}'"}

            qty = self.normalize_qty(symbol, qty)
            price = self.normalize_price(symbol, price)
            
            if qty <= 0 or price <= 0:
                logger.error(f"❌ Invalid qty or price after normalization")
                return {"status": "ERROR", "error": "Invalid qty or price"}

            logger.info(f"🧾 Sending LIMIT order: {symbol} {side_clean} qty={qty} @ {price}")

            res = self.client.place_order(
                category=self.category,
                symbol=symbol,
                side=side_clean,
                orderType="Limit",
                qty=str(qty),
                price=str(price),
                timeInForce=time_in_force,
                reduceOnly=reduce_only,
            )

            if res.get("retCode") != 0:
                err = res.get("retMsg")
                logger.error(f"❌ Failed to place limit order for {symbol}: {err}")
                return {"status": "ERROR", "error": err, "raw": res}

            result = res.get("result", {})
            order_id = result.get("orderId")

            logger.success(f"✅ Limit order placed: {symbol} {side_clean} qty={qty} @ {price}")

            return {
                "status": "OK",
                "orderId": order_id,
                "price": price,
                "qty": qty,
                "side": side_clean,
                "raw": res,
            }

        except Exception as e:
            logger.error(f"❌ Exception in place_limit_order for {symbol}: {e}")
            return {"status": "ERROR", "error": str(e)}

    # =====================================================
    # 🧾 Orders - Cancel / Status
    # =====================================================
    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Cancel an open order.
        """
        try:
            res = self.client.cancel_order(category=self.category, symbol=symbol, orderId=order_id)
            if res.get("retCode") != 0:
                err = res.get("retMsg")
                logger.error(f"❌ Failed to cancel order {order_id}: {err}")
                return {"status": "ERROR", "error": err, "raw": res}
            
            logger.success(f"✅ Order cancelled: {order_id}")
            return {"status": "OK", "orderId": order_id, "raw": res}
            
        except Exception as e:
            logger.error(f"❌ cancel_order exception: {e}")
            return {"status": "ERROR", "error": str(e)}

    def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel all open orders.
        """
        try:
            params: Dict[str, Any] = {"category": self.category}
            if symbol:
                params["symbol"] = symbol
            else:
                params["settleCoin"] = "USDT"
            
            res = self.client.cancel_all_orders(**params)
            if res.get("retCode") != 0:
                err = res.get("retMsg")
                logger.error(f"❌ Failed to cancel all orders: {err}")
                return {"status": "ERROR", "error": err, "raw": res}
            
            logger.success(f"✅ All orders cancelled")
            return {"status": "OK", "raw": res}
            
        except Exception as e:
            logger.error(f"❌ cancel_all_orders exception: {e}")
            return {"status": "ERROR", "error": str(e)}

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all open orders.
        """
        try:
            params: Dict[str, Any] = {"category": self.category}
            if symbol:
                params["symbol"] = symbol
            else:
                params["settleCoin"] = "USDT"
            
            res = self.client.get_open_orders(**params)
            return res.get("result", {}).get("list", []) if res.get("retCode") == 0 else []
            
        except Exception as e:
            logger.error(f"❌ get_open_orders exception: {e}")
            return []

    def get_order_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get order history.
        """
        try:
            params: Dict[str, Any] = {"category": self.category, "limit": limit}
            if symbol:
                params["symbol"] = symbol
            
            res = self.client.get_order_history(**params)
            return res.get("result", {}).get("list", []) if res.get("retCode") == 0 else []
            
        except Exception as e:
            logger.error(f"❌ get_order_history exception: {e}")
            return []

    # =====================================================
    # 🎯 TP/SL – set_trading_stop (Position-level)
    # =====================================================
    def set_trading_stop(
        self, symbol: str, side: str, stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None, trailing_stop: Optional[float] = None,
        sl_trigger_by: str = "LastPrice", tp_trigger_by: str = "LastPrice", tpsl_mode: str = "Full"
    ) -> Dict[str, Any]:
        """
        Set TP/SL on the position (position-level).
        """
        try:
            symbol = symbol.upper()
            if stop_loss is None and take_profit is None and trailing_stop is None:
                logger.debug(f"set_trading_stop called for {symbol} without SL/TP/Trail – skipping")
                return {"status": "SKIP"}

            side_clean = str(side).strip().capitalize()
            if side_clean not in ("Buy", "Sell"):
                logger.warning(f"⚠️ Invalid side '{side}' in set_trading_stop, defaulting to Buy")
                side_clean = "Buy"

            payload: Dict[str, Any] = {
                "category": self.category,
                "symbol": symbol,
                "positionIdx": 0,
                "tpslMode": tpsl_mode,
            }

            if stop_loss is not None:
                payload["stopLoss"] = self._format_price_string(symbol, stop_loss)
                payload["slTriggerBy"] = sl_trigger_by

            if take_profit is not None:
                payload["takeProfit"] = self._format_price_string(symbol, take_profit)
                payload["tpTriggerBy"] = tp_trigger_by

            if trailing_stop is not None:
                payload["trailingStop"] = str(trailing_stop)

            logger.info(f"🎯 set_trading_stop → {symbol} side={side_clean}, SL={payload.get('stopLoss')}, TP={payload.get('takeProfit')}, Trail={trailing_stop}")
            logger.debug(f"📤 Payload: {payload}")

            try:
                res = self.client.set_trading_stop(**payload)
                if res.get("retCode") != 0:
                    err = res.get("retMsg")
                    logger.error(f"❌ set_trading_stop error for {symbol}: retCode={res.get('retCode')}, msg={err}")
                    return {"status": "ERROR", "error": err, "raw": res, "retCode": res.get("retCode")}

                logger.success(f"✅ set_trading_stop success for {symbol}: SL={payload.get('stopLoss')}, TP={payload.get('takeProfit')}")
                return {"status": "OK", "raw": res}
                
            except Exception as api_error:
                error_str = str(api_error)
                if "34040" in error_str or "not modified" in error_str.lower() or "110043" in error_str:
                    logger.info(f"ℹ️ SL/TP already set for {symbol} (not modified) - treating as success")
                    return {"status": "OK", "already_set": True}
                raise

        except Exception as e:
            error_str = str(e)
            if "34040" in error_str or "not modified" in error_str.lower():
                logger.info(f"ℹ️ SL/TP already set for {symbol} (not modified) - treating as success")
                return {"status": "OK", "already_set": True}
            
            logger.error(f"❌ Exception in set_trading_stop for {symbol}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {"status": "ERROR", "error": str(e)}

    # =====================================================
    # 📜 History / PnL
    # =====================================================
    def get_closed_pnl(self, symbol: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get closed PnL (realized PnL) history.
        """
        try:
            params: Dict[str, Any] = {"category": self.category, "limit": limit}
            if symbol:
                params["symbol"] = symbol
            
            res = self.client.get_closed_pnl(**params)
            return res.get("result", {}).get("list", []) if res.get("retCode") == 0 else []
        except Exception as e:
            logger.error(f"❌ Exception in get_closed_pnl: {e}")
            return []

    def get_closed_pnl_by_order_id(self, symbol: str, order_id: str) -> Optional[float]:
        """
        Get closed PnL for a specific order.
        """
        trades = self.get_closed_pnl(symbol, limit=50)
        for t in trades:
            if t.get("orderId") == order_id:
                return float(t.get("closedPnl", 0.0))
        return None

    def get_execution_list(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get trade execution history.
        """
        try:
            params: Dict[str, Any] = {"category": self.category, "limit": limit}
            if symbol:
                params["symbol"] = symbol
            
            res = self.client.get_executions(**params)
            return res.get("result", {}).get("list", []) if res.get("retCode") == 0 else []
        except Exception as e:
            logger.error(f"❌ Exception in get_execution_list: {e}")
            return []

    # =====================================================
    # 📋 Symbols List
    # =====================================================
    def get_trading_symbols_list(self, status: str = "Trading") -> List[str]:
        """
        Get list of all active trading symbols.
        """
        logger.info(f"Fetching active trading symbols for category '{self.category}'...")
        try:
            response = self.client.get_instruments_info(category=self.category, status=status)
            if response.get("retCode") != 0:
                logger.error(f"❌ Failed to fetch symbols list: {response}")
                return []
            symbols = [item["symbol"] for item in response.get("result", {}).get("list", [])]
            logger.success(f"✅ Successfully fetched {len(symbols)} active symbols.")
            return symbols
        except Exception as e:
            logger.error(f"❌ Exception while fetching symbols list: {e}")
            return []

    def get_instruments_info(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get detailed instruments info.
        """
        try:
            params: Dict[str, Any] = {"category": self.category}
            if symbol:
                params["symbol"] = symbol
            res = self.client.get_instruments_info(**params)
            return res.get("result", {}).get("list", []) if res.get("retCode") == 0 else []
        except Exception as e:
            logger.error(f"❌ get_instruments_info exception: {e}")
            return []

    # =====================================================
    # 🕒 SESSION HELPERS
    # =====================================================
    def is_open_now_utc(self, now_utc: datetime) -> bool:
        """
        Crypto markets are 24/7.
        """
        return True

    def get_server_time(self) -> Optional[int]:
        """
        Get Bybit server time in milliseconds.
        """
        try:
            res = self.client.get_server_time()
            return int(res.get("result", {}).get("timeSecond", 0)) * 1000 if res.get("retCode") == 0 else None
        except Exception as e:
            logger.error(f"❌ get_server_time failed: {e}")
            return None

    def check_time_sync(self) -> Dict[str, Any]:
        """
        Check if local time is synced with server time.
        """
        try:
            server_time = self.get_server_time()
            if server_time is None:
                return {"is_synced": None, "drift_ms": None, "error": "Could not get server time"}
            local_time = int(time.time() * 1000)
            drift_ms = local_time - server_time
            is_synced = abs(drift_ms) < 5000
            return {"is_synced": is_synced, "drift_ms": drift_ms, "server_time": server_time, "local_time": local_time}
        except Exception as e:
            return {"is_synced": None, "drift_ms": None, "error": str(e)}

    # =====================================================
    # 🔧 UTILITY METHODS
    # =====================================================
    def close_position(self, symbol: str, side: Optional[str] = None) -> Dict[str, Any]:
        """
        Close an entire position for a symbol using a market order.
        """
        try:
            pos = self.get_position(symbol)
            if not pos:
                logger.info(f"ℹ️ No open position for {symbol}")
                return {"status": "NO_POSITION"}
            
            size = pos.get("size", 0)
            if size <= 0:
                logger.info(f"ℹ️ No open position for {symbol}")
                return {"status": "NO_POSITION"}
            
            pos_side = pos.get("side", side)
            if not pos_side:
                logger.error(f"❌ Could not determine position side for {symbol}")
                return {"status": "ERROR", "error": "Unknown position side"}
            
            close_side = "Sell" if pos_side.capitalize() == "Buy" else "Buy"
            logger.info(f"🔄 Closing position: {symbol} {pos_side} size={size}")
            
            return self.place_market_order(symbol=symbol, side=close_side, qty=size, reduce_only=True)
            
        except Exception as e:
            logger.error(f"❌ close_position failed for {symbol}: {e}")
            return {"status": "ERROR", "error": str(e)}

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear cached data for a symbol or all symbols.
        """
        if symbol:
            symbol = symbol.upper()
            self._limits_cache.pop(symbol, None)
            self._limits_last_update.pop(symbol, None)
            self._kline_cache.pop(symbol, None)
            # Clear ticker cache for this symbol
            keys_to_remove = [k for k in self._ticker_cache.keys() if symbol in k]
            for key in keys_to_remove:
                self._ticker_cache.pop(key, None)
        else:
            self._limits_cache.clear()
            self._limits_last_update.clear()
            self._kline_cache.clear()
            self._ticker_cache.clear()
        
        logger.debug(f"🧹 Cache cleared for {symbol or 'all symbols'}")