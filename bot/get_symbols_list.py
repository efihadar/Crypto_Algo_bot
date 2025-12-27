# get_symbols_list.py
import sys
import configparser
from pathlib import Path
from typing import List, Dict, Any, Tuple
from loguru import logger

# --- CONFIGURATION ---
project_root = Path(__file__).parent
INI_FILE_PATH = project_root.parent / "strategy_config.ini"

try:
    from bot.sessions import BybitSession
except ImportError:
    try:
        from sessions import BybitSession
    except ImportError:
        logger.error("❌ Critical: Could not find BybitSession module.")
        sys.exit(1)

def get_budget_constraints() -> float:
    """
    Reads the allowed USDT per order from the config.
    Used to filter out symbols where the minimum lot size exceeds our budget.
    """
    config = configparser.ConfigParser()
    try:
        config.read(INI_FILE_PATH)
        # Assuming your INI key is USDT_PER_ORDER. Adjust if different.
        return config.getfloat("trading", "USDT_PER_ORDER", fallback=20.0)
    except Exception:
        return 20.0

def get_instrument_info(session: BybitSession) -> Dict[str, float]:
    """
    Fetches minimum order quantities for all symbols to ensure tradeability.
    """
    try:
        instruments = session.get_instruments_info()
        return {
            ins['symbol']: float(ins['lotSizeFilter']['minOrderQty']) 
            for ins in instruments if 'lotSizeFilter' in ins
        }
    except Exception as e:
        logger.error(f"❌ Error fetching instrument info: {e}")
        return {}

def filter_symbols_elite(tickers_data: List[Dict[str, Any]], min_qtys: Dict[str, float], budget: float) -> Tuple[List[str], List[str]]:
    config = configparser.ConfigParser()
    config.read(INI_FILE_PATH)

    MIN_TURNOVER = config.getfloat("momentum_trading", "MIN_VOLUME_USDT", fallback=5000000.0)
    MIN_CHANGE = config.getfloat("momentum_trading", "MIN_MOMENTUM_PCT", fallback=4.0)
    MAX_CHANGE = config.getfloat("momentum_trading", "MAX_MOMENTUM_PCT", fallback=25.0)
    
    SAFE_SYMBOLS = {
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
    'ADAUSDT', 'SUIUSDT', 'LINKUSDT', 'BNBUSDT', 'HYPEUSDT',
    'WIFUSDT', 'BCHUSDT', 'AVAXUSDT', 'DOTUSDT', 'LTCUSDT',
    'ATOMUSDT', 'ETCUSDT', 'UNIUSDT', 'AAVEUSDT', 'MATICUSDT',
    'NEARUSDT', 'ARBUSDT', 'OPUSDT', 'APTUSDT', 'INJUSDT'
    }
    
    BLACKLIST = {
        'ZKPUSDT', 'BEATUSDT', 'LIGHTUSDT', 'PIPPINUSDT', 
        'FARTCOINUSDT', 'ZBTUSDT', '0GUSDT', 'NIGHTUSDT',
        'RAVEUSDT', 'XAUTUSDT'
    }
    
    trading_candidates = []
    momentum_candidates = []

    for ticker in tickers_data:
        symbol = ticker.get('symbol', '')
        if not symbol.endswith('USDT'): 
            continue
        
        if symbol in BLACKLIST:
            continue
        if symbol not in SAFE_SYMBOLS:
            continue

        try:
            last_price = float(ticker.get('lastPrice', 0))
            turnover_24h = float(ticker.get('turnover24h', 0))
            price_change_pct = float(ticker.get('price24hPcnt', 0))
            min_qty = min_qtys.get(symbol, 0)

            if (min_qty * last_price) > budget: 
                continue
            if turnover_24h < MIN_TURNOVER: 
                continue

            # Regular Trading
            if turnover_24h > 20_000_000:
                trading_candidates.append({'symbol': symbol, 'vol': turnover_24h})

            # Momentum
            if MIN_CHANGE <= price_change_pct <= MAX_CHANGE:
                score = price_change_pct * (turnover_24h / 1_000_000)
                momentum_candidates.append({'symbol': symbol, 'score': score})

        except (ValueError, TypeError):
            continue

    final_trading = [s['symbol'] for s in sorted(trading_candidates, key=lambda x: x['vol'], reverse=True)[:15]]
    final_momentum = [s['symbol'] for s in sorted(momentum_candidates, key=lambda x: x['score'], reverse=True)[:10]]

    return final_trading, final_momentum

def update_config_file(trading_list: List[str], momentum_list: List[str]):
    """
    Writes the filtered symbols back to strategy_config.ini.
    Only writes if changes are detected to save disk I/O.
    """
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(INI_FILE_PATH)

    sections_to_update = [
        ("trading", "SYMBOLS", trading_list),
        ("momentum_trading", "MOMENTUM_SYMBOLS", momentum_list)
    ]

    modified = False
    for section, key, data in sections_to_update:
        if not data: continue
        if section not in config: config.add_section(section)
        
        new_val = ",".join(data)
        if config.get(section, key, fallback="") != new_val:
            config.set(section, key, new_val)
            modified = True

    if modified:
        with open(INI_FILE_PATH, "w") as f:
            config.write(f)
        logger.success(f"✅ Config updated: {len(trading_list)} trading, {len(momentum_list)} momentum symbols.")

def run_update_process():
    """
    Main execution flow for symbol synchronization.
    """
    session = BybitSession()
    budget = get_budget_constraints()
    
    # Fetch live market data
    tickers_res = session.get_tickers(category="linear")
    tickers_data = tickers_res.get('result', {}).get('list', [])
    min_qtys = get_instrument_info(session)
    
    if not tickers_data:
        logger.error("❌ Failed to fetch tickers from exchange.")
        return

    # Process Elite filtering
    trading_final, momentum_final = filter_symbols_elite(tickers_data, min_qtys, budget)
    
    # Sync with INI
    update_config_file(trading_final, momentum_final)

if __name__ == "__main__":
    run_update_process()