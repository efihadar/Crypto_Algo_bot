# startup_check.py
import os
import sys
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Load environment variables - FIXED PATH
current_dir = Path(__file__).parent
project_root = current_dir.parent
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"✅ Loaded .env from: {env_path}")
else:
    logger.error(f"❌ .env file not found at: {env_path}")
    sys.exit(1)

# Add project root to path for imports
sys.path.insert(0, str(project_root))

def check_env_variables() -> bool:
    """Check if essential environment variables are set"""
    required_vars = {
        "BYBIT_API_KEY": "Bybit API Key",
        "BYBIT_API_SECRET": "Bybit API Secret",
        "POSTGRES_DB": "PostgreSQL Database Name",
        "POSTGRES_USER": "PostgreSQL User", 
        "POSTGRES_PASSWORD": "PostgreSQL Password"
    }
    
    missing = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            missing.append(f"  ❌ {var} ({description})")
        else:
            # Mask sensitive values
            if "SECRET" in var or "PASSWORD" in var:
                display_value = "*" * 8
            elif "KEY" in var:
                display_value = value[:8] + "..." if len(value) > 8 else value
            else:
                display_value = value
            logger.info(f"  ✅ {var}: {display_value}")
    
    if missing:
        logger.error("Missing environment variables:")
        for m in missing:
            logger.error(m)
        return False
    
    return True

def check_bybit_connection() -> bool:
    """Check Bybit API connection"""
    try:
        from pybit.unified_trading import HTTP
        
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")
        testnet = os.getenv("BYBIT_TESTNET", "true").lower() in ("true", "1", "yes")
        
        if not api_key or not api_secret:
            logger.error("❌ Bybit API credentials not found")
            return False
        
        client = HTTP(api_key=api_key, api_secret=api_secret, testnet=testnet)
        
        # Test connection
        response = client.get_server_time()
        if response['retCode'] == 0:
            logger.success("✅ Bybit API connection successful")
            logger.info(f"  Testnet mode: {testnet}")
        else:
            raise Exception(f"API error: {response['retMsg']}")
        
        # Get account info - USE SPOT TRADING, NOT WALLET BALANCE
        try:
            # For spot trading, we use the spot endpoints directly
            # Check if we can get ticker data and place orders
            
            # Test market data
            ticker = client.get_tickers(category="spot", symbol="BTCUSDT")
            if ticker['retCode'] == 0:
                price = float(ticker['result']['list'][0]['lastPrice'])
                logger.info(f"  BTC/USDT price: ${price:,.2f}")
                
                # Try to get account balance for spot trading
                try:
                    account = client.get_wallet_balance(accountType="UNIFIED")
                    if account['retCode'] == 0 and account['result']['list']:
                        balance_data = account['result']['list'][0]
                        
                        # Check different balance fields
                        total_balance = float(balance_data.get('totalWalletBalance', 0))
                        total_equity = float(balance_data.get('totalEquity', 0))
                        available_balance = float(balance_data.get('availableToWithdraw', 0))
                        
                        if total_balance > 0:
                            logger.success(f"  ✅ UNIFIED Account Balance: ${total_balance:.2f}")
                        elif total_equity > 0:
                            logger.success(f"  ✅ UNIFIED Account Equity: ${total_equity:.2f}")
                        elif available_balance > 0:
                            logger.success(f"  ✅ Available Balance: ${available_balance:.2f}")
                        else:
                            logger.warning("  ⚠️ No balance found in UNIFIED account")
                            logger.info("  💡 Please transfer funds from SPOT to UNIFIED account")
                            
                    else:
                        logger.warning(f"  ⚠️ Could not get balance: {account.get('retMsg', 'Unknown error')}")
                        
                except Exception as e:
                    logger.warning(f"  ⚠️ Balance check failed: {e}")
                    
            else:
                logger.warning(f"  ⚠️ Could not get market data: {ticker['retMsg']}")
                
        except Exception as e:
            logger.warning(f"  ⚠️ Account check failed: {e}")
        
        return True
        
    except ImportError:
        logger.error("❌ pybit not installed. Run: pip install pybit")
        return False
    except Exception as e:
        logger.error(f"❌ Bybit connection failed: {e}")
        return False

def check_database_connection() -> bool:
    """Check PostgreSQL connection"""
    try:
        import psycopg2
        
        # Try multiple connection options
        db_configs = [
            {
                'host': os.getenv("DB_HOST", "localhost"),
                'port': os.getenv("DB_PORT", "5432"),
                'dbname': os.getenv("DB_NAME", os.getenv("POSTGRES_DB", "trading_db")),
                'user': os.getenv("DB_USER", os.getenv("POSTGRES_USER", "trader")),
                'password': os.getenv("DB_PASS", os.getenv("POSTGRES_PASSWORD", "trader_pass")),
            },
            {
                'host': 'localhost',
                'port': '5433',  # Docker compose default
                'dbname': 'trading_db',
                'user': 'trader', 
                'password': 'trader_pass',
            }
        ]
        
        for i, config in enumerate(db_configs):
            try:
                conn = psycopg2.connect(
                    host=config['host'],
                    port=config['port'],
                    dbname=config['dbname'],
                    user=config['user'],
                    password=config['password'],
                    connect_timeout=5
                )
                
                cur = conn.cursor()
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                logger.success(f"✅ Database connection successful")
                logger.info(f"  PostgreSQL: {version.split(',')[0]}")
                logger.info(f"  Host: {config['host']}:{config['port']}")
                
                cur.close()
                conn.close()
                return True
                
            except Exception as e:
                if i == len(db_configs) - 1:  # Last attempt
                    logger.error(f"❌ Database connection failed: {e}")
                else:
                    logger.debug(f"  Attempt {i+1} failed: {e}")
        
        return False
        
    except ImportError:
        logger.error("❌ psycopg2 not installed. Run: pip install psycopg2-binary")
        return False

def check_python_dependencies() -> bool:
    """Check if required Python packages are installed"""
    required_packages = {
        "pybit": "pybit",
        "pandas": "pandas",
        "numpy": "numpy", 
        "psycopg2": "psycopg2-binary",
        "loguru": "loguru",
        "dotenv": "python-dotenv"
    }
    
    missing = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            logger.info(f"  ✅ {package}")
        except ImportError:
            missing.append(f"  ❌ {pip_name}")
    
    if missing:
        logger.error("Missing Python packages:")
        for m in missing:
            logger.error(m)
        logger.info("\nInstall missing packages:")
        logger.info("  pip install -r requirements.txt")
        return False
    
    logger.success("✅ All required packages installed")
    return True

def pre_flight_check() -> bool:
    """Comprehensive startup check"""
    print("=" * 70)
    print("🚀 BYBIT TRADING BOT - PRE-FLIGHT CHECK")
    print("=" * 70)
    print()
    
    checks = {
        "Python Dependencies": check_python_dependencies,
        "Environment Variables": check_env_variables,
        "Bybit API Connection": check_bybit_connection,
        "Database Connection": check_database_connection,
    }
    
    results = {}
    for check_name, check_func in checks.items():
        logger.info(f"\n📋 Checking: {check_name}")
        try:
            results[check_name] = check_func()
        except Exception as e:
            logger.error(f"❌ Check failed with exception: {e}")
            results[check_name] = False
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 CHECK SUMMARY")
    print("=" * 70)
    
    passed = 0
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status:12s} {check_name}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\n  Results: {passed}/{total} checks passed")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 All checks passed! Bot is ready to trade.")
        print("\n📝 Next steps:")
        print("  1. Review your configuration in .env")
        print("  2. Start with small position sizes (POSITION_SIZE_USDT=10)")
        print("  3. Enable DEBUG_MODE=true for testing")
        print("  4. Run: python bot_main.py")
        return True
    else:
        print(f"\n🔴 {total - passed} checks failed. Please fix issues above.")
        return False

if __name__ == "__main__":
    success = pre_flight_check()
    sys.exit(0 if success else 1)