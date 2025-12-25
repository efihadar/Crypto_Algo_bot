# 🤖 Crypto Trading Bot - Bybit Automated Trading System

[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://www.python.org/)
[![Bybit](https://img.shields.io/badge/Exchange-Bybit-orange)](https://www.bybit.com/)
[![License](https://img.shields.io/badge/License-Private-red)]()


Real-time trading signals
Risk management and stop-loss logic
Position grouping and multi-order execution
Historical data analysis (TradingView webhook support)
Telegram notifications
PostgreSQL + Grafana dashboards for analytics

---

## 📋 Table of Contents

- [Features](#-features)
- [System Requirements](#-system-requirements)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Risk Management](#-risk-management)
- [Monitoring](#-monitoring)
- [Troubleshooting](#-troubleshooting)
- [Security](#-security)
- [Disclaimer](#-disclaimer)

---

## ✨ Features

### Core Trading
- ✅ **Multi-Symbol Trading** - Supports 15+ cryptocurrency pairs
- ✅ **Advanced Technical Analysis** - EMA, RSI, ATR, Volume indicators
- ✅ **Hybrid Strategy** - Combines trend-following and breakout detection
- ✅ **Adaptive Position Sizing** - Dynamic position calculation based on account balance
- ✅ **Smart Stop Loss & Take Profit** - ATR-based dynamic protection

### Risk Management
- 🛡️ **Position Limits** - Per-symbol and total position control
- 🛡️ **Daily Loss Limits** - Automatic trading halt on threshold breach
- 🛡️ **Smart Safety Manager** - Multi-layer protection system
- 🛡️ **Kelly Criterion Support** - Optimal position sizing (optional)
- 🛡️ **TP Ladder** - Partial profit-taking at multiple levels
- 🛡️ **Trailing Stop** - Dynamic stop-loss adjustment

### Monitoring & Alerts
- 📱 **Telegram Integration** - Real-time trade notifications
- 📊 **Performance Tracking** - P&L, win rate, streaks monitoring
- 💓 **Heartbeat System** - Health monitoring and crash detection
- 📈 **Database Logging** - Complete trade history and metrics

### Technical
- 🐳 **Docker Containerized** - Easy deployment and scaling
- ⚙️ **Hot Configuration Reload** - Update settings without restart
- 🔄 **Session Management** - Automatic reconnection and error recovery
- 🗄️ **PostgreSQL/SQLite Support** - Flexible database options

---

## 💻 System Requirements

- **Docker** 20.10+
- **Docker Compose** 2.0+
- **Minimum RAM:** 512MB
- **Minimum Storage:** 1GB
- **OS:** Linux, macOS, or Windows (with WSL2)
- **Internet:** Stable connection required

---

## 📁 Project Structure

```
crypto_trading_bot/
│
├── 📄 README.md                    # This file
├── 📄 docker-compose.yml           # Docker orchestration
├── 📄 .env                         # Environment variables
├── 📄 .env.example                 # Example environment file
├── 📄 .gitignore                   # Git ignore rules
├── 📄 config_loader.py             # Configuration management
├── 📄 strategy_config.ini          # strategy config variables
├── 📄 BALANCE_GUIDE.md             # Balance Configuration Guide
│
├── 📁 bot/                         # Main bot application
│   ├── 📄 Dockerfile               # Bot container definition
│   ├── 📄 requirements.txt         # Python dependencies
│   ├── 📄 bot_main.py              # Main entry point
│   ├── 📄 sessions.py              # Bybit API wrapper - Trading sessions config
│   ├── 📄 trade_strategy.py        # Trading strategy logic
│   ├── 📄 risk_manager.py          # Risk management system
│   ├── 📄 order_manager.py         # Order execution & monitoring
│   ├── 📄 smart_safety.py          # Safety checks
│   ├── 📄 db_utils.py              # Database operations
│   ├── 📄 heartbeat_manager.py     # Health monitoring
│   ├── 📄 data_feeds.py            # Market data handling
│   ├── 📄 instruments.json         # Symbol configurations
│   └── 📄emergency_exit_manager.py # Advanced Protection System
│
├── 📁 watchdog/                    
│   ├── watchdog_config.yaml         
│   ├── Dockerfile                           
│   └── watchdog.py                
│
├── 📁 grafana/                        # Grafana section will be update in the next version
│   └── dashboards/                    
│       ├── monthly_calendar.json                   
│       └── realtime_activity.json    
│   └── provisioning/ 
│       └── dashboards/        
│           └── default.yml                   
│       └── datasources/                  
│           └── datasource.yml               
│
└── 📁 telegram/                   # Telegram bot
    ├── telegram_reporter.py       # Notification system
    ├── Dockerfile                 # Dockerfile
    └── requirements.txt           # Telegram Docker requirements
```

---

## 🚀 Installation

### Step 1: Clone Repository

```bash
# Clone from GitHub (replace with your repo URL)
git clone https://github.com/YOUR_USERNAME/crypto_trading_bot.git
cd crypto_trading_bot
```

### Step 2: Create Environment File

```bash
# Copy example environment file
cp .env.example .env

# Edit with your credentials
nano .env
```

**Required Environment Variables:**
```env
# Bybit API (Get from: https://www.bybit.com/app/user/api-management)
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
BYBIT_TESTNET=false                    # true for testnet, false for live

# Trading
ENABLE_LIVE_TRADING=false               # Set to true when ready for live trading

# Telegram (Get from: https://t.me/BotFather)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Database - postgres
POSTGRES_HOST=crypto_postgres_db
POSTGRES_PORT=5432
POSTGRES_USER=trader
POSTGRES_PASSWORD=trader_pass
POSTGRES_DB=trading_db

# Optional
LOG_LEVEL=INFO
TZ=UTC
```

### Step 3: Configure Strategy

Edit `strategy_config.ini` based on your account balance:

**For $5-10 balance:**
```ini
[risk_management]
POSITION_SIZE_USDT = 3.0
MIN_POSITION_USDT = 2.0
MAX_POSITION_USDT = 4.0
MAX_TOTAL_POSITIONS = 1

[trading]
SYMBOLS = DOGEUSDT,TRXUSDT,ADAUSDT,XRPUSDT
```

**For $50-100 balance:**
```ini
[risk_management]
POSITION_SIZE_USDT = 10.0
MIN_POSITION_USDT = 5.0
MAX_POSITION_USDT = 20.0
MAX_TOTAL_POSITIONS = 5

[trading]
SYMBOLS = SOLUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,DOGEUSDT,DOTUSDT,LINKUSDT,LTCUSDT,TRXUSDT,ATOMUSDT,FILUSDT
```

See [BALANCE_GUIDE.md](BALANCE_GUIDE.md) for complete configuration guide.

### Step 4: Build and Start

```bash
# Build Docker containers
docker compose up -d --build      

# Start in detached mode
docker-compose up -d

# View logs
docker-compose logs -f crypto_trading_bot
```

---

## ⚙️ Configuration

### Strategy Configuration (`strategy_config.ini`)

```ini
[enhanced_strategy]
# Technical Indicators
EMA_FAST = 9
EMA_SLOW = 21
EMA_LONG = 50
RSI_PERIOD = 14
ATR_PERIOD = 14

# Signal Filtering
MIN_SIGNAL_STRENGTH = 20
STRICTNESS_MODE = NORMAL        # LOOSE / NORMAL / STRICT
STRATEGY_STYLE = HYBRID         # HYBRID / TREND / BREAKOUT

# Volume
ENABLE_VOLUME_FILTER = true
MIN_VOLUME_RATIO = 0.15

# Stop Loss / Take Profit
SL_ATR_MULTIPLIER = 2.0
TP_ATR_MULTIPLIER = 2.0
```

### Risk Management (`config/strategy_config.ini`)

```ini
[risk_management]
# Position Sizing
POSITION_SIZE_USDT = 10.0
MIN_POSITION_USDT = 5.0
MAX_POSITION_USDT = 20.0

# Position Limits
MAX_POSITIONS_PER_SYMBOL = 1
MAX_TOTAL_POSITIONS = 5
POSITION_COOLDOWN_MINUTES = 5

# Risk Limits
MAX_DAILY_LOSS_PCT = 5.0
MAX_DRAWDOWN_PCT = 10.0
RISK_PER_TRADE_PCT = 1.0

# Advanced
USE_KELLY_CRITERION = false
TP_METHOD = ladder               # single / ladder
TP_LADDER_PCTS = 0.4,0.8,1.2
TRAILING_STOP_ENABLED = false
```

---

## 🎯 Usage

### Start Bot
```bash
docker-compose up -d
```

### View Logs
```bash
# All logs
docker-compose logs -f

# Only bot logs
docker-compose logs -f trading_bot

# Last 100 lines
docker-compose logs --tail=100 trading_bot
```

### Stop Bot
```bash
docker-compose down
```

### Restart Bot (after config changes)
```bash
docker-compose restart trading_bot
```

### Check Status
```bash
docker-compose ps
```

### Database Access
```bash
# PostgreSQL
docker-compose exec db psql -U trading_user -d trading

# SQLite
docker-compose exec trading_bot sqlite3 /app/db/trader.db
```

---

## 🛡️ Risk Management

### Position Size Guidelines

| Balance | Position Size | Max Positions | Risk Level |
|---------|--------------|---------------|------------|
| $5-10 | $2-4 | 1 | High (70%) |
| $10-25 | $5-8 | 2 | Medium-High (50%) |
| $25-50 | $5-12 | 3 | Medium (40%) |
| $50-100 | $5-20 | 5 | Moderate (35%) |
| $100-200 | $5-30 | 6 | Low (30%) |
| $200+ | $10-50 | 8 | Very Low (25%) |

**See full guide:** [docs/BALANCE_GUIDE.md](docs/BALANCE_GUIDE.md)

### Safety Features

1. **Pre-Trade Checks**
   - Balance verification
   - Position limits
   - Cooldown periods
   - Exchange minimum validation

2. **Active Monitoring**
   - Stop-loss enforcement
   - Take-profit ladder
   - Trailing stops
   - Daily loss limits

3. **Emergency Shutdown**
   - Automatic on max daily loss
   - Manual via Telegram
   - Signal handler (Ctrl+C)

---

## 📊 Monitoring

### Telegram Commands
```
/status      - Bot status and open positions
/balance     - Account balance and equity
/positions   - Detailed position information
/stats       - Performance statistics
/stop        - Emergency stop (close all positions)
/config      - Current configuration
```

### Key Metrics
- **Win Rate**: Percentage of profitable trades
- **Average P&L**: Average profit/loss per trade
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline

### Database Queries

**View recent trades:**
```sql
SELECT * FROM trade_executions ORDER BY opened_at DESC LIMIT 10;
```

**Calculate win rate:**
```sql
SELECT 
    COUNT(*) as total_trades,
    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
    ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate
FROM trade_executions 
WHERE status = 'closed';
```

---

## 🔧 Troubleshooting

### Bot Not Starting
```bash
# Check logs
docker-compose logs -f trading_bot

# Common issues:
# 1. Invalid API keys → Check .env file
# 2. Database connection → Check DATABASE_URL
# 3. Port conflicts → Check docker-compose.yml
```

### No Trades Executing
```bash
# Check if live trading is enabled
grep ENABLE_LIVE_TRADING .env

# Check balance vs position size
docker-compose logs crypto_trading_bot | grep "Balance too low"

# Verify strategy generates signals
docker-compose logs crypto_trading_bot | grep "Signal detected"
```

### Telegram Not Working
```bash
# Test bot token
curl https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getMe

# Check chat ID
docker-compose logs crypto_trading_bot | grep "Telegram"
```

### Database Issues
```bash
# Rebuild database
docker-compose down -v
docker-compose up -d

# Check schema
docker-compose exec db psql -U trading_user -d trading -c "\dt"
```

---

## 🔐 Security

### ⚠️ Critical Security Practices

1. **Use environment-specific API keys**
   - Testnet keys for testing
   - Live keys with IP whitelist only

2. **Restrict API permissions**
   - Enable: Trading, Account Info
   - Disable: Withdrawals, Sub-accounts

3. **Regular key rotation**
   ```bash
   # Rotate API keys monthly
   # Update .env and restart:
   docker-compose restart crypto_trading_bot
   ```

4. **Monitor API usage**
   - Check Bybit API logs regularly
   - Set up 2FA on exchange account

### Recommended Bybit API Settings
- ✅ Enable Trading
- ✅ Enable Read-Only Account Info
- ✅ Restrict by IP (add your server IP)
- ❌ Disable Withdrawals
- ❌ Disable Sub-account Transfer

---

## 📝 Maintenance

### Daily Tasks
- [ ] Check Telegram for trade notifications
- [ ] Review P&L and performance
- [ ] Verify bot is running: `docker-compose ps`

### Weekly Tasks
- [ ] Analyze win rate and adjust strategy
- [ ] Check database size: `du -sh db/`
- [ ] Review log files for errors
- [ ] Update balance-based config if needed

### Monthly Tasks
- [ ] Rotate API keys
- [ ] Backup database
- [ ] Update Docker images: `docker-compose pull`
- [ ] Review and optimize strategy parameters

---

## 🔄 Updates

### Update Bot Code
```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Update Dependencies
```bash
# Rebuild with new dependencies
docker-compose build --no-cache crypto_trading_bot
docker-compose up -d
```

---

## 📚 Additional Documentation

- [Balance Configuration Guide](BALANCE_GUIDE.md)

---

## ⚖️ Disclaimer

**⚠️ IMPORTANT: READ CAREFULLY**

This trading bot is provided for **educational and research purposes only**.

- ❌ **NOT FINANCIAL ADVICE** - This software does not constitute financial, investment, trading, or other advice.
- ⚠️ **HIGH RISK** - Cryptocurrency trading involves substantial risk of loss. Never trade with money you cannot afford to lose.
- 🔓 **NO WARRANTY** - This software is provided "as is" without warranty of any kind, express or implied.
- 💸 **POTENTIAL LOSSES** - You may lose some or all of your invested capital. Past performance does not guarantee future results.
- 👤 **YOUR RESPONSIBILITY** - You are solely responsible for determining whether any trading strategy is appropriate for you based on your personal investment objectives, financial circumstances, and risk tolerance.
- 🌍 **REGULATORY COMPLIANCE** - Ensure compliance with local laws and regulations regarding cryptocurrency trading.

**By using this software, you acknowledge that you understand and accept these risks.**

---

## 📄 License

⚠️ Private Project – Personal Use Only
This repository contains a private automated trading bot developed for personal testing and research purposes.
The code, configurations, and assets stored here are not intended for public distribution or commercial use.
Unauthorized access, reproduction, or redistribution of this code is strictly prohibited.

© 2025 Efi Hadar. All rights reserved.

---

## 🙏 Acknowledgments

- Bybit API Documentation
- Python TA-Lib Community
- Docker & Docker Compose
- PostgreSQL Team

---

**Made with ❤️ for automated trading**

*Last updated: November 2025*