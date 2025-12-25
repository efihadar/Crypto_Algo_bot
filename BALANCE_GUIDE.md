# 📊 מדריך עדכון הגדרות הבוט לפי יתרה
## 🎯 כללי עדכון:
- **כל פעם שהיתרה עולה ב-50%** → בדוק אם צריך לעדכן
- **אחרי כל $20 נוספים** → שקול עדכון
- **אל תשכח להפעיל מחדש את הבוט** אחרי כל שינוי!
---

## 💰 יתרה: $5-10 (Micro Account)
### strategy_config.ini:
[risk_management]
POSITION_SIZE_USDT = 3.0
MIN_POSITION_USDT = 2.0
MAX_POSITION_USDT = 4.0
MAX_TOTAL_POSITIONS = 1
MAX_POSITIONS_PER_SYMBOL = 1
RISK_PER_TRADE_PCT = 5.0
MAX_DAILY_LOSS_PCT = 15.0

MAX_BALANCE_FRACTION = 0.70  # 70% עובד על אחוז מהיתרה

[trading]
SYMBOLS = DOGEUSDT,TRXUSDT,ADAUSDT,XRPUSDT
```
**מאפיינים:**
- פוזיציה אחת בלבד
- טריידים של $2-4
- רק 4 מטבעות זולים
- סיכון גבוה (70% ליתרה)
---

## 💰 יתרה: $10-25 (Small Account)
### strategy_config.ini:
[risk_management]
POSITION_SIZE_USDT = 5.0
MIN_POSITION_USDT = 5.0
MAX_POSITION_USDT = 8.0
MAX_TOTAL_POSITIONS = 2
MAX_POSITIONS_PER_SYMBOL = 1
RISK_PER_TRADE_PCT = 3.0
MAX_DAILY_LOSS_PCT = 10.0

MAX_BALANCE_FRACTION = 0.50  # 50% עובד על אחוז מהיתרה

[trading]
SYMBOLS = DOGEUSDT,TRXUSDT,ADAUSDT,XRPUSDT,ATOMUSDT,DOTUSDT,LINKUSDT,FILUSDT
```
**מאפיינים:**
- עד 2 פוזיציות בו-זמנית
- טריידים של $5-8
- 8 מטבעות זמינים
- סיכון בינוני (50%)
---

## 💰 יתרה: $25-50 (Medium Account)
### config/strategy_config.ini:
[risk_management]
POSITION_SIZE_USDT = 7.0
MIN_POSITION_USDT = 5.0
MAX_POSITION_USDT = 12.0
MAX_TOTAL_POSITIONS = 3
MAX_POSITIONS_PER_SYMBOL = 1
RISK_PER_TRADE_PCT = 2.0
MAX_DAILY_LOSS_PCT = 8.0

MAX_BALANCE_FRACTION = 0.40  # 40% עובד על אחוז מהיתרה

[trading]
SYMBOLS = DOGEUSDT,TRXUSDT,ADAUSDT,XRPUSDT,ATOMUSDT,DOTUSDT,LINKUSDT,FILUSDT,AVAXUSDT,NEARUSDT,APTUSDT
```
**מאפיינים:**
- עד 3 פוזיציות בו-זמנית
- טריידים של $5-12
- 11 מטבעות זמינים
- סיכון מבוקר (40%)
---

## 💰 יתרה: $50-100 (Standard Account)
### strategy_config.ini:
[risk_management]
POSITION_SIZE_USDT = 10.0
MIN_POSITION_USDT = 5.0
MAX_POSITION_USDT = 20.0
MAX_TOTAL_POSITIONS = 5
MAX_POSITIONS_PER_SYMBOL = 1
RISK_PER_TRADE_PCT = 1.0
MAX_DAILY_LOSS_PCT = 5.0

MAX_BALANCE_FRACTION = 0.35  # 35% עובד על אחוז מהיתרה

[trading]
SYMBOLS = SOLUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,DOGEUSDT,DOTUSDT,LINKUSDT,LTCUSDT,ZECUSDT,TRXUSDT,ATOMUSDT,UNIUSDT,FILUSDT,NEARUSDT,AAVEUSDT,APTUSDT
```
**מאפיינים:**
- עד 5 פוזיציות בו-זמנית
- טריידים של $5-20
- 16 מטבעות (כמעט הכל)
- סיכון סביר (35%)
---

## 💰 יתרה: $100-200 (Large Account)
### strategy_config.ini:
[risk_management]
POSITION_SIZE_USDT = 15.0
MIN_POSITION_USDT = 5.0
MAX_POSITION_USDT = 30.0
MAX_TOTAL_POSITIONS = 6
MAX_POSITIONS_PER_SYMBOL = 1
RISK_PER_TRADE_PCT = 0.8
MAX_DAILY_LOSS_PCT = 4.0

MAX_BALANCE_FRACTION = 0.30  # 30% עובד על אחוז מהיתרה

[trading]
SYMBOLS = SOLUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,DOGEUSDT,DOTUSDT,LINKUSDT,LTCUSDT,ZECUSDT,TRXUSDT,ATOMUSDT,MATICUSDT,UNIUSDT,FILUSDT,NEARUSDT,AAVEUSDT,APTUSDT
```
**מאפיינים:**
- עד 6 פוזיציות בו-זמנית
- טריידים של $5-30
- כל 17 המטבעות
- סיכון נמוך (30%)
---

## 💰 יתרה: $200-500 (Professional Account)
### strategy_config.ini:
[risk_management]
POSITION_SIZE_USDT = 25.0
MIN_POSITION_USDT = 10.0
MAX_POSITION_USDT = 50.0
MAX_TOTAL_POSITIONS = 8
MAX_POSITIONS_PER_SYMBOL = 1
RISK_PER_TRADE_PCT = 0.5
MAX_DAILY_LOSS_PCT = 3.0

MAX_BALANCE_FRACTION = 0.25  # 25% עובד על אחוז מהיתרה

[trading]
SYMBOLS = SOLUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,DOGEUSDT,DOTUSDT,LINKUSDT,LTCUSDT,ZECUSDT,TRXUSDT,ATOMUSDT,MATICUSDT,UNIUSDT,FILUSDT,NEARUSDT,AAVEUSDT,APTUSDT
```
**מאפיינים:**
- עד 8 פוזיציות בו-זמנית
- טריידים של $10-50
- כל 17 המטבעות
- סיכון מינימלי (25%)
---

## 💰 יתרה: $500+ (Advanced Account)
### strategy_config.ini:
[risk_management]
POSITION_SIZE_USDT = 50.0
MIN_POSITION_USDT = 20.0
MAX_POSITION_USDT = 100.0
MAX_TOTAL_POSITIONS = 10
MAX_POSITIONS_PER_SYMBOL = 2
RISK_PER_TRADE_PCT = 0.5
MAX_DAILY_LOSS_PCT = 2.0

MAX_BALANCE_FRACTION = 0.20  # 20% עובד על אחוז מהיתרה

[trading]
SYMBOLS = SOLUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,DOGEUSDT,DOTUSDT,LINKUSDT,LTCUSDT,ZECUSDT,TRXUSDT,ATOMUSDT,MATICUSDT,UNIUSDT,FILUSDT,NEARUSDT,AAVEUSDT,APTUSDT
```
**מאפיינים:**
- עד 10 פוזיציות בו-זמנית
- טריידים של $20-100
- כל המטבעות + 2 פוזיציות לסימבול
- סיכון מאוד נמוך (20%)
---

## 📋 טבלת השוואה מהירה

| יתרה | גודל טרייד | מקס' פוזיציות | מטבעות | סיכון % | Balance Fraction |
|------|-----------|---------------|---------|---------|------------------|
| $5-10 | $2-4 | 1 | 4 | גבוה | 70% |
| $10-25 | $5-8 | 2 | 8 | בינוני-גבוה | 50% |
| $25-50 | $5-12 | 3 | 11 | בינוני | 40% |
| $50-100 | $5-20 | 5 | 16 | סביר | 35% |
| $100-200 | $5-30 | 6 | 17 | נמוך | 30% |
| $200-500 | $10-50 | 8 | 17 | מאוד נמוך | 25% |
| $500+ | $20-100 | 10 | 17 | אולטרה נמוך | 20% |

---

## ⚠️ תזכורות חשובות:

### אחרי כל עדכון:
# 1. הפעל מחדש
docker-compose restart crypto_trading_bot

# 2. בדוק שהבוט רץ
docker-compose ps

# 3. עקוב אחרי הלוג
docker-compose logs -f crypto_trading_bot

# 4. וודא שאין שגיאות
# חפש את השורה: "✅ BOT INITIALIZATION COMPLETE"
```

### סימנים שהעדכון עבד:
```
✅ Position size calculated: $XX.XX  (הסכום השתנה)
✅ Processing X signals  (יותר סיגנלים)
✅ Open Positions: X  (יותר פוזיציות אפשריות)
```

### סימנים לבעיה:
```
❌ Balance too low for safe trade
❌ Skipping XXXUSDT: balance too low
⚠️ Not enough balance or risk conditions
```
---

## 📝 דוגמה: מעבר מ-$6 ל-$15
### 1. עדכן את הקונפיג:
# לפני
POSITION_SIZE_USDT = 3.0
MIN_POSITION_USDT = 2.0
MAX_POSITION_USDT = 4.0
MAX_TOTAL_POSITIONS = 1

# אחרי
POSITION_SIZE_USDT = 5.0
MIN_POSITION_USDT = 5.0
MAX_POSITION_USDT = 8.0
MAX_TOTAL_POSITIONS = 2
```

### 2. עדכן bot_main.py:
# לפני
max_balance_fraction = 0.70

# אחרי
max_balance_fraction = 0.50
```

### 3. הוסף מטבעות:
# לפני
SYMBOLS = DOGEUSDT,TRXUSDT,ADAUSDT,XRPUSDT

# אחרי
SYMBOLS = DOGEUSDT,TRXUSDT,ADAUSDT,XRPUSDT,ATOMUSDT,DOTUSDT,LINKUSDT,FILUSDT
```

### 4. הפעל מחדש:
docker-compose restart crypto_trading_bot
docker-compose logs -f crypto_trading_bot | grep "Position size calculated"
```
---

## 🎯 מתי לעדכן? (Checklist)

- [ ] היתרה עלתה ב-50% או יותר
- [ ] עברתי רף של $10, $25, $50, $100, $200, או $500
- [ ] הבוט חוסם הרבה סיגנלים בגלל "balance too low"
- [ ] אני רואה "Skipping XXXUSDT" על הרבה מטבעות
- [ ] עברו שבועיים מהעדכון האחרון ויש רווח
---