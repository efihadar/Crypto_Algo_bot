class PumpDetector:
    def __init__(self):
        self.thresholds = {
            'volume_spike': 3.0,  # פי 3 מה-volume הממוצע
            'price_spike': 20.0,   # עלייה של 20% ב-2 שעות
            'min_volume': 1000000, # לפחות $1M volume
        }
    
    def detect_pump(self, df: pd.DataFrame) -> Dict:
        """זיהוי פאמפ פוטנציאלי"""
        if len(df) < 50:
            return {}
        
        # 1. בדיקת volume spike
        avg_volume = df["volume"].rolling(20).mean().iloc[-1]
        current_volume = df["volume"].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # 2. בדיקת מחיר
        price_2h_ago = df.iloc[-8]["close"] 
        current_price = df.iloc[-1]["close"]
        price_change = ((current_price - price_2h_ago) / price_2h_ago) * 100
        
        # 3. זיהוי פאמפ
        is_pump = (
            volume_ratio >= self.thresholds['volume_spike'] and
            price_change >= self.thresholds['price_spike'] and
            current_volume >= self.thresholds['min_volume']
        )
        
        if is_pump:
            return {
                'is_pump': True,
                'volume_ratio': volume_ratio,
                'price_change': price_change,
                'strength': min(100, price_change * 2)  # חוזק האות
            }
        
        return {'is_pump': False}