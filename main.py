import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ============================================================
# CONFIG — ตั้งค่าผ่าน Environment Variables ใน Railway
# ============================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID        = os.environ.get("CHAT_ID")
SYMBOL         = os.environ.get("SYMBOL", "BTCUSDT")
INTERVAL       = os.environ.get("INTERVAL", "5m")
CHECK_EVERY    = int(os.environ.get("CHECK_EVERY", "300"))  # วินาที (5 นาที)

# EMA Settings
EMA_FAST   = 20
EMA_MID    = 50
EMA_SLOW   = 200

# BOS Lookback
BOS_LOOKBACK = 20

# Risk (จำนวน $ ต่อ SL 1 unit)
TP_RATIOS = [1, 2, 3, 4, 5]

# ============================================================
# TELEGRAM
# ============================================================
def send_telegram(message: str):
    try:
        url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
        requests.post(url, data=data, timeout=10)
        print(f"[Telegram] Sent: {message[:60]}...")
    except Exception as e:
        print(f"[Telegram] Error: {e}")

# ============================================================
# BINANCE API — ดึง OHLCV
# ============================================================
def get_ohlcv(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    url    = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp   = requests.get(url, params=params, timeout=10)
    data   = resp.json()

    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","tbbav","tbqav","ignore"
    ])
    df["open"]  = df["open"].astype(float)
    df["high"]  = df["high"].astype(float)
    df["low"]   = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"]= df["volume"].astype(float)
    df["time"]  = pd.to_datetime(df["open_time"], unit="ms")
    return df.reset_index(drop=True)

# ============================================================
# INDICATORS
# ============================================================
def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_pivot_high(df: pd.DataFrame, lookback: int) -> pd.Series:
    """หา swing high"""
    ph = pd.Series(np.nan, index=df.index)
    for i in range(lookback, len(df) - lookback):
        window = df["high"].iloc[i - lookback: i + lookback + 1]
        if df["high"].iloc[i] == window.max():
            ph.iloc[i] = df["high"].iloc[i]
    return ph

def calc_pivot_low(df: pd.DataFrame, lookback: int) -> pd.Series:
    """หา swing low"""
    pl = pd.Series(np.nan, index=df.index)
    for i in range(lookback, len(df) - lookback):
        window = df["low"].iloc[i - lookback: i + lookback + 1]
        if df["low"].iloc[i] == window.min():
            pl.iloc[i] = df["low"].iloc[i]
    return pl

def analyze(df: pd.DataFrame) -> dict:
    """คำนวณ indicators ทั้งหมด แล้ว return signal"""

    # ต้องมีข้อมูลเพียงพอก่อน
    if len(df) < EMA_SLOW + 10:
        raise ValueError(f"Not enough data: {len(df)} bars")

    # EMAs
    df["ema_fast"] = calc_ema(df["close"], EMA_FAST)
    df["ema_mid"]  = calc_ema(df["close"], EMA_MID)
    df["ema_slow"] = calc_ema(df["close"], EMA_SLOW)

    # Trend
    last = df.iloc[-1]
    trend_up   = last["ema_fast"] > last["ema_mid"] > last["ema_slow"]
    trend_down = last["ema_fast"] < last["ema_mid"] < last["ema_slow"]

    # Pivot High/Low
    df["ph"] = calc_pivot_high(df, BOS_LOOKBACK)
    df["pl"] = calc_pivot_low(df, BOS_LOOKBACK)

    # หา last valid pivot — ข้าม BOS_LOOKBACK แท่งท้ายสุดที่ยังไม่ confirm
    ph_series = df["ph"].iloc[:-BOS_LOOKBACK].dropna()
    pl_series = df["pl"].iloc[:-BOS_LOOKBACK].dropna()

    last_ph = float(ph_series.iloc[-1]) if len(ph_series) > 0 else None
    last_pl = float(pl_series.iloc[-1]) if len(pl_series) > 0 else None

    prev_close = float(df["close"].iloc[-2])
    curr_close = float(df["close"].iloc[-1])

    # BOS Detection
    bos_bull = (last_ph is not None) and (curr_close > last_ph) and (prev_close <= last_ph)
    bos_bear = (last_pl is not None) and (curr_close < last_pl) and (prev_close >= last_pl)

    # Supply/Demand Zone
    supply_zone = last_ph
    demand_zone = last_pl

    # Signals
    bull_candle = curr_close > float(df["open"].iloc[-1])
    bear_candle = curr_close < float(df["open"].iloc[-1])

    buy_signal  = trend_up   and bos_bull and bull_candle
    sell_signal = trend_down and bos_bear and bear_candle

    # SL / TP (ใช้ ATR)
    atr = float((df["high"] - df["low"]).rolling(14).mean().iloc[-1])
    sl_dist = atr * 1.5

    sl_buy   = curr_close - sl_dist
    sl_sell  = curr_close + sl_dist
    tps_buy  = [curr_close + sl_dist * r for r in TP_RATIOS]
    tps_sell = [curr_close - sl_dist * r for r in TP_RATIOS]

    return {
        "price":       curr_close,
        "trend":       "Bullish" if trend_up else "Bearish" if trend_down else "Neutral",
        "bos_bull":    bos_bull,
        "bos_bear":    bos_bear,
        "buy_signal":  buy_signal,
        "sell_signal": sell_signal,
        "sl_buy":      sl_buy,
        "sl_sell":     sl_sell,
        "tps_buy":     tps_buy,
        "tps_sell":    tps_sell,
        "supply_zone": supply_zone,
        "demand_zone": demand_zone,
        "time":        df["time"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC"),
    }

# ============================================================
# MESSAGE BUILDER
# ============================================================
def build_message(signal_type: str, result: dict) -> str:
    p = result["price"]

    if signal_type == "BUY":
        emoji = "🟢"
        tps   = result["tps_buy"]
        sl    = result["sl_buy"]
    elif signal_type == "SELL":
        emoji = "🔴"
        tps   = result["tps_sell"]
        sl    = result["sl_sell"]
    elif signal_type == "BOS_BULL":
        return (
            f"🔷 <b>BOS Bullish — {SYMBOL}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💰 Price     : <b>{p:,.2f}</b>\n"
            f"⏱ Time      : {result['time']}\n"
            f"📊 Trend     : {result['trend']}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"⚡ Structure Break ขึ้น — รอ Pullback เข้า Demand"
        )
    elif signal_type == "BOS_BEAR":
        return (
            f"🔶 <b>BOS Bearish — {SYMBOL}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💰 Price     : <b>{p:,.2f}</b>\n"
            f"⏱ Time      : {result['time']}\n"
            f"📊 Trend     : {result['trend']}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"⚡ Structure Break ลง — รอ Pullback เข้า Supply"
        )
    else:
        return ""

    return (
        f"{emoji} <b>SMC Signal — {signal_type} {SYMBOL}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"⏱ Time      : {result['time']}\n"
        f"💰 Price     : <b>{p:,.2f}</b>\n"
        f"📊 Trend     : {result['trend']}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"🛑 SL        : {sl:,.2f}\n"
        f"🎯 TP1       : {tps[0]:,.2f}\n"
        f"🎯 TP2       : {tps[1]:,.2f}\n"
        f"🎯 TP3       : {tps[2]:,.2f}\n"
        f"🎯 TP4       : {tps[3]:,.2f}\n"
        f"🎯 TP5       : {tps[4]:,.2f}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"⚡ Powered by SMC Bot"
    )

# ============================================================
# MAIN LOOP
# ============================================================
def main():
    print(f"🚀 SMC Bot Started — {SYMBOL} {INTERVAL}")
    send_telegram(f"🚀 <b>SMC Bot เริ่มทำงานแล้ว!</b>\nSymbol: {SYMBOL}\nTimeframe: {INTERVAL}\nรอ Signal อยู่นะครับ...")

    # State — ป้องกันแจ้งเตือนซ้ำ
    last_signal = {"buy": False, "sell": False, "bos_bull": False, "bos_bear": False}

    while True:
        try:
            print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Checking {SYMBOL}...")
            df     = get_ohlcv(SYMBOL, INTERVAL, limit=300)
            result = analyze(df)

            print(f"  Price: {result['price']:,.2f} | Trend: {result['trend']} | BOS Bull: {result['bos_bull']} | BOS Bear: {result['bos_bear']} | Buy: {result['buy_signal']} | Sell: {result['sell_signal']}")

            # BOS Bull
            if result["bos_bull"] and not last_signal["bos_bull"]:
                send_telegram(build_message("BOS_BULL", result))
                last_signal["bos_bull"] = True
            elif not result["bos_bull"]:
                last_signal["bos_bull"] = False

            # BOS Bear
            if result["bos_bear"] and not last_signal["bos_bear"]:
                send_telegram(build_message("BOS_BEAR", result))
                last_signal["bos_bear"] = True
            elif not result["bos_bear"]:
                last_signal["bos_bear"] = False

            # Buy Signal
            if result["buy_signal"] and not last_signal["buy"]:
                send_telegram(build_message("BUY", result))
                last_signal["buy"] = True
            elif not result["buy_signal"]:
                last_signal["buy"] = False

            # Sell Signal
            if result["sell_signal"] and not last_signal["sell"]:
                send_telegram(build_message("SELL", result))
                last_signal["sell"] = True
            elif not result["sell_signal"]:
                last_signal["sell"] = False

        except Exception as e:
            print(f"[ERROR] {e}")

        time.sleep(CHECK_EVERY)

if __name__ == "__main__":
    main()
