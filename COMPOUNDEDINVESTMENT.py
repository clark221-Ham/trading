import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# SETTINGS
# ==============================
symbol = "NVDA"
start = "2015-01-01"
end = "2026-01-01"
starting_capital = 20000
atr_window = 14           # ATR period for volatility/trailing stop
stop_loss_atr = 2         # Sell if price drops 2*ATR from peak

# ==============================
# DOWNLOAD DATA
# ==============================
data = yf.download(symbol, start=start, end=end, auto_adjust=True)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)
data["Close"] = data["Close"].astype(float)

# ==============================
# INDICATORS
# ==============================
# Faster EMAs
data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()
data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean()

# ATR for trailing stop
data["High-Low"] = data["High"] - data["Low"]
data["High-ClosePrev"] = (data["High"] - data["Close"].shift(1)).abs()
data["Low-ClosePrev"] = (data["Low"] - data["Close"].shift(1)).abs()
data["TR"] = data[["High-Low", "High-ClosePrev", "Low-ClosePrev"]].max(axis=1)
data["ATR"] = data["TR"].rolling(atr_window).mean()

# ==============================
# STRATEGY STATE
# ==============================
capital = starting_capital
position_size = 0
peak_price = 0           # track peak for trailing stop
account_values = []
buy_points = []
sell_points = []

# ==============================
# MAIN SIMULATION LOOP
# ==============================
for i in range(1, len(data)):
    price_today = data["Close"].iloc[i]
    ema20 = data["EMA20"].iloc[i]
    ema50 = data["EMA50"].iloc[i]
    atr = data["ATR"].iloc[i]

    # Skip if indicators not ready
    if pd.isna(ema20) or pd.isna(ema50) or pd.isna(atr):
        account_values.append(capital + position_size * price_today)
        continue

    # ----------------------
    # BUY SIGNAL
    # ----------------------
    if ema20 > ema50 and capital > 0:
        # Full exposure in uptrend
        invest_amount = capital
        shares_bought = invest_amount / price_today
        capital -= invest_amount
        position_size += shares_bought
        peak_price = price_today  # reset peak
        buy_points.append((data.index[i], price_today))

    # ----------------------
    # SELL SIGNAL — Trailing stop
    # ----------------------
    if position_size > 0:
        # Update peak price while invested
        peak_price = max(peak_price, price_today)
        trailing_stop_price = peak_price - stop_loss_atr * atr

        if price_today < trailing_stop_price:
            # Sell entire position on stop hit
            capital += position_size * price_today
            sell_points.append((data.index[i], price_today))
            position_size = 0
            peak_price = 0

    # Track daily account value
    account_values.append(capital + position_size * price_today)

# ==============================
# FINAL RESULTS
# ==============================
final_value = capital + position_size * data["Close"].iloc[-1]

print(f"Starting Capital: {starting_capital}")
print(f"Ending Value: {final_value:.2f}")
print(f"Profit: {final_value - starting_capital:.2f}")
print(f"Final Cash: {capital:.2f}")
print(f"Final Shares Held: {position_size:.4f}")
print(f"Final Share Value: {position_size * data['Close'].iloc[-1]:.2f}")

# ==============================
# PERFORMANCE COMPARISON
# ==============================
strategy_return_pct = (final_value - starting_capital) / starting_capital * 100
start_price = data["Close"].iloc[0]
end_price = data["Close"].iloc[-1]
buy_hold_return_pct = (end_price - start_price) / start_price * 100

print("\n=== PERFORMANCE COMPARISON ===")
print(f"Strategy Return:     {strategy_return_pct:.2f}%")
print(f"Buy & Hold Return:   {buy_hold_return_pct:.2f}%")
if strategy_return_pct > buy_hold_return_pct:
    print("\n✔ Your strategy beat the market.")
else:
    print("\n✖ Buy & hold would have performed better.")

# ==============================
# PLOT 1 — PRICE + TRADES
# ==============================
plt.figure(figsize=(14, 7))
plt.plot(data.index, data["Close"], label="Price")
if buy_points:
    plt.scatter(*zip(*buy_points), marker="^", color="green", label="Buy", s=80)
if sell_points:
    plt.scatter(*zip(*sell_points), marker="v", color="red", label="Sell", s=80)
plt.title(f"{symbol} — Price with Trades")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# ==============================
# PLOT 2 — EQUITY CURVE
# ==============================
plt.figure(figsize=(14, 5))
plt.plot(data.index[len(data) - len(account_values):], account_values, label="Account Value")
plt.title("Equity Curve")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.show()
