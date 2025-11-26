import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# SETTINGS
# =========================
ticker = "NVDA"
start_date = "2019-01-01"
trade_amount = 10000  # We trade $500 per signal

# =========================
# DOWNLOAD DATA
# =========================
data = yf.download(ticker, start=start_date)
data.dropna(inplace=True)

# =========================
# CALCULATE 5-DAY RETURNS
# =========================
data["5d_return"] = data["Close"].pct_change(5)

# =========================
# HELPER: PRICE EXTRACTION
# =========================
def price_at(i):
    return float(data["Close"].iloc[i])

# =========================
# SIMULATION STORAGE
# =========================
cash = 0.0
equity_curve = []

drop_points = data[data["5d_return"] <= -0.10].index
gain_points = data[data["5d_return"] >= 0.10].index

drop_results = []
gain_results = []

# =========================
# 10% DROP → BUY $500 OF STOCK
# =========================
for idx in drop_points:
    start_i = data.index.get_loc(idx)
    end_i = start_i + 5

    if end_i < len(data):
        buy_price = price_at(start_i)
        sell_price = price_at(end_i)

        shares = trade_amount / buy_price

        drop_results.append(sell_price > buy_price)

        cash -= trade_amount
        cash += shares * sell_price

        equity_curve.append((data.index[end_i], cash))

drop_results = pd.Series(drop_results)

print(f"Occurrences of 10% weekly drops: {len(drop_results)}")
print(f"Probability of being higher 5 days later after drop: {drop_results.mean():.2%}")

# =========================
# 10% GAIN → SHORT $500 OF STOCK
# =========================
for idx in gain_points:
    start_i = data.index.get_loc(idx)
    end_i = start_i + 5

    if end_i < len(data):
        short_price = price_at(start_i)
        cover_price = price_at(end_i)

        shares = trade_amount / short_price

        gain_results.append(cover_price > short_price)

        cash += trade_amount            # proceeds from short sale
        cash -= shares * cover_price    # cost to cover

        equity_curve.append((data.index[end_i], cash))

gain_results = pd.Series(gain_results)

print(f"Occurrences of 10% weekly gains: {len(gain_results)}")
print(f"Probability of being higher 5 days later after gain: {gain_results.mean():.2%}")

# =========================
# FINAL RESULT
# =========================
print("\n=== FINAL SIMULATION RESULT ===")
print(f"Final cash balance: ${cash:,.2f}")

# =========================
# BUILD EQUITY CURVE DF
# =========================
equity_df = pd.DataFrame(equity_curve, columns=["Date", "Cash"])
equity_df.set_index("Date", inplace=True)

# =========================
# PRICE PLOT
# =========================
plt.figure(figsize=(14, 6))
plt.plot(data.index, data["Close"], label=f"{ticker} Price", linewidth=1)

# Drop lines
for idx in drop_points:
    i = data.index.get_loc(idx)
    end_i = i + 5
    if end_i < len(data):
        start_price = price_at(i)
        end_price = price_at(end_i)
        color = "green" if end_price > start_price else "red"
        plt.hlines(start_price, data.index[i], data.index[end_i], colors=color, linewidth=2)

# Gain lines
for idx in gain_points:
    i = data.index.get_loc(idx)
    end_i = i + 5
    if end_i < len(data):
        start_price = price_at(i)
        end_price = price_at(end_i)
        color = "blue" if end_price > start_price else "orange"
        plt.hlines(start_price, data.index[i], data.index[end_i], colors=color, linewidth=2)

plt.title(f"{ticker} – Price with 10% Weekly Events Marked")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# =========================
# EQUITY CURVE PLOT
# =========================
plt.figure(figsize=(14, 6))
plt.plot(equity_df.index, equity_df["Cash"], linewidth=2)
plt.title("Strategy Profit/Loss Over Time")
plt.xlabel("Date")
plt.ylabel("Profit / Loss ($)")
plt.grid(True)
plt.show()
