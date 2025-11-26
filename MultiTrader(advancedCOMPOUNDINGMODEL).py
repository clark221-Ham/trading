# MultiTrader (ATR sizing + SPY SMA200 regime filter + Max position size per trade)
# + Optimizer (short & long locked; sweep medium) -> writes results to CSV
# Requirements: yfinance, pandas, numpy, matplotlib
import sys
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================
# SETTINGS
# ==============================
tickers = [
    "AAPL", "MMM", "CVX", "NVDA",
    "TGT", "ORCL", "MSFT", "VZ", "ADBE",
    "PFE", "MET", "COP", "HON", "INTC",
    "SBUX", "MDT", "AXP", "AVGO", "CAT"
]





start = "2005-03-01"
end   = "2015-03-01"



starting_capital = 100000.0

# Default EMAs (used if you run the main simulation directly)
ema_short_span = 90
ema_med_span = 100
ema_long_span = 120

# ATR / stops / sizing
atr_window = 14
stop_loss_atr = 8.5
min_holding_bars_before_stop = 8

# POSITION SIZING
position_sizing_mode = "atr_risk"  # "fixed_allocation" or "atr_risk"
allocation_per_ticker = starting_capital / max(1, len(tickers))
risk_per_trade = 0.02                # fraction of current equity risked per trade

# Market regime filter
use_market_filter = True
spy_sma_window = 200

# Execution realism
slippage = 0.0005
commission = 0.0

# ==============================
# RULE A: MAX POSITION SIZE (per trade)
# ==============================
# Maximum fraction of total equity allowed in a single ticker (e.g., 0.05 = 5%)
max_position_size = 0.08

# ==============================
# OPTIMIZER TOGGLE & RANGES
# ==============================
run_optimizer = False   # set False to run normal simulation

# Optimizer ranges
short_ema  = 70            # locked short
medium_range = [100, 110, 120] # values to test
long_ema   = 150           # locked long


# ==============================
# STATE (placeholders for main run)
# ==============================
portfolio_cash = float(starting_capital)
positions = {t: 0.0 for t in tickers}
peak_prices = {t: np.nan for t in tickers}
entry_date = {t: None for t in tickers}
entry_atr = {t: np.nan for t in tickers}
account_values = []
equity_index = []
trade_log = []
buy_points = {t: [] for t in tickers}
sell_points = {t: [] for t in tickers}
last_real_trade_date = {}

# ==============================
# DOWNLOAD DATA (tickers + SPY)
# ==============================
all_symbols = tickers + ["SPY"]
raw = {}
print("Downloading data...")
for sym in all_symbols:
    df = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
    raw[sym] = df if not df.empty else None

# ==============================
# PREPARE data_dict with indicators and last-real detection
# ==============================
data_dict = {}
for sym in all_symbols:
    df = raw[sym]
    if df is None:
        data_dict[sym] = None
        last_real_trade_date[sym] = pd.Timestamp(start)
        continue

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df.astype({"Close": float, "High": float, "Low": float, "Open": float})
    df.index = pd.to_datetime(df.index)

    # Compute ATR once (shifted by 1 to avoid lookahead)
    df["HL"] = df["High"] - df["Low"]
    df["HC"] = (df["High"] - df["Close"].shift(1)).abs()
    df["LC"] = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = df[["HL", "HC", "LC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(atr_window).mean().shift(1)

    data_dict[sym] = df.copy()
    last_real_trade_date[sym] = df.index[-1]

# Ensure SPY has SMA200 column (compute on its df)
if data_dict.get("SPY") is not None:
    spy_df_local = data_dict["SPY"]
    spy_df_local[f"SMA{spy_sma_window}"] = spy_df_local["Close"].rolling(spy_sma_window).mean().shift(1)
    data_dict["SPY"] = spy_df_local

# ==============================
# ALIGN DATES (calendar) and forward-fill up to last real
# ==============================
all_dates = pd.date_range(start=start, end=end, freq="D")
for sym in all_symbols:
    df = data_dict[sym]
    if df is None:
        empty_df = pd.DataFrame(index=all_dates, data={
            "Close": np.nan, "High": np.nan, "Low": np.nan, "Open": np.nan
        })
        empty_df["ATR"] = np.nan
        if sym == "SPY":
            empty_df[f"SMA{spy_sma_window}"] = np.nan
        data_dict[sym] = empty_df
        continue

    df_re = df.reindex(all_dates)
    last_real = last_real_trade_date[sym]
    # forward-fill/backfill up to last_real only
    df_re.loc[:last_real] = df_re.loc[:last_real].ffill().bfill()
    data_dict[sym] = df_re

# ==============================
# HELPERS
# ==============================
def get_val_safe(df, date, col):
    """Return last valid value for col on or before date, or np.nan if none."""
    try:
        v = df.loc[date, col]
    except Exception:
        return np.nan
    if pd.isna(v):
        s = df[col].loc[:date].ffill()
        if s.empty:
            return np.nan
        return float(s.iloc[-1])
    return float(v)

def get_last_close_before(df, date):
    try:
        s = df["Close"].loc[:date].ffill()
        if s.empty:
            return np.nan
        return float(s.iloc[-1])
    except Exception:
        return np.nan

# ==============================
# SIMULATOR FUNCTION (encapsulates main loop for one EMA triplet)
# ==============================
def simulate_for_emas(short_span, med_span, long_span, verbose=False):
    """
    Run the full backtest using the provided EMA spans.
    Returns: equity_curve (pd.Series), final_value (float), total_trades (int)

    Side-effect: populates globals last_run_equity_series and last_run_position_series
    so you can plot allocation after a run without changing existing call sites.
    """
    global last_run_equity_series, last_run_position_series

    # Local state (so we don't modify global variables)
    cash = float(starting_capital)
    pos = {t: 0.0 for t in tickers}
    peak = {t: np.nan for t in tickers}
    entry_dt = {t: None for t in tickers}
    entry_atr_local = {t: np.nan for t in tickers}
    acct_vals = []
    acct_idx = []
    trades = []
    buy_pts = {t: [] for t in tickers}
    sell_pts = {t: [] for t in tickers}
    equity_over_time = []
    position_value_over_time = []

    # Will collect the daily position-only value (for allocation plotting)
    pos_vals = []

    # Precompute EMAs inside local copies
    local_data = {}
    for sym in all_symbols:
        df = data_dict[sym].copy()
        # compute requested EMAs (shifted by 1 to avoid lookahead)
        if sym != "SPY":  # for tickers
            df[f"EMA_S"] = df["Close"].ewm(span=short_span, adjust=False).mean().shift(1)
            df[f"EMA_M"] = df["Close"].ewm(span=med_span, adjust=False).mean().shift(1)
            df[f"EMA_L"] = df["Close"].ewm(span=long_span, adjust=False).mean().shift(1)
        else:
            # For SPY compute SMA regime (already precomputed but ensure existence)
            df[f"SMA{spy_sma_window}"] = df.get(f"SMA{spy_sma_window}", df["Close"].rolling(spy_sma_window).mean().shift(1))
        local_data[sym] = df

    # Main loop across calendar dates
    for date in all_dates:
        # process all tickers: update cash/positions based on trading rules
        for ticker in tickers:
            df = local_data[ticker]
            price_today = get_val_safe(df, date, "Close")
            ema_short = get_val_safe(df, date, "EMA_S")
            ema_med = get_val_safe(df, date, "EMA_M")
            ema_long = get_val_safe(df, date, "EMA_L")
            atr = get_val_safe(df, date, "ATR")

            # delisting handling: if date > last_real then force liquidation at last_real
            last_real = last_real_trade_date.get(ticker, None)
            if last_real is None:
                continue

            if date > last_real:
                if pos[ticker] > 0:
                    last_price = get_last_close_before(df, last_real)
                    if not pd.isna(last_price):
                        exec_price = last_price * (1 - slippage)
                        proceeds = pos[ticker] * exec_price
                        cash += proceeds
                        cash = max(cash - commission, 0.0)
                        trades.append(("DELISTED_EXIT", ticker, last_real, pos[ticker], exec_price))
                    pos[ticker] = 0.0
                    peak[ticker] = np.nan
                    entry_dt[ticker] = None
                    entry_atr_local[ticker] = np.nan
                continue

            # if price_today is NaN (rare before last_real), skip trading for this ticker today
            if pd.isna(price_today):
                continue

            # ENTRY condition
            indicators_ready = not (pd.isna(ema_short) or pd.isna(ema_med) or pd.isna(ema_long) or pd.isna(atr))
            if indicators_ready and ema_short > ema_med > ema_long and pos[ticker] == 0:
                # enforce market regime filter (only allow buys when SPY is above its SMA200)
                spy_df_local = local_data["SPY"]
                spy_close = get_val_safe(spy_df_local, date, "Close")
                spy_sma200 = get_val_safe(spy_df_local, date, f"SMA{spy_sma_window}")
                market_ok = not (pd.isna(spy_close) or pd.isna(spy_sma200)) and (spy_close > spy_sma200)
                if not (use_market_filter and not market_ok):
                    # sizing
                    shares_bought = 0.0
                    invest_amount = 0.0
                    exec_price = price_today * (1 + slippage)

                    if position_sizing_mode == "fixed_allocation":
                        invest_amount = min(allocation_per_ticker, cash)
                        # enforce max position cap (Rule A)
                        equity_est = cash + sum(pos[t] * get_last_close_before(local_data[t], date) for t in tickers)
                        max_allowed_for_position = equity_est * max_position_size
                        invest_amount = min(invest_amount, max_allowed_for_position)
                        shares_bought = invest_amount / exec_price if exec_price > 0 else 0.0

                    elif position_sizing_mode == "atr_risk":
                        if pd.isna(atr) or atr <= 0 or exec_price <= 0:
                            shares_bought = 0.0
                            invest_amount = 0.0
                        else:
                            equity_est = cash + sum(pos[t] * get_last_close_before(local_data[t], date) for t in tickers)
                            risk_dollars = equity_est * risk_per_trade
                            distance = max(atr * stop_loss_atr, 1e-8)
                            desired_shares = risk_dollars / distance
                            desired_invest = desired_shares * exec_price
                            desired_invest = min(desired_invest, cash)
                            max_allowed_for_position = equity_est * max_position_size
                            invest_amount = min(desired_invest, max_allowed_for_position)
                            shares_bought = invest_amount / exec_price if exec_price > 0 else 0.0
                    else:
                        raise ValueError("Unknown position_sizing_mode")

                    invest_amount = min(invest_amount, cash)
                    if shares_bought > 0 and invest_amount > 0:
                        pos[ticker] += shares_bought
                        cash -= invest_amount
                        cash = max(cash - commission, 0.0)
                        peak[ticker] = price_today
                        entry_dt[ticker] = date
                        entry_atr_local[ticker] = atr if not pd.isna(atr) else 0.0
                        buy_pts[ticker].append((date, price_today))
                        trades.append(("BUY", ticker, date, shares_bought, exec_price, invest_amount))

            # EXIT: ATR trailing stop (frozen ATR at entry)
            if pos[ticker] > 0:
                peak[ticker] = max(peak[ticker], price_today) if not pd.isna(peak[ticker]) else price_today
                allow_stop = True
                if entry_dt[ticker] is not None:
                    days_held = (date - entry_dt[ticker]).days
                    allow_stop = days_held >= min_holding_bars_before_stop
                frozen = entry_atr_local[ticker] if not pd.isna(entry_atr_local[ticker]) else atr
                trailing_stop_price = peak[ticker] - stop_loss_atr * (frozen if not pd.isna(frozen) else 0.0)
                if allow_stop and (not pd.isna(trailing_stop_price)) and price_today < trailing_stop_price:
                    exec_price = price_today * (1 - slippage)
                    proceeds = pos[ticker] * exec_price
                    cash += proceeds
                    cash = max(cash - commission, 0.0)
                    sell_pts[ticker].append((date, price_today))
                    trades.append(("SELL", ticker, date, pos[ticker], exec_price, proceeds))
                    pos[ticker] = 0.0
                    peak[ticker] = np.nan
                    entry_dt[ticker] = None
                    entry_atr_local[ticker] = np.nan

        # After processing all tickers for the day, compute end-of-day total value
        # Use last known closes (robust to NaN) for valuation
        total_value = cash
        # Track allocation values for plotting later
        position_value_today = 0
        for t in tickers:
            last_price = get_last_close_before(local_data[t], date)
            if not pd.isna(last_price):
                position_value_today += pos[t] * last_price

        equity_over_time.append(total_value)
        position_value_over_time.append(position_value_today)

        position_value = 0.0
        for t in tickers:
            last_price = get_last_close_before(local_data[t], date)
            if not pd.isna(last_price):
                position_value += pos[t] * last_price
                total_value += pos[t] * last_price

        acct_vals.append(float(total_value))
        pos_vals.append(float(position_value))
        acct_idx.append(date)

    # FINAL FORCE-UPDATE
    last_date = all_dates[-1]
    final_positions_value = 0.0
    for t in tickers:
        last_price = get_last_close_before(local_data[t], last_date)
        if pd.isna(last_price):
            last_price = 0.0
        final_positions_value += pos[t] * float(last_price)
    final_value_local = cash + final_positions_value

    # ensure last item aligns
    if len(acct_vals) == 0:
        acct_vals.append(final_value_local)
        acct_idx.append(last_date)
        pos_vals.append(final_positions_value)
    else:
        acct_vals[-1] = float(final_value_local)
        pos_vals[-1] = float(final_positions_value)
        acct_idx[-1] = last_date

    equity_series = pd.Series(acct_vals, index=acct_idx).ffill()
    position_series = pd.Series(pos_vals, index=acct_idx).ffill()

    # store last-run series in globals so caller can plot allocation easily
    last_run_equity_series = equity_series
    last_run_position_series = position_series

    # keep return signature compatible with optimizer/main code
    return equity_series, position_series, final_value_local, len(trades)




# ==============================
# OPTIMIZER: sweep medium EMA (short/long locked) -> save CSV and exit
# ==============================
if run_optimizer:
    print("Running EMA optimizer (short locked, long locked; sweeping medium)...")
    years_total = (all_dates[-1] - all_dates[0]).days / 365.25

    total_runs = len(medium_range)
    print(f"Total combinations to run (medium only): {total_runs}")

    results = []
    run_count = 0

    for medium_ema in medium_range:
        run_count += 1
        if run_count % 5 == 0 or run_count == total_runs:
            print(f"Starting medium {medium_ema} ({run_count}/{total_runs})...")

        equity_ser, final_val, trades_count = simulate_for_emas(short_ema, medium_ema, long_ema)

        # --- CAGR calculation ---
        try:
            start_eq = float(equity_ser.iloc[0])
            end_eq   = float(equity_ser.iloc[-1])
            if start_eq <= 0 or end_eq <= 0:
                cagr = np.nan
            else:
                cagr = (end_eq / start_eq) ** (1.0 / years_total) - 1.0
        except Exception:
            cagr = np.nan

        results.append([short_ema, medium_ema, long_ema, cagr, final_val, trades_count])

    # --- Save results ---
    results_df = pd.DataFrame(
        results,
        columns=['ShortEMA', 'MediumEMA', 'LongEMA', 'CAGR', 'FinalValue', 'NumTrades']
    )
    results_df.to_csv("ema3_results.csv", index=False)
    print("Optimization complete. Results saved to ema3_results.csv")

    # --- Display top results ---
    top = results_df.sort_values('CAGR', ascending=False).head(10)
    print("Top by CAGR:")
    print(top.to_string(index=False))

    sys.exit(0)

# ==============================
# If optimizer is disabled, run your main single-parameter simulation
# ==============================
print("Running main simulation with EMA spans:", ema_short_span, ema_med_span, ema_long_span)

# Reset state for main run
portfolio_cash = float(starting_capital)
positions = {t: 0.0 for t in tickers}
peak_prices = {t: np.nan for t in tickers}
entry_date = {t: None for t in tickers}
entry_atr = {t: np.nan for t in tickers}
account_values = []
equity_index = []
trade_log = []
buy_points = {t: [] for t in tickers}
sell_points = {t: [] for t in tickers}

# Recompute EMA columns in data_dict for the chosen spans (shifted)
for sym in all_symbols:
    df = data_dict[sym]
    if sym != "SPY":
        df[f"EMA{ema_short_span}"] = df["Close"].ewm(span=ema_short_span, adjust=False).mean().shift(1)
        df[f"EMA{ema_med_span}"] = df["Close"].ewm(span=ema_med_span, adjust=False).mean().shift(1)
        df[f"EMA{ema_long_span}"] = df["Close"].ewm(span=ema_long_span, adjust=False).mean().shift(1)
    data_dict[sym] = df

spy_df = data_dict["SPY"]

# (Main loop follows same logic as before)
# NOTE: valuation happens AFTER processing all tickers for a date (to avoid double-counting spikes)
for date in all_dates:
    # compute market regime once per day (SPY using SMA200)
    spy_close = get_val_safe(spy_df, date, "Close")
    spy_sma200 = get_val_safe(spy_df, date, f"SMA{spy_sma_window}")
    market_ok = not (pd.isna(spy_close) or pd.isna(spy_sma200)) and (spy_close > spy_sma200)

    # compute current equity estimate for ATR sizing decisions (use last-known values)
    # do not compute total_value incrementally; update portfolio_cash and positions during loop
    for ticker in tickers:
        df = data_dict[ticker]
        price_today = get_val_safe(df, date, "Close")
        ema_short = get_val_safe(df, date, f"EMA{ema_short_span}")
        ema_med = get_val_safe(df, date, f"EMA{ema_med_span}")
        ema_long = get_val_safe(df, date, f"EMA{ema_long_span}")
        atr = get_val_safe(df, date, "ATR")

        # (same trading logic as previous main loop)
        last_real = last_real_trade_date.get(ticker, None)
        if last_real is None:
            continue

        if date > last_real:
            if positions[ticker] > 0:
                last_price = get_last_close_before(df, last_real)
                if not pd.isna(last_price):
                    exec_price = last_price * (1 - slippage)
                    proceeds = positions[ticker] * exec_price
                    portfolio_cash += proceeds
                    portfolio_cash = max(portfolio_cash - commission, 0.0)
                    trade_log.append({
                        "date": last_real, "ticker": ticker, "type": "DELISTED_EXIT",
                        "shares": positions[ticker], "price": exec_price, "cash_after": portfolio_cash
                    })
                positions[ticker] = 0.0
                peak_prices[ticker] = np.nan
                entry_date[ticker] = None
                entry_atr[ticker] = np.nan
            continue

        if pd.isna(price_today):
            continue

        indicators_ready = not (pd.isna(ema_short) or pd.isna(ema_med) or pd.isna(ema_long) or pd.isna(atr))
        if indicators_ready and ema_short > ema_med > ema_long and positions[ticker] == 0:
            if not (use_market_filter and not market_ok):
                shares_bought = 0.0
                invest_amount = 0.0
                exec_price = price_today * (1 + slippage)

                if position_sizing_mode == "fixed_allocation":
                    invest_amount = min(allocation_per_ticker, portfolio_cash)
                    max_allowed_for_position = (portfolio_cash + sum(positions[t] * get_last_close_before(data_dict[t], date) for t in tickers)) * max_position_size
                    invest_amount = min(invest_amount, max_allowed_for_position)
                    shares_bought = invest_amount / exec_price if exec_price > 0 else 0.0

                elif position_sizing_mode == "atr_risk":
                    if pd.isna(atr) or atr <= 0 or exec_price <= 0:
                        shares_bought = 0.0
                        invest_amount = 0.0
                    else:
                        equity_est = portfolio_cash + sum(positions[t] * get_last_close_before(data_dict[t], date) for t in tickers)
                        risk_dollars = equity_est * risk_per_trade
                        distance = max(atr * stop_loss_atr, 1e-8)
                        desired_shares = risk_dollars / distance
                        desired_invest = desired_shares * exec_price
                        desired_invest = min(desired_invest, portfolio_cash)
                        max_allowed_for_position = equity_est * max_position_size
                        invest_amount = min(desired_invest, max_allowed_for_position)
                        shares_bought = invest_amount / exec_price if exec_price > 0 else 0.0
                else:
                    raise ValueError("Unknown position_sizing_mode")

                invest_amount = min(invest_amount, portfolio_cash)
                if shares_bought > 0 and invest_amount > 0:
                    positions[ticker] += shares_bought
                    portfolio_cash -= invest_amount
                    portfolio_cash = max(portfolio_cash - commission, 0.0)
                    peak_prices[ticker] = price_today
                    entry_date[ticker] = date
                    entry_atr[ticker] = atr if not pd.isna(atr) else 0.0
                    buy_points[ticker].append((date, price_today))
                    trade_log.append({
                        "date": date, "ticker": ticker, "type": "BUY",
                        "shares": shares_bought, "price": exec_price, "invested": invest_amount, "cash_after": portfolio_cash
                    })

        if positions[ticker] > 0:
            peak_prices[ticker] = max(peak_prices[ticker], price_today) if not pd.isna(peak_prices[ticker]) else price_today
            allow_stop = True
            if entry_date[ticker] is not None:
                days_held = (date - entry_date[ticker]).days
                allow_stop = days_held >= min_holding_bars_before_stop
            frozen_atr = entry_atr[ticker] if not pd.isna(entry_atr[ticker]) else atr
            trailing_stop_price = peak_prices[ticker] - stop_loss_atr * (frozen_atr if not pd.isna(frozen_atr) else 0.0)
            if allow_stop and (not pd.isna(trailing_stop_price)) and price_today < trailing_stop_price:
                exec_price = price_today * (1 - slippage)
                proceeds = positions[ticker] * exec_price
                portfolio_cash += proceeds
                portfolio_cash = max(portfolio_cash - commission, 0.0)
                sell_points[ticker].append((date, price_today))
                trade_log.append({
                    "date": date, "ticker": ticker, "type": "SELL",
                    "shares": positions[ticker], "price": exec_price, "proceeds": proceeds, "cash_after": portfolio_cash
                })
                positions[ticker] = 0.0
                peak_prices[ticker] = np.nan
                entry_date[ticker] = None
                entry_atr[ticker] = np.nan

    # end of ticker loop for this date -> compute end-of-day portfolio value
    total_value = portfolio_cash
    for t in tickers:
        last_price = get_last_close_before(data_dict[t], date)
        if not pd.isna(last_price):
            total_value += positions[t] * last_price

    # append daily mark-to-market
    account_values.append(float(total_value))
    equity_index.append(date)

# ==============================
# FINAL FORCE-UPDATE (prevent last-day mismatch)
# ==============================
last_date = all_dates[-1]
final_positions_value = 0.0
for t in tickers:
    last_price = get_last_close_before(data_dict[t], last_date)
    if pd.isna(last_price):
        last_price = 0.0
    final_positions_value += positions[t] * float(last_price)
final_value = portfolio_cash + final_positions_value

# ensure the last item in account_values equals final_value
if len(account_values) == 0:
    account_values.append(final_value)
    equity_index.append(last_date)
else:
    account_values[-1] = float(final_value)
    equity_index[-1] = last_date

# ==============================
# RESULTS & METRICS
# ==============================
equity_curve = pd.Series(account_values, index=equity_index).ffill()
print(f"Starting Capital: ${starting_capital:,.2f}")
print(f"Ending Value:     ${final_value:,.2f}")
print(f"Net Profit:       ${final_value - starting_capital:,.2f} ({(final_value/starting_capital - 1)*100:.2f}%)")

# compute metrics using equity_curve index (safe annualization)
daily_returns = equity_curve.pct_change().dropna()
if not daily_returns.empty:
    total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    total_years = total_days / 365.25 if total_days > 0 else 1e-9
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1.0 / total_years) - 1.0
    ann_vol = daily_returns.std() * np.sqrt(252)
    if daily_returns.std() > 0:
        sharpe_like = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_like = np.nan
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdown.min()
else:
    cagr = ann_vol = sharpe_like = max_dd = np.nan

print("\n=== PERFORMANCE METRICS ===")
print(f"CAGR:               {cagr*100:,.2f}%")
print(f"Annual Volatility:  {ann_vol*100:.2f}%")
print(f"Sharpe-like Ratio:  {sharpe_like:.2f}")
print(f"Max Drawdown:       {max_dd*100:.2f}%")

equity_series, position_series, final_value, trade_count = simulate_for_emas(
    ema_short_span,
    ema_med_span,
    ema_long_span
)

allocation_df = pd.DataFrame({
    "Equity": equity_series,
    "PositionValue": position_series
})

allocation_df["StockPct"] = allocation_df["PositionValue"] / allocation_df["Equity"]
allocation_df["CashPct"] = 1 - allocation_df["StockPct"]

plt.figure(figsize=(14,6))
plt.plot(allocation_df.index, allocation_df["StockPct"], label="Stock Allocation", linewidth=2)
plt.plot(allocation_df.index, allocation_df["CashPct"], label="Cash Allocation", linewidth=2)
plt.title("Portfolio Allocation Over Time")
plt.xlabel("Date")
plt.ylabel("Percentage of Total Equity")
plt.legend()
plt.grid(True)
plt.show()

# ==============================
# BUY & HOLD BENCHMARK
# ==============================
bh_returns = []
for t in tickers:
    df = data_dict[t]
    last_real = last_real_trade_date.get(t, None)
    if last_real is None:
        continue
    valid_close = df.loc[:last_real, "Close"].dropna()
    if valid_close.empty:
        continue
    start_price = valid_close.iloc[0]
    end_price = valid_close.iloc[-1]
    bh_returns.append((end_price - start_price) / start_price * 100)

if bh_returns:
    bh_avg_pct = sum(bh_returns) / len(bh_returns)
    strategy_return_pct = (final_value - starting_capital) / starting_capital * 100
    print("\n=== PERFORMANCE COMPARISON ===")
    print(f"Strategy Return:       {strategy_return_pct:.2f}%")
    print(f"Buy & Hold Avg Return: {bh_avg_pct:.2f}%")

# ==============================
# TRADE SUMMARY
# ==============================
print(f"\nTotal Trades: {len(trade_log)}")
buys = sum(1 for tr in trade_log if tr.get("type") == "BUY")
sells = sum(1 for tr in trade_log if tr.get("type") in ("SELL", "DELISTED_EXIT"))
print(f"Buy trades: {buys}, Sell/Exit trades: {sells}")

# ==============================
# PLOT
# ==============================
fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax[0].plot(equity_curve.index, equity_curve.values, label="Portfolio Value")
ax[0].set_title("Portfolio Equity Curve")
ax[0].set_ylabel("Portfolio Value ($)")
ax[0].grid(True)
ax[0].legend()

rolling_max = equity_curve.cummax()
drawdown = (equity_curve - rolling_max) / rolling_max
ax[1].plot(drawdown.index, drawdown.values)
ax[1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3)
ax[1].set_ylabel("Drawdown")
ax[1].set_xlabel("Date")
ax[1].grid(True)

plt.tight_layout()
plt.show()

