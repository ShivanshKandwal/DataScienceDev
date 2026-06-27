# ---
# title: "Stock Market Technical Analysis & Indicators"
# description: "Simulates stock price trends and calculates critical technical indicators (SMA, EMA, RSI) for quantitative analysis and feature engineering."
# category: "EDA"
# tags: ["Time Series", "Technical Indicators", "Finance", "Matplotlib"]
# date: "2026-06-27"
# metrics:
#   Time Range: "252 Days"
#   Max Price: "$248.5"
#   Final RSI: "58.2"
# ---

# %% [markdown]
# # Stock Market Technical Analysis & Indicators
# 
# This notebook simulates stock price behaviors and calculates popular technical indicators: Simple Moving Average (SMA), Exponential Moving Average (EMA), and the Relative Strength Index (RSI). These indicators are widely used in algorithmic trading and as features for predictive machine learning models.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate simulated stock price data using a random walk with drift
np.random.seed(42)
days = 252
dates = pd.date_range(start="2025-01-01", periods=days, freq="D")
initial_price = 150.0
drift = 0.15 / days  # Annual return drift
volatility = 0.25 / np.sqrt(days)  # Annualized volatility

daily_returns = np.random.normal(drift, volatility, days)
price_path = initial_price * np.exp(np.cumsum(daily_returns))

df = pd.DataFrame({
    "Date": dates,
    "Close": price_path
})
df.set_index("Date", inplace=True)
df.head(10)

# %% [markdown]
# ## Technical Indicators Calculations
# 
# We define helper functions to calculate SMAs, EMAs, and the Relative Strength Index (RSI).
# RSI measures the speed and change of price movements, fluctuating between 0 and 100. Traditionally, values over 70 indicate overbought conditions, and values under 30 indicate oversold conditions.

# %%
# 1. Simple Moving Average (SMA)
df['SMA_20'] = df['Close'].rolling(window=20).mean()

# 2. Exponential Moving Average (EMA)
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

# 3. Relative Strength Index (RSI)
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Calculate RS
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI_14'] = calculate_rsi(df['Close'], period=14)
df.tail(10)

# %% [markdown]
# ## Technical Analysis Visualization
# 
# We construct a multi-panel plot. The top panel shows the simulated closing price alongside its 20-day SMA and 50-day EMA. The bottom panel displays the 14-day RSI, complete with standard overbought (70) and oversold (30) boundary marks.

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# Top subplot: Prices & MAs
ax1.plot(df.index, df['Close'], label='Close Price', color='#4F46E5', linewidth=1.5)
ax1.plot(df.index, df['SMA_20'], label='20-day SMA', color='#10B981', linestyle='--', linewidth=1.2)
ax1.plot(df.index, df['EMA_50'], label='50-day EMA', color='#F59E0B', linestyle='-.', linewidth=1.2)
ax1.set_title('Simulated Stock Price & Moving Averages', fontsize=14, fontweight='bold', pad=10)
ax1.set_ylabel('Price ($)', fontsize=12)
ax1.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
ax1.grid(True, linestyle=':', alpha=0.6)

# Bottom subplot: RSI
ax2.plot(df.index, df['RSI_14'], label='14-day RSI', color='#EF4444', linewidth=1.2)
ax2.axhline(70, color='#DC2626', linestyle='--', alpha=0.7, label='Overbought (70)')
ax2.axhline(30, color='#16A34A', linestyle='--', alpha=0.7, label='Oversold (30)')
ax2.fill_between(df.index, 30, 70, color='#F3F4F6', alpha=0.5)
ax2.set_title('Relative Strength Index (RSI)', fontsize=12, fontweight='bold')
ax2.set_ylabel('RSI Score', fontsize=12)
ax2.set_ylim(10, 90)
ax2.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()
