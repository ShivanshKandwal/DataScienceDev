# ---
# title: "Financial Stock Market Candlestick & Volume Tracker"
# description: "Renders standard OHLC candlesticks alongside transactional volume bars and moving average (SMA) overlaps."
# category: "EDA"
# tags: ["Finance", "Time Series", "Candlestick", "Plotly"]
# date: "2026-07-08"
# metrics:
#   Stock Symbols: "1"
#   Duration: "180 Days"
#   Indicators: "SMA-20, Volume"
# ---

# %% [markdown]
# # Financial Stock Market Candlestick & Volume Tracker
# 
# Candlestick charts present Open, High, Low, and Close (OHLC) values over time, colored red for downward sessions and green for upward sessions. Overlaid metrics (such as Simple Moving Averages - SMA) and volume subplots provide essential momentum indicators. This notebook simulates 180 days of daily equity prices and visualizes metrics using a dual-plot interactive chart in Plotly.

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Simulate 180 trading days of stock prices
np.random.seed(110)
n_days = 180

dates = pd.date_range(start='2026-01-01', periods=n_days, freq='B')

# Generate daily price movements using random walk
initial_price = 150.0
price_drift = 0.05
daily_returns = np.random.normal(price_drift, 1.8, n_days)

close_prices = initial_price + np.cumsum(daily_returns)
open_prices = close_prices - daily_returns + np.random.normal(0, 0.4, n_days)

# High/Low bounds
high_prices = np.maximum(open_prices, close_prices) + np.random.exponential(0.6, n_days)
low_prices = np.minimum(open_prices, close_prices) - np.random.exponential(0.6, n_days)

# Daily volume traded (in thousands)
volumes = np.random.poisson(120, n_days) * 100

df_stock = pd.DataFrame({
    'Date': dates,
    'Open': open_prices,
    'High': high_prices,
    'Low': low_prices,
    'Close': close_prices,
    'Volume': volumes
})

# Calculate 20-day Simple Moving Average (SMA-20)
df_stock['SMA_20'] = df_stock['Close'].rolling(window=20).mean()

df_stock.tail(10)

# %% [markdown]
# ## Moving Average Stats
# 
# We calculate summary statistics for closing prices and indicators to confirm rolling averages.

# %%
print("Stock Index Summary Stats:")
df_stock[['Close', 'Volume', 'SMA_20']].describe()

# %% [markdown]
# ## Dual-Axis Candlestick & Volume Visualization
# 
# Using Plotly subplots, we map the Candlestick and SMA curve on Row 1, and the volume bars on Row 2, linking horizontal zoom controls.

# %%
# Create subplots (Row 1: Candlestick, Row 2: Volume)
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.08,
    row_heights=[0.7, 0.3]
)

# 1. Add Candlestick trace
fig.add_trace(go.Candlestick(
    x=df_stock['Date'],
    open=list(df_stock['Open']),
    high=list(df_stock['High']),
    low=list(df_stock['Low']),
    close=list(df_stock['Close']),
    name='OHLC Price'
), row=1, col=1)

# 2. Add SMA-20 Line
fig.add_trace(go.Scatter(
    x=df_stock['Date'],
    y=list(df_stock['SMA_20']),
    mode='lines',
    line=dict(color='#8B5CF6', width=1.5),
    name='SMA-20 Trend'
), row=1, col=1)

# 3. Add Volume Bar chart
fig.add_trace(go.Bar(
    x=df_stock['Date'],
    y=list(df_stock['Volume']),
    marker=dict(color='#3B82F6', opacity=0.7),
    name='Volume Traded'
), row=2, col=1)

# Configure layout and range slider
fig.update_layout(
    title='Daily Price & Volume History: Stock OHLC Candlestick Tracker',
    xaxis_rangeslider_visible=False, # Disable bottom slider for clean subplots
    xaxis2_title='Trading Date',
    yaxis_title='Stock Price ($)',
    yaxis2_title='Volume',
    template='plotly_white',
    width=650,
    height=550,
    margin=dict(l=50, r=20, t=50, b=40)
)

fig.show()
