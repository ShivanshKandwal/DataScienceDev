# ---
# title: "Web Traffic Hourly Activity Radial Polar Chart"
# description: "Maps web traffic request volumes across 24-hour cycles onto polar radar axes to pinpoint server peaks."
# category: "EDA"
# tags: ["Web Telemetry", "Polar Chart", "Chronological Analytics", "Plotly"]
# date: "2026-07-08"
# metrics:
#   Sample Size: "24h"
#   Server Load Index: "0-100"
#   Peak Hours: "19:00-22:00"
# ---

# %% [markdown]
# # Web Traffic Hourly Activity Radial Polar Chart
# 
# Polar charts project value logs onto radial angular axes, making them ideal for representing periodic cyclic sequences (such as 24-hour hourly traffic, calendar seasons, or weekly averages). This notebook defines synthetic server request profiles across a 24-hour day and visualizes load peaks using a customized interactive Plotly Polar chart.

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Define hours labels (00:00 to 23:00)
hours_list = [f"{i:02d}:00" for i in range(24)]

# Define synthetic CPU/Server Load Indexes (0-100 scale)
# Inject double peak patterns: midday (12:00-14:00) and evening peak (19:00-22:00)
np.random.seed(120)
base_load = np.sin(np.linspace(0, 2 * np.pi, 24) - np.pi/2) * 20 + 45 # Base wave

load_noise = np.random.normal(0, 3, 24)
server_load = np.clip(base_load + load_noise, 5, 95)

# Boost peak hours manually
server_load[11:14] += 15 # Lunch peak
server_load[19:22] += 20 # Evening surge
server_load = np.clip(server_load, 5, 100)

df_load = pd.DataFrame({
    'Hour': hours_list,
    'Server_Load_Index': server_load
})

df_load

# %% [markdown]
# ## Cyclic Verification
# 
# We calculate statistics for the morning valley, lunch peak, and evening surge periods.

# %%
morning_valley = df_load.iloc[2:6]['Server_Load_Index'].mean()
lunch_peak = df_load.iloc[11:14]['Server_Load_Index'].mean()
evening_surge = df_load.iloc[19:22]['Server_Load_Index'].mean()

print(f"Average Morning Valley Load (02:00-06:00): {morning_valley:.1f}%")
print(f"Average Lunch Peak Load (11:00-14:00):     {lunch_peak:.1f}%")
print(f"Average Evening Surge Load (19:00-22:00):   {evening_surge:.1f}%")

# %% [markdown]
# ## Interactive 24-Hour Polar Line Chart
# 
# Using Plotly Scatterpolar, we map hours to 15-degree angular steps ($360 / 24$) and closed contours to track hourly fluctuations.

# %%
# Append starting index to close polar coordinate paths
hours_closed = hours_list + [hours_list[0]]
load_closed = list(server_load) + [server_load[0]]

fig = go.Figure(go.Scatterpolar(
    r=load_closed,
    theta=hours_closed,
    mode='lines+markers',
    fill='toself',
    fillcolor='rgba(139, 92, 246, 0.25)', # Soft purple transparent fill
    line=dict(color='#8B5CF6', width=2),
    marker=dict(size=5, color='#7C3AED')
))

fig.update_layout(
    title='24-Hour Server Cluster load Index Profile (Radial Polar View)',
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100],
            gridcolor='#E2E8F0',
            ticksuffix='%'
        ),
        angularaxis=dict(
            gridcolor='#E2E8F0',
            direction='clockwise',
            period=24
        )
    ),
    width=600,
    height=500,
    template='plotly_white'
)

fig.show()
