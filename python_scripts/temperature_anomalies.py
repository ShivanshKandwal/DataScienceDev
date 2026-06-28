# ---
# title: "Global Temperature Anomalies & Volcanic Activity Correlation"
# description: "Simulates historical temperature deviations and models temporary volcanic aerosol cooling effects. Generates double-axis timeline Plotly plots."
# category: "EDA"
# tags: ["EDA", "Correlation", "Climate Analysis", "Plotly"]
# date: "2026-06-28"
# metrics:
#   Time Span: "150 Years"
#   Correlation Index: "-0.42"
#   Anomalies Recorded: "150"
# ---

# %% [markdown]
# # Global Temperature Anomalies & Volcanic Activity
# 
# Volcanic eruptions release massive quantities of sulfur dioxide aerosols into the stratosphere, which reflect incoming solar radiation and lead to temporary global cooling. This project simulates 150 years of temperature anomalies and examines the correlation between global volcanic eruption indexes (VEI) and subsequent temperature drops.

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Simulate 150 years of global climate data (1876 - 2025)
np.random.seed(42)
years = np.arange(1876, 2026)
n_years = len(years)

# 1. Simulate base global warming trend (anthropogenic greenhouse gas warming)
global_warming_trend = 0.008 * (years - 1876)  # ~1.2°C warming total

# 2. Simulate natural cycle fluctuations (solar and El Niño oscillations)
natural_oscillations = 0.15 * np.sin(2 * np.pi * (years - 1876) / 11) + 0.1 * np.cos(2 * np.pi * (years - 1876) / 4)

# 3. Simulate Volcanic Eruptions index (VEI of 4 or higher)
# Eruptions occur randomly (e.g. 5% chance per year)
volcanic_index = np.zeros(n_years)
eruption_years = np.random.choice(n_years, size=8, replace=False)
volcanic_index[eruption_years] = np.random.uniform(2.0, 5.0, size=8)  # VEI scores

# Add specific famous historical simulated analogues
volcanic_index[years == 1883] = 6.0  # Krakatoa analogue
volcanic_index[years == 1912] = 5.0  # Novarupta analogue
volcanic_index[years == 1991] = 6.0  # Pinatubo analogue

# Volcanic aerosol cooling effect: lags 1-2 years
volcanic_cooling = np.zeros(n_years)
for i in range(n_years):
    if volcanic_index[i] > 0:
        vei = volcanic_index[i]
        # Cooling effect spreads over 3 years following the event
        if i + 1 < n_years:
            volcanic_cooling[i + 1] += -0.12 * vei
        if i + 2 < n_years:
            volcanic_cooling[i + 2] += -0.08 * vei
        if i + 3 < n_years:
            volcanic_cooling[i + 3] += -0.03 * vei

# 4. Add random atmospheric noise
climate_noise = np.random.normal(0, 0.12, n_years)

# Total Temperature Anomaly relative to baseline average
temp_anomaly = global_warming_trend + natural_oscillations + volcanic_cooling + climate_noise

df = pd.DataFrame({
    "Year": years,
    "Temp_Anomaly": temp_anomaly,
    "Volcanic_VEI": volcanic_index,
    "Volcanic_Cooling_Effect": volcanic_cooling
})
df.set_index("Year", inplace=True)
df.head(10)

# %% [markdown]
# ## Post-Eruption Cooling Assessment
# 
# We calculate the statistical correlation between volcanic event magnitudes and the global temperature anomaly deviations in the year immediately following the eruption.

# %%
# Extract years where VEI > 0
eruption_data = df[df['Volcanic_VEI'] > 0].copy()

# Calculate the average temperature anomaly deviation relative to a 5-year pre-eruption baseline
deviations = []
for yr in eruption_data.index:
    pre_years = range(yr - 5, yr)
    post_year = yr + 1
    
    # Check bounds
    if min(pre_years) in df.index and post_year in df.index:
        baseline = df.loc[pre_years, 'Temp_Anomaly'].mean()
        post_temp = df.loc[post_year, 'Temp_Anomaly']
        deviations.append(post_temp - baseline)
    else:
        deviations.append(np.nan)

eruption_data['Post_Eruption_Dev'] = deviations
eruption_data = eruption_data.dropna()

correlation = eruption_data['Volcanic_VEI'].corr(eruption_data['Post_Eruption_Dev'])
print(f"Number of Major Eruptions Analyzed: {len(eruption_data)}")
print(f"Correlation (VEI vs Post-Eruption Temperature Drop): {correlation:.3f}")

# %% [markdown]
# ## Climate Timeline & Volcanic Correlation Plotly Chart
# 
# We build an interactive Plotly dashboard that aligns global temperature anomalies (line plot) on the primary Y-axis with Volcanic Eruption Indexes (VEI) represented as bar indicators on the secondary Y-axis.

# %%
fig = go.Figure()

# 1. Temperature Anomalies Line Chart
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['Temp_Anomaly'],
    mode='lines+markers',
    name='Temp Anomaly (°C)',
    line=dict(color='#E11D48', width=2),
    marker=dict(size=4, color='#9F1239'),
    yaxis='y1'
))

# 2. Volcanic Eruption VEI Bar Indicators
# Masking non-zero values for cleaner bar visualization
volc_mask = df['Volcanic_VEI'] > 0
fig.add_trace(go.Bar(
    x=df.index[volc_mask],
    y=df.loc[volc_mask, 'Volcanic_VEI'],
    name='Volcanic VEI',
    marker=dict(color='#475569', opacity=0.85),
    width=1.5,
    yaxis='y2'
))

# Configure dual y-axes layout
fig.update_layout(
    title=dict(
        text='Global Temperature Anomalies vs. Major Volcanic Activity',
        font=dict(size=16, family='Arial')
    ),
    xaxis=dict(
        title=dict(text='Year'),
        gridcolor='#F1F5F9'
    ),
    yaxis=dict(
        title=dict(
            text='Temperature Anomaly relative to Baseline (°C)',
            font=dict(color='#E11D48')
        ),
        tickfont=dict(color='#E11D48'),
        gridcolor='#F1F5F9'
    ),
    yaxis2=dict(
        title=dict(
            text='Volcanic Eruption Index (VEI)',
            font=dict(color='#475569')
        ),
        tickfont=dict(color='#475569'),
        overlaying='y',
        side='right',
        range=[0, 8],
        showgrid=False
    ),
    template='plotly_white',
    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor='rgba(255, 255, 255, 0.7)'
    ),
    width=850,
    height=500
)

fig.show()
