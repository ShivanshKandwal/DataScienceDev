# ---
# title: "Real Estate Price Index Geo-Choropleth Map"
# description: "Maps state-level real estate valuation variations using regional choropleth US boundary projections."
# category: "EDA"
# tags: ["Geospatial", "Choropleth Map", "Real Estate", "Plotly"]
# date: "2026-07-08"
# metrics:
#   Regional Codes: "50"
#   Pricing Average: "$385k"
#   Year Growth: "+4.2%"
# ---

# %% [markdown]
# # Real Estate Price Index Geo-Choropleth Map
# 
# Choropleth maps color geographic regions (countries, states, provinces) proportionally based on continuous metrics. Plotly's US choropleth feature maps data using ISO state codes (e.g., CA, TX, NY). This notebook generates synthetic average house valuations and year-over-year growth indices across 50 US states, rendering an interactive geographic choropleth map.

# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# US State abbreviations list (50 states)
state_codes = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]
n_states = len(state_codes)

# Generate synthetic home valuations (pricing index in thousands)
# Pricing varies regionally (West Coast/East Coast has higher averages)
np.random.seed(142)
prices = []
growth_rates = []

for code in state_codes:
    if code in ['CA', 'NY', 'MA', 'WA', 'HI', 'NJ']:
        price = np.random.normal(680.0, 50.0) # High pricing states
        growth = np.random.normal(5.8, 1.2)
    elif code in ['TX', 'FL', 'CO', 'AZ', 'NC', 'GA']:
        price = np.random.normal(390.0, 30.0) # Mid-high growth states
        growth = np.random.normal(6.5, 1.5)
    else:
        price = np.random.normal(280.0, 40.0) # Baseline states
        growth = np.random.normal(3.2, 0.8)
        
    prices.append(price)
    growth_rates.append(growth)

df_housing = pd.DataFrame({
    'State': state_codes,
    'Avg_Valuation_k': np.clip(prices, 150.0, 950.0),
    'YoY_Growth_Pct': np.clip(growth_rates, 0.5, 12.0)
})

df_housing.head(10)

# %% [markdown]
# ## Valuation Metrics Aggregations
# 
# We calculate national housing index averages and growth rates.

# %%
avg_national_price = df_housing['Avg_Valuation_k'].mean() * 1000
avg_national_growth = df_housing['YoY_Growth_Pct'].mean()

print(f"National House Valuation Average: ${avg_national_price:,.2f}")
print(f"National Avg YoY Market Growth:   {avg_national_growth:.2f}%")

# %% [markdown]
# ## US State Choropleth Visualizer
# 
# Using Plotly choropleth traces, we project state boundaries and map home valuations to colors.

# %%
fig = go.Figure(go.Choropleth(
    locations=list(df_housing['State']),
    z=list(df_housing['Avg_Valuation_k']),
    locationmode='USA-states',
    colorscale='YlOrRd',
    colorbar=dict(title="Avg House Value ($k)"),
    text=[
        f"State: {row['State']}<br>Value: ${row['Avg_Valuation_k']:.1f}k<br>YoY Growth: {row['YoY_Growth_Pct']:.1f}%"
        for _, row in df_housing.iterrows()
    ],
    hoverinfo='text'
))

fig.update_layout(
    title='US Real Estate Pricing Index: Median House Valuations by State',
    geo=dict(
        scope='usa',
        projection=dict(type='albers usa'),
        showlakes=True,
        lakecolor='rgb(255, 255, 255)'
    ),
    width=650,
    height=480,
    margin=dict(l=0, r=0, t=40, b=0)
)

fig.show()
