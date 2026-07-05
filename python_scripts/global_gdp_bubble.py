# ---
# title: "GDP vs. Life Expectancy Global Bubble Analysis"
# description: "Models global health and wealth indicators. Renders Gapminder-style interactive bubble charts using Plotly."
# category: "EDA"
# tags: ["EDA", "Global Metrics", "Bubble Chart", "Plotly"]
# date: "2026-07-05"
# metrics:
#   Countries: "150"
#   Indicators: "GDP, Life Exp"
#   Year Reference: "2025"
# ---

# %% [markdown]
# # GDP vs. Life Expectancy Global Bubble Analysis
# 
# Gapminder-style bubble charts are powerful visualizations mapping wealth (GDP per capita) on the horizontal axis against health (life expectancy) on the vertical axis, with country populations scaling the bubble dimensions. This notebook simulates demographic indices across 150 countries grouped into four global regions and visualizes indicators using an interactive Plotly bubble chart.

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Generate synthetic global country indexes (150 countries)
np.random.seed(72)
n_countries = 150

regions = np.random.choice(['Americas', 'Europe', 'Asia-Pacific', 'Africa'], size=n_countries, p=[0.25, 0.25, 0.3, 0.2])

# Construct coordinates correlated to regions (e.g. Europe has higher GDP & Life Expectancy)
gdp_per_capita = []
life_expectancy = []
population = []

for region in regions:
    if region == 'Europe':
        gdp = np.random.lognormal(mean=10.2, sigma=0.4) # High GDP
        life = np.random.normal(79, 3)
        pop = np.random.uniform(5, 120) * 1e6
    elif region == 'Americas':
        gdp = np.random.lognormal(mean=9.5, sigma=0.6)
        life = np.random.normal(75, 4)
        pop = np.random.uniform(2, 350) * 1e6
    elif region == 'Asia-Pacific':
        gdp = np.random.lognormal(mean=9.0, sigma=0.8)
        life = np.random.normal(72, 5)
        pop = np.random.uniform(5, 1400) * 1e6 # Higher population scale
    else: # Africa
        gdp = np.random.lognormal(mean=7.8, sigma=0.5)
        life = np.random.normal(63, 6)
        pop = np.random.uniform(2, 200) * 1e6
        
    gdp_per_capita.append(gdp)
    life_expectancy.append(life)
    population.append(pop)

df_global = pd.DataFrame({
    'Country': [f"Country_{i+1:03d}" for i in range(n_countries)],
    'Region': regions,
    'GDP_per_Capita': gdp_per_capita,
    'Life_Expectancy': np.clip(life_expectancy, 45, 90),
    'Population': population
})

df_global.head(10)

# %% [markdown]
# ## Regional Summaries
# 
# We calculate weighted average life expectancies and total populations per region to outline global indicators profiles.

# %%
df_regional = df_global.groupby('Region').agg(
    Avg_GDP=('GDP_per_Capita', 'mean'),
    Avg_Life_Exp=('Life_Expectancy', 'mean'),
    Total_Population=('Population', 'sum')
)
print("Regional Aggregations Summary:")
df_regional

# %% [markdown]
# ## Interactive Global Bubble Chart
# 
# Using Plotly, we construct our Gapminder-style bubble chart, mapping regions to distinct categorical colors and population vectors to bubble sizes.

# %%
# Convert structures to standard lists for JSON serialization compatibility
regions_list = ['Americas', 'Europe', 'Asia-Pacific', 'Africa']
region_colors = {
    'Americas': '#3B82F6',
    'Europe': '#10B981',
    'Asia-Pacific': '#F59E0B',
    'Africa': '#EF4444'
}

fig = go.Figure()

for reg in regions_list:
    df_reg = df_global[df_global['Region'] == reg]
    
    # Calculate bubble sizing based on population square-roots
    sizes = np.sqrt(df_reg['Population'] / 1e6) * 3 + 6
    
    fig.add_trace(go.Scatter(
        x=list(df_reg['GDP_per_Capita']),
        y=list(df_reg['Life_Expectancy']),
        mode='markers',
        name=reg,
        text=list(df_reg['Country']),
        marker=dict(
            size=list(sizes),
            color=region_colors[reg],
            opacity=0.75,
            line=dict(width=1, color='white')
        ),
        hoverinfo='text',
        hovertext=[
            f"<b>{row['Country']}</b><br>Region: {row['Region']}<br>GDP/Capita: ${row['GDP_per_Capita']:,.0f}<br>Life Expectancy: {row['Life_Expectancy']:.1f} Yrs<br>Population: {row['Population']/1e6:.1f}M"
            for _, row in df_reg.iterrows()
        ]
    ))

fig.update_layout(
    title='Global Wealth vs. Health: GDP per Capita vs. Life Expectancy (2025)',
    xaxis=dict(
        title='GDP per Capita ($ USD, Log Scale)',
        type='log',
        gridcolor='#F1F5F9'
    ),
    yaxis=dict(
        title='Life Expectancy at Birth (Years)',
        gridcolor='#F1F5F9'
    ),
    template='plotly_white',
    width=650,
    height=480,
    legend_title='Global Region'
)

fig.show()
