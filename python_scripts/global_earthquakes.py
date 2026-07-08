# ---
# title: "Global Earthquakes Scatterglobe Visualization"
# description: "Parses historical earthquake coordinates and plots magnitudes/depths onto a 3D spinning Scattergeo globe."
# category: "EDA"
# tags: ["Geospatial", "3D Globe", "Seismology", "Plotly"]
# date: "2026-07-08"
# metrics:
#   Events: "1000"
#   Max Magnitude: "8.2"
#   Depth Range: "0-700km"
# ---

# %% [markdown]
# # Global Earthquakes Scatterglobe Visualization
# 
# Map visualizations are highly effective for displaying geographical distributions of natural phenomena. Plotly's 3D Scattergeo maps project latitude and longitude coordinates onto an interactive spherical globe. This notebook generates synthetic global seismic data (1,000 events) and visualizes the epicenters, depths, and magnitudes on a 3D earth globe.

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Generate synthetic global earthquake events (1,000 records)
np.random.seed(101)
n_events = 1000

# Earthquakes occur primarily along tectonic boundaries (simulated via latitude/longitude clusters)
fault_centers = [
    (35.0, 140.0),   # Pacific Ring of Fire (Japan)
    (-15.0, -75.0),  # Nazca Plate (South America)
    (38.0, 15.0),    # Mediterranean
    (5.0, 95.0),     # Sunda Trench (Sumatra)
    (60.0, -145.0)   # Aleutian Trench (Alaska)
]

lats, lons = [], []
for _ in range(n_events):
    center = fault_centers[np.random.randint(len(fault_centers))]
    lat = center[0] + np.random.normal(0, 8.0)
    lon = center[1] + np.random.normal(0, 12.0)
    
    # Clip coordinates to physical boundaries
    lats.append(np.clip(lat, -90.0, 90.0))
    lons.append(np.clip(lon, -180.0, 180.0))

# Magnitude follows an exponential Gutenberg-Richter distribution
magnitudes = np.random.exponential(1.2, n_events) + 3.0
magnitudes = np.clip(magnitudes, 3.0, 8.2)

# Depth ranges from shallow (0km) to deep (700km)
depths = np.random.uniform(5, 700, n_events)

df_quake = pd.DataFrame({
    'Latitude': lats,
    'Longitude': lons,
    'Magnitude': magnitudes,
    'Depth_km': depths
})

df_quake.head(10)

# %% [markdown]
# ## Seismic Activity Stats
# 
# We calculate summary statistics for seismic events to quantify average magnitudes and depth distributions.

# %%
print("Seismic Data Metrics:")
df_quake.describe()

# %% [markdown]
# ## Plotly 3D Earth Globe Visualizer
# 
# We build an interactive 3D Scattergeo globe. The marker size represents earthquake magnitude, and color gradients reflect depth.

# %%
# Convert numpy structures to standard lists for JSON stability
fig = go.Figure(go.Scattergeo(
    lat=list(df_quake['Latitude']),
    lon=list(df_quake['Longitude']),
    mode='markers',
    marker=dict(
        size=list(df_quake['Magnitude'] * 2.2),
        color=list(df_quake['Depth_km']),
        colorscale='Viridis',
        reversescale=True,
        colorbar=dict(title="Depth (km)"),
        opacity=0.75,
        line=dict(width=0.5, color='white')
    ),
    text=[
        f"Mag: {m:.1f}<br>Depth: {d:.0f}km<br>Lat: {la:.2f}, Lon: {lo:.2f}"
        for m, d, la, lo in zip(df_quake['Magnitude'], df_quake['Depth_km'], df_quake['Latitude'], df_quake['Longitude'])
    ],
    hoverinfo='text'
))

fig.update_layout(
    title='Global Seismic Activity: Magnitude & Depth Globe Projection',
    geo=dict(
        projection_type='orthographic', # Creates a 3D rotating globe layout
        showland=True,
        landcolor='#E2E8F0',
        showocean=True,
        oceancolor='#F1F5F9',
        showlakes=True,
        lakecolor='#E2E8F0',
        countrycolor='#CBD5E1',
        showcountries=True
    ),
    width=650,
    height=550,
    margin=dict(l=0, r=0, t=40, b=0)
)

fig.show()
