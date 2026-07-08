# ---
# title: "Climate Temperature Anomaly Hexbin Density Map"
# description: "Aggregates coordinates measurements of global weather deviations using hexbin density grids."
# category: "EDA"
# tags: ["Climatology", "Density Map", "Hexbin", "Matplotlib"]
# date: "2026-07-08"
# metrics:
#   Coordinates Points: "5000"
#   Bins: "30"
#   Anomaly Limits: "-3.5C to +4.0C"
# ---

# %% [markdown]
# # Climate Temperature Anomaly Hexbin Density Map
# 
# Hexagonal binning (hexbin) divides coordinate planes into regular hexagons, counting measurements falling in each bin. This approach handles dense spatial overlays better than standard scatterplots, which suffer from overlapping marks. This notebook generates synthetic latitude/longitude temperature anomaly logs (5,000 measurements) and plots aggregates using a Matplotlib Hexbin plot.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic global temperature anomalies (5,000 spatial measurements)
np.random.seed(125)
n_points = 5000

# Generate coordinates centered around hot zones (e.g. Equator and dry regions)
lats = np.random.uniform(-60.0, 75.0, n_points)
lons = np.random.uniform(-150.0, 150.0, n_points)

# Temperature anomaly values: higher anomalies at high latitudes and land clusters
base_anomaly = np.sin(np.radians(lats)) * 1.5 + 0.8
noise = np.random.normal(0, 0.9, n_points)
anomalies = np.clip(base_anomaly + noise, -3.5, 4.0)

df_temp = pd.DataFrame({
    'Latitude': lats,
    'Longitude': lons,
    'Anomaly_C': anomalies
})

df_temp.head(10)

# %% [markdown]
# ## Spatial Statistics Summary
# 
# We calculate summary statistics for regional temperature anomalies.

# %%
print("Anomaly Temperature Distribution:")
df_temp['Anomaly_C'].describe()

# %% [markdown]
# ## Hexagonal Binning Density Visualization
# 
# We construct a spatial hexbin plot, specifying bin size, aggregating anomaly values using the mean, and coloring hexagons via a coolwarm scale.

# %%
plt.figure(figsize=(10, 6.5))

# Plot hexbin
hb = plt.hexbin(
    df_temp['Longitude'], 
    df_temp['Latitude'], 
    C=df_temp['Anomaly_C'], # Aggregate anomaly values
    gridsize=30, 
    cmap='coolwarm', 
    reduce_C_function=np.mean, # Average values in each bin
    edgecolors='grey',
    linewidths=0.2
)

# Add colorbar representing temperature scale
cb = plt.colorbar(hb, spacing='uniform')
cb.set_label('Mean Temperature Anomaly (°C)', fontsize=11)

plt.title('Global Climate Temperature Anomaly Density Map (Hexagonal Binning)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Longitude Coordinate (Degrees East/West)', fontsize=11)
plt.ylabel('Latitude Coordinate (Degrees North/South)', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.4)
plt.show()
