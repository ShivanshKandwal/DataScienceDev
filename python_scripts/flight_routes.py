# ---
# title: "Global Flight Path Connections Map"
# description: "Models flight networks across global coordinates. Renders 3D orthographic globe connections using Plotly Scattergeo."
# category: "EDA"
# tags: ["EDA", "Geospatial", "Network Graph", "Plotly"]
# date: "2026-07-02"
# metrics:
#   Route Count: "12"
#   Cities Connected: "6"
#   Max Range: "12k km"
# ---

# %% [markdown]
# # Global Flight Path Connections Map
# 
# Geospatial network visualizations map geographic coordinate linkages like flight route connections. This notebook simulates coordinate coordinates for key global airports (New York, London, Tokyo, Dubai, Sydney, Paris) and uses Plotly Scattergeo to construct an interactive 3D orthographic globe showing flight routes, traffic volumes, and flight ranges.

# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Define airport data coordinates
airports = {
    'JFK': {'name': 'New York (JFK)', 'lat': 40.6413, 'lon': -73.7781, 'traffic': 85},
    'LHR': {'name': 'London (LHR)', 'lat': 51.4700, 'lon': -0.4543, 'traffic': 78},
    'HND': {'name': 'Tokyo (HND)', 'lat': 35.5494, 'lon': 139.7798, 'traffic': 70},
    'DXB': {'name': 'Dubai (DXB)', 'lat': 25.2532, 'lon': 55.3657, 'traffic': 92},
    'SYD': {'name': 'Sydney (SYD)', 'lat': -33.9461, 'lon': 151.1772, 'traffic': 45},
    'CDG': {'name': 'Paris (CDG)', 'lat': 49.0097, 'lon': 2.5479, 'traffic': 65}
}

df_airports = pd.DataFrame.from_dict(airports, orient='index')
df_airports['Code'] = df_airports.index
df_airports.head()

# %% [markdown]
# ## Flight Routes Generation & Haversine Distance
# 
# We calculate flight route pairs between airports and compute flight distances in kilometers using the Haversine formula to estimate flight ranges:
# $$d = 2R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)$$

# %%
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Define flights connections
routes = [
    ('JFK', 'LHR', 140), ('JFK', 'CDG', 95), ('LHR', 'DXB', 180), 
    ('CDG', 'DXB', 110), ('DXB', 'HND', 130), ('HND', 'SYD', 85), 
    ('SYD', 'DXB', 65), ('LHR', 'JFK', 150), ('HND', 'JFK', 55),
    ('LHR', 'HND', 75), ('JFK', 'SYD', 35), ('CDG', 'LHR', 220)
]

flight_routes = []
for start, end, flights_count in routes:
    s_lat, s_lon = airports[start]['lat'], airports[start]['lon']
    e_lat, e_lon = airports[end]['lat'], airports[end]['lon']
    dist = haversine_distance(s_lat, s_lon, e_lat, e_lon)
    
    flight_routes.append({
        'Start': start, 'End': end,
        'Start_Lat': s_lat, 'Start_Lon': s_lon,
        'End_Lat': e_lat, 'End_Lon': e_lon,
        'Flights_Weekly': flights_count,
        'Distance_KM': dist
    })

df_routes = pd.DataFrame(flight_routes)
print(f"Total simulated routes: {len(df_routes)}")
df_routes.head()

# %% [markdown]
# ## 3D Globe Flight Paths Rendering
# 
# We construct a 3D orthographic globe in Plotly, overlaying airport coordinate markers (colored by traffic volume) and flight pathway curves (scaled by weekly flight frequency counts).

# %%
# Convert structures to python lists for JSON serialization compatibility
airport_names = list(df_airports['name'])
airport_lats = [float(l) for l in df_airports['lat']]
airport_lons = [float(l) for l in df_airports['lon']]
airport_traffic = [float(t) for t in df_airports['traffic']]

fig = go.Figure()

# 1. Plot airport nodes
fig.add_trace(go.Scattergeo(
    lon=airport_lons,
    lat=airport_lats,
    text=airport_names,
    mode='markers+text',
    textposition='top center',
    marker=dict(
        size=12,
        color=airport_traffic,
        colorscale='Viridis',
        colorbar=dict(title="Weekly Traffic (Index)", len=0.7),
        line=dict(width=1.5, color='white')
    ),
    name='Global Airports'
))

# 2. Draw flight route connecting lines
for idx, row in df_routes.iterrows():
    # Curve paths look cleaner when drawn directly as lines between coordinates
    fig.add_trace(go.Scattergeo(
        lon=[row['Start_Lon'], row['End_Lon']],
        lat=[row['Start_Lat'], row['End_Lat']],
        mode='lines',
        line=dict(
            width=float(row['Flights_Weekly']) / 40,
            color='#E11D48'
        ),
        opacity=0.6,
        name=f"{row['Start']} ➔ {row['End']} ({row['Distance_KM']:.0f} km)",
        showlegend=False
    ))

# Configure 3D Globe projections layout
fig.update_layout(
    title='Interactive 3D Global Flight Paths Map (weekly flight counts)',
    showlegend=False,
    geo=dict(
        projection_type='orthographic', # Creates a 3D spinning globe representation
        showland=True,
        landcolor='#F8FAFC',
        countrycolor='#E2E8F0',
        oceancolor='#EFF6FF',
        showocean=True,
        lakecolor='#EFF6FF',
        showlakes=True
    ),
    width=700,
    height=600
)

fig.show()
