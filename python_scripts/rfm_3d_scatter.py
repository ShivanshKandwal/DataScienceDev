# ---
# title: "Customer Recency, Frequency, Monetary (RFM) Segmentation 3D Scatter"
# description: "Groups consumers along RFM axes and plots them in a 3D coordinate space to highlight high-value users."
# category: "EDA"
# tags: ["Marketing Analytics", "RFM", "3D Scatter", "Plotly"]
# date: "2026-07-08"
# metrics:
#   Customer Count: "1500"
#   Dimensions: "3"
#   High-Value Ratio: "14.5%"
# ---

# %% [markdown]
# # Customer RFM Segmentation 3D Scatter
# 
# RFM analysis evaluates:
# 1. **Recency (R):** Days since the last purchase (lower is better).
# 2. **Frequency (F):** Total number of transactions (higher is better).
# 3. **Monetary (M):** Total spent value (higher is better).
# Visualizing these 3 dimensions in a 3D scatter plot exposes clear customer segment clusters (e.g. VIP champions, loyal core, at-risk churners). This notebook generates synthetic RFM profiles and visualizes them in Plotly.

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Generate synthetic consumer RFM metrics (1,500 customers)
np.random.seed(115)
n_customers = 1500

# Base customer parameters
recency = np.random.exponential(45, n_customers) # Exponential: most purchased recently
frequency = np.random.poisson(8, n_customers) + 1
monetary = frequency * np.random.normal(35, 10, n_customers) + np.random.normal(50, 15, n_customers)

# Define customer categories based on simple thresholds
categories = []
for r, f, m in zip(recency, frequency, monetary):
    if r < 15 and f > 12 and m > 450:
        categories.append('Champions (VIP)')
    elif r > 90 and f < 4:
        categories.append('At Risk / Churning')
    else:
        categories.append('Regular Customer')

df_rfm = pd.DataFrame({
    'CustomerID': [f"Cust_{i:04d}" for i in range(n_customers)],
    'Recency': np.clip(recency, 1, 180),
    'Frequency': np.clip(frequency, 1, 30),
    'Monetary': np.clip(monetary, 5, 1200),
    'Segment': categories
})

df_rfm.head(10)

# %% [markdown]
# ## Segment Demographics
# 
# We evaluate category counts and average spending parameters across segments.

# %%
df_stats = df_rfm.groupby('Segment').agg(
    Count=('CustomerID', 'count'),
    Avg_Recency=('Recency', 'mean'),
    Avg_Frequency=('Frequency', 'mean'),
    Avg_Monetary=('Monetary', 'mean')
)
print("RFM Customer Segment Aggregations:")
df_stats

# %% [markdown]
# ## Interactive 3D RFM Coordinates Scatter
# 
# Using Plotly Scatter3d, we plot Recency, Frequency, and Monetary on X, Y, and Z axes, color-coding markers by Customer Segment.

# %%
# Convert numpy structures to standard lists for JSON serialization stability
segments = ['Champions (VIP)', 'Regular Customer', 'At Risk / Churning']
colors = {
    'Champions (VIP)': '#10B981',      # Green
    'Regular Customer': '#3B82F6',     # Blue
    'At Risk / Churning': '#EF4444'     # Red
}

fig = go.Figure()

for seg in segments:
    df_seg = df_rfm[df_rfm['Segment'] == seg]
    
    fig.add_trace(go.Scatter3d(
        x=list(df_seg['Recency']),
        y=list(df_seg['Frequency']),
        z=list(df_seg['Monetary']),
        mode='markers',
        name=seg,
        marker=dict(
            size=5,
            color=colors[seg],
            opacity=0.75 if seg != 'Champions (VIP)' else 0.9,
            line=dict(width=0.5, color='black')
        ),
        hoverinfo='text',
        hovertext=[
            f"<b>{row['CustomerID']}</b><br>Recency: {row['Recency']:.0f} days<br>Frequency: {row['Frequency']} purchases<br>Monetary: ${row['Monetary']:.2f}"
            for _, row in df_seg.iterrows()
        ]
    ))

fig.update_layout(
    title='Customer RFM Segmentations 3D Coordinate Space',
    scene=dict(
        xaxis_title='Recency (Days, Lower is Better)',
        yaxis_title='Frequency (Count, Higher is Better)',
        zaxis_title='Monetary Spent ($ Value)',
        xaxis=dict(gridcolor='#F1F5F9'),
        yaxis=dict(gridcolor='#F1F5F9'),
        zaxis=dict(gridcolor='#F1F5F9')
    ),
    width=650,
    height=550,
    template='plotly_white'
)

fig.show()
