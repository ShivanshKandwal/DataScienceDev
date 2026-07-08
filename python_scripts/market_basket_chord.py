# ---
# title: "Retail Basket Association Matrix Chord Diagram"
# description: "Simulates cross-purchase associations and plots correlation relationships on a circular node layout in Plotly."
# category: "EDA"
# tags: ["Market Basket Analysis", "Chord Visual", "Relationships Matrix", "Plotly"]
# date: "2026-07-08"
# metrics:
#   Products: "10"
#   Link Counts: "41"
#   Confidence Min: "0.20"
# ---

# %% [markdown]
# # Retail Basket Association Matrix Circular Chart
# 
# Chord diagrams and circular relation graphs display multi-group relationships and transaction dependencies between categories. In retail market basket analysis, they map co-purchase associations between product categories. This notebook generates synthetic transaction associations across ten departments and visualizes correlation relationships on a circular coordinate node layout in Plotly.

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Define 10 product categories
categories = [
    'Bakery', 'Dairy', 'Produce', 'Meat', 'Pantry',
    'Snacks', 'Beverages', 'Frozen', 'Deli', 'Bakeware'
]
n_items = len(categories)

# Generate symmetric correlation association matrix (co-purchase frequencies)
np.random.seed(135)
assoc_matrix = np.random.uniform(0.05, 0.65, (n_items, n_items))
assoc_matrix = (assoc_matrix + assoc_matrix.T) / 2.0 # Make symmetric
np.fill_diagonal(assoc_matrix, 0.0) # Zero self associations

df_assoc = pd.DataFrame(assoc_matrix, index=categories, columns=categories)
df_assoc

# %% [markdown]
# ## Circular Node Coordinates Layout
# 
# We distribute the 10 nodes evenly along a unit circle circumference to establish visual coordinates.

# %%
angles = np.linspace(0, 2 * np.pi, n_items, endpoint=False)
node_x = np.cos(angles)
node_y = np.sin(angles)

df_nodes = pd.DataFrame({
    'Category': categories,
    'X': node_x,
    'Y': node_y
})
df_nodes

# %% [markdown]
# ## Circular Relations Network Graph
# 
# Using Plotly Scatter, we plot nodes as circles and co-purchase strengths above a threshold (0.20) as connecting paths with variable opacity.

# %%
fig = go.Figure()

# 1. Plot connection link lines (connections)
# Loop through lower triangle to prevent duplicates
link_count = 0
for i in range(n_items):
    for j in range(i + 1, n_items):
        weight = assoc_matrix[i, j]
        if weight >= 0.20:
            link_count += 1
            # Coordinate line
            x_line = [node_x[i], node_x[j]]
            y_line = [node_y[i], node_y[j]]
            
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                line=dict(color='rgba(99, 102, 241, 0.35)', width=weight * 8.0), # Width represents correlation
                showlegend=False,
                hoverinfo='none'
            ))

print(f"Rendered Connection Links: {link_count}")

# 2. Plot categories nodes
fig.add_trace(go.Scatter(
    x=list(node_x),
    y=list(node_y),
    mode='markers+text',
    marker=dict(
        size=24,
        color='#6366F1',
        line=dict(width=1.5, color='white')
    ),
    text=categories,
    textposition='top center',
    name='Product Category',
    hoverinfo='text',
    hovertext=[f"<b>{cat}</b><br>Active Associations" for cat in categories]
))

fig.update_layout(
    title='Retail Basket Co-Purchase Associations Network Matrix',
    xaxis=dict(visible=False, range=[-1.4, 1.4]),
    yaxis=dict(visible=False, range=[-1.4, 1.4]),
    width=600,
    height=550,
    template='plotly_white'
)

fig.show()
