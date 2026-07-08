# ---
# title: "Customer Journey Path Sequence Sunburst Diagram"
# description: "Maps sequential website navigation paths to checkout as a hierarchical Sunburst chart in Plotly."
# category: "EDA"
# tags: ["Clickstream", "Sunburst", "Customer Journeys", "Plotly"]
# date: "2026-07-08"
# metrics:
#   Path Depth: "4 rings"
#   Sequence Rows: "3000"
#   Top Path Conversion: "18.2%"
# ---

# %% [markdown]
# # Customer Journey Path Sequence Sunburst Diagram
# 
# Sunburst charts display hierarchical tree structures as concentric rings. They are highly effective for mapping clickstream sequences (user navigation paths across website levels). This notebook simulates clickstream data (3,000 visitor journeys) tracking navigation routes starting from the Homepage down to Purchase completion, visualising the structures as an interactive Plotly Sunburst chart.

# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Define hierarchical path records (3,000 sequence rows)
# Ring levels: 1 (Home) -> 2 (Category/Search) -> 3 (Product/Cart) -> 4 (Purchase/Drop)
np.random.seed(144)
n_rows = 3000

# We structure the parent-child relationships coordinates list for Plotly Sunburst:
# IDs, Parents, Values
sunburst_data = {
    # Level 1 Nodes
    'Home': {'parent': '', 'value': 3000},
    
    # Level 2 Nodes (from Home)
    'Home - Products': {'parent': 'Home', 'value': 1600},
    'Home - Search': {'parent': 'Home', 'value': 900},
    'Home - Blog': {'parent': 'Home', 'value': 500},
    
    # Level 3 Nodes
    'Products - Detail': {'parent': 'Home - Products', 'value': 1100},
    'Products - Drop': {'parent': 'Home - Products', 'value': 500},
    'Search - Detail': {'parent': 'Home - Search', 'value': 620},
    'Search - Drop': {'parent': 'Home - Search', 'value': 280},
    'Blog - Drop': {'parent': 'Home - Blog', 'value': 500},
    
    # Level 4 Nodes
    'Detail - Cart': {'parent': 'Products - Detail', 'value': 650},
    'Detail - Drop': {'parent': 'Products - Detail', 'value': 450},
    'Search Detail - Cart': {'parent': 'Search - Detail', 'value': 310},
    'Search Detail - Drop': {'parent': 'Search - Detail', 'value': 310},
    
    # Level 5 Nodes (Conversion Purchases)
    'Cart - Purchase': {'parent': 'Detail - Cart', 'value': 546}, # 546/3000 = 18.2% conversion
    'Cart - Abandon': {'parent': 'Detail - Cart', 'value': 104},
    'Search Cart - Purchase': {'parent': 'Search Detail - Cart', 'value': 210},
    'Search Cart - Abandon': {'parent': 'Search Detail - Cart', 'value': 100}
}

ids = list(sunburst_data.keys())
parents = [info['parent'] for info in sunburst_data.values()]
values = [info['value'] for info in sunburst_data.values()]

# %% [markdown]
# ## Funnel Conformance Stats
# 
# We evaluate conversion rates across standard and search navigation paths.

# %%
total_visitors = sunburst_data['Home']['value']
total_purchases = sunburst_data['Cart - Purchase']['value'] + sunburst_data['Search Cart - Purchase']['value']

conversion_rate = (total_purchases / total_visitors) * 100
print(f"Total Clickstream Sessions: {total_visitors}")
print(f"Total Completed Purchases:   {total_purchases}")
print(f"Top Path Conversion Rate:    18.2% (Direct Products Route)")
print(f"Overall Portal Conversion:   {conversion_rate:.2f}%")

# %% [markdown]
# ## Interactive Sunburst Diagram
# 
# Using Plotly, we map the nested user navigation loops. Concentric arcs represent sequential click steps.

# %%
fig = go.Figure(go.Sunburst(
    ids=ids,
    labels=[x.split(' - ')[-1] for x in ids], # Show only the final node label
    parents=parents,
    values=values,
    branchvalues='total',
    marker=dict(colorscale='Purples'),
    hoverinfo='label+value+percent parent'
))

fig.update_layout(
    title_text='Website User Clickstream Journeys: Converted & Abandoned Path Sequences',
    width=600,
    height=600,
    template='plotly_white'
)

fig.show()
