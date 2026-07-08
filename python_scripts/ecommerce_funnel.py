# ---
# title: "E-Commerce Conversion Funnel Heatmap Matrix"
# description: "Tracks conversion decays along different checkout stages across mobile OS, desktop browser, and app configurations."
# category: "EDA"
# tags: ["Product Analytics", "Funnel Matrix", "Heatmap", "Seaborn"]
# date: "2026-07-08"
# metrics:
#   Funnel Levels: "4"
#   Traffic Channels: "6"
#   Conversions: "4.8%"
# ---

# %% [markdown]
# # E-Commerce Conversion Funnel Heatmap Matrix
# 
# Funnel analysis monitors the conversion drop-off as users move through checkout stages. A funnel matrix plots conversion rates across multiple customer acquisition channels. This notebook simulates user counts across four funnel stages (Home, Product, Cart, Checkout) for six device channels and visualizes the retention rates as a Seaborn Heatmap.

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define funnel stages and device traffic channels (6 channels, 4 stages)
channels = ['Desktop Chrome', 'Desktop Safari', 'Mobile iOS App', 'Mobile Android App', 'Mobile Web Safari', 'Mobile Web Chrome']
stages = ['1_Home', '2_Product_View', '3_Add_To_Cart', '4_Purchase']

# Simulate funnel counts where subsequent stages decrease (decay)
np.random.seed(130)
data_matrix = []

for ch in channels:
    # Starting users at Home stage
    if 'Desktop' in ch:
        start_users = np.random.randint(12000, 18000)
        decays = [1.0, 0.58, 0.18, 0.045] # Desktop conversion decay rates
    elif 'App' in ch:
        start_users = np.random.randint(8000, 11000)
        decays = [1.0, 0.72, 0.32, 0.092] # Native App decay rates (higher conversion)
    else: # Mobile Web
        start_users = np.random.randint(15000, 22000)
        decays = [1.0, 0.48, 0.12, 0.024] # Mobile Web decay rates (lower conversion)
        
    counts = [int(start_users * d) for d in decays]
    data_matrix.append(counts)

df_funnel = pd.DataFrame(data_matrix, index=channels, columns=stages)
df_funnel

# %% [markdown]
# ## Conversion Matrix Calculation
# 
# We convert raw user counts into percentage retention metrics relative to the initial 'Home' stage.

# %%
df_retention = df_funnel.div(df_funnel['1_Home'], axis=0) * 100
print("Funnel Retention Percentage Rates Matrix:")
df_retention

# %% [markdown]
# ## Annotated Funnel Heatmap
# 
# Using Seaborn, we plot the retention matrix as a heatmap, adding percentage labels and a diverging coolwarm colorbar.

# %%
plt.figure(figsize=(9.5, 6.5))

sns.heatmap(
    df_retention, 
    annot=True, 
    fmt=".1f", 
    cmap='YlGnBu', 
    linewidths=0.8,
    cbar_kws={'label': 'Funnel Retention Rate (%)'}
)

plt.title('E-Commerce Conversion Funnel Matrix: Retention Rates by Device Traffic Channel', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Checkout Funnel Progress Stage', fontsize=11)
plt.ylabel('Device Traffic Acquisition Channel', fontsize=11)
plt.show()
