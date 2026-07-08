# ---
# title: "Server RAM & CPU Usage Streamgraph Tracker"
# description: "Tracks dynamic server cluster resource usage stacks over 24-hour time windows in streamgraph formats."
# category: "EDA"
# tags: ["SysOps", "Telemetry", "Streamgraph", "Matplotlib"]
# date: "2026-07-08"
# metrics:
#   Server Nodes: "5"
#   Timestamps: "288"
#   Max Capacity: "64GB"
# ---

# %% [markdown]
# # Server RAM & CPU Usage Streamgraph Tracker
# 
# Streamgraphs are stylized stacked area charts centered around a horizontal baseline axis. They visualize flow fluctuations of multiple variables across time sequences without sharp angular bounds. This notebook simulates 24 hours of RAM usage (288 five-minute intervals) across five servers in a cluster, generating a streamgraph visualization using Matplotlib.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate resource telemetry data (288 timestamps representing 24h in 5m intervals)
np.random.seed(140)
n_steps = 288
time_axis = np.linspace(0, 24, n_steps)

# Generate RAM usage streams per server node
# Base sinusoidal activity models daily workload peaks
base_activity = np.sin(time_axis * np.pi / 12 - np.pi/2) * 8 + 12

server_streams = []
for i in range(5):
    noise = np.random.normal(0, 1.5, n_steps)
    # Different offsets and frequencies per server node
    node_load = base_activity + (i * 3) + np.sin(time_axis * (i+1) * np.pi / 6) * 4 + noise
    server_streams.append(np.clip(node_load, 1.0, 64.0))

server_streams = np.array(server_streams)

df_sys = pd.DataFrame(server_streams.T, columns=[f"Server_{i+1}" for i in range(5)])
df_sys['Hour'] = time_axis
df_sys.head(10)

# %% [markdown]
# ## Cluster Usage Aggregations
# 
# We calculate total cluster RAM consumption statistics across time intervals.

# %%
df_sys['Total_Cluster_RAM'] = df_sys[[f"Server_{i+1}" for i in range(5)]].sum(axis=1)
print("RAM Cluster Telemetry Stats (GB):")
df_sys['Total_Cluster_RAM'].describe()

# %% [markdown]
# ## Matplotlib Symmetric Streamgraph Plot
# 
# We construct the streamgraph by calculating the cumulative sum baseline and shifting the stacked components symmetrically around zero.

# %%
# Calculate cumulative sum stacks for baseline offsets
cum_sum = np.sum(server_streams, axis=0)
baseline = -cum_sum / 2.0 # Symmetrical offset baseline

# Offset each stream relative to baseline
streams_offset = []
current_baseline = baseline.copy()
for stream in server_streams:
    streams_offset.append(current_baseline.copy())
    current_baseline += stream

plt.figure(figsize=(10, 6))

colors = ['#3B82F6', '#60A5FA', '#93C5FD', '#1E40AF', '#1D4ED8']
labels = [f"Server Node {i+1}" for i in range(5)]

# Plot symmetric stream areas
for i in range(5):
    plt.fill_between(
        time_axis, 
        streams_offset[i], 
        streams_offset[i] + server_streams[i],
        color=colors[i],
        alpha=0.85,
        edgecolor='white',
        linewidth=0.3,
        label=labels[i]
    )

plt.title('24-Hour Server Cluster Memory Allocation: Streamgraph Telemetry Tracker', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Telemetry Timeline Hour (00:00 to 24:00)', fontsize=11)
plt.ylabel('Symmetric Memory Stack Intensity (GB)', fontsize=11)
plt.xlim(0, 24)
plt.xticks(np.arange(0, 25, 4))
plt.grid(True, linestyle=':', alpha=0.4)
plt.legend(loc='upper right')
plt.show()
