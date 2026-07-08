# ---
# title: "Molecular Structure Valence Orbital Visualization"
# description: "Plots 3D structural coordinate spheres and bonds of organic molecules (e.g. methane, benzene) in interactive spaces."
# category: "EDA"
# tags: ["Cheminformatics", "3D Visualization", "Molecular Modeling", "Plotly"]
# date: "2026-07-08"
# metrics:
#   Atoms: "12"
#   Bonds: "15"
#   Coordinate System: "3D Cart"
# ---

# %% [markdown]
# # Molecular Structure Valence Orbital Visualization
# 
# 3D spatial models of atomic configurations render positions as spheres (atomic radius dimensions) connected by cylinders or lines (chemical valences/bonds). Visualizing spatial coordinates in cheminformatics clarifies stereochemistry. This notebook generates synthetic 3D coordinates for a multi-atom organic molecule (simulating a Benzene ring framework with 12 atoms and 15 bonds) and plots it in Plotly 3D space.

# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Define 12 atoms of a Benzene ring structure (6 Carbons, 6 Hydrogens)
# Coordinates in Angstroms
atom_labels = [
    "C1", "C2", "C3", "C4", "C5", "C6",
    "H1", "H2", "H3", "H4", "H5", "H6"
]

elements = ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"]

# 3D coordinates layout
x_coords = [0.0, 1.2, 1.2, 0.0, -1.2, -1.2,  0.0, 2.15, 2.15, 0.0, -2.15, -2.15]
y_coords = [1.4, 0.7, -0.7, -1.4, -0.7, 0.7,  2.45, 1.25, -1.25, -2.45, -1.25, 1.25]
z_coords = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,     0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

df_atoms = pd.DataFrame({
    'Atom': atom_labels,
    'Element': elements,
    'X': x_coords,
    'Y': y_coords,
    'Z': z_coords
})

df_atoms

# %% [markdown]
# ## Bonds Connections Table
# 
# We define the indices of atoms forming covalent bonds in our benzene ring molecule (15 bonds total).

# %%
bonds = [
    # Carbon ring bonds (6 bonds)
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
    # Carbon-Hydrogen bonds (6 bonds)
    (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11),
    # Double bonds representations (3 extra offset bonds)
    (0, 1), (2, 3), (4, 5)
]

print(f"Total Molecule Atoms: {len(df_atoms)}")
print(f"Total Covalent Bonds: {len(bonds)}")

# %% [markdown]
# ## Interactive 3D Ball-and-Stick Plot
# 
# Using Plotly Scatter3d, we draw atoms as color-coded spheres (Carbon = Grey, Hydrogen = Sky Blue) and bonds as connecting coordinate lines.

# %%
fig = go.Figure()

# 1. Draw bond lines
for start_idx, end_idx in bonds:
    x_line = [df_atoms.loc[start_idx, 'X'], df_atoms.loc[end_idx, 'X']]
    y_line = [df_atoms.loc[start_idx, 'Y'], df_atoms.loc[end_idx, 'Y']]
    z_line = [df_atoms.loc[start_idx, 'Z'], df_atoms.loc[end_idx, 'Z']]
    
    fig.add_trace(go.Scatter3d(
        x=x_line, y=y_line, z=z_line,
        mode='lines',
        line=dict(color='#94A3B8', width=4.5),
        showlegend=False,
        hoverinfo='none'
    ))

# 2. Draw atoms spheres
atom_colors = {'C': '#475569', 'H': '#38BDF8'}
atom_sizes = {'C': 12, 'H': 8}

for el in ['C', 'H']:
    df_el = df_atoms[df_atoms['Element'] == el]
    
    fig.add_trace(go.Scatter3d(
        x=list(df_el['X']),
        y=list(df_el['Y']),
        z=list(df_el['Z']),
        mode='markers',
        name=f'{el} Atoms',
        marker=dict(
            size=atom_sizes[el],
            color=atom_colors[el],
            opacity=0.9,
            line=dict(width=1, color='white')
        ),
        text=list(df_el['Atom']),
        hoverinfo='text'
    ))

fig.update_layout(
    title='Benzene Molecule Framework: 3D Ball-and-Stick Orbitals Visualization',
    scene=dict(
        xaxis=dict(title='X (Å)', gridcolor='#F1F5F9', range=[-3.0, 3.0]),
        yaxis=dict(title='Y (Å)', gridcolor='#F1F5F9', range=[-3.0, 3.0]),
        zaxis=dict(title='Z (Å)', gridcolor='#F1F5F9', range=[-1.5, 1.5]),
        aspectratio=dict(x=1, y=1, z=0.5)
    ),
    width=650,
    height=550,
    template='plotly_white'
)

fig.show()
