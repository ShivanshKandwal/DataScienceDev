# Shivansh Kandwal | AI & Data Science Portfolio Sandbox

An interactive, high-fidelity visual computational portfolio displaying Exploratory Data Analysis, Machine Learning, and Deep Learning projects. This sandbox integrates live Jupyter Notebook parsing, D3 interactive simulations, GPU telemetry dashboards, and real-time Python script compilation.

---

## 🚀 Key Features

1. **Jupyter Notebook Parser**: Reads `.ipynb` files on-the-fly and renders them on the client side with styled print logs, scrollable dataframes, base64 Matplotlib/Seaborn plots, and zoomable interactive Plotly charts.
2. **Stateful Python Script Compiler**: A developer utility that compiles standard `.py` script files (using VS Code `# %%` cells) into fully executed notebooks, capturing outputs and plots without requiring a browser GUI.
3. **Interactive D3 & ECharts Simulations**:
   - **D3 Feedforward Network**: A fully draggable neural network simulation with pulsing forward-activation flows.
   - **Gradient Descent Simulator**: An interactive weight convergence graph showcasing optimal vs. high/low learning rate paths.
   - **GPU Telemetry Dashboard**: Simulates real-time system metrics (VRAM, temperature, step speeds) during neural network training epochs.
4. **Frosted Glassmorphism Design**: Sleek light-theme visual layout with mouse spotlight glows, layered blueprints, topographic grids, and smooth vector micro-animations.

---

## 📂 Project Structure

```text
DataScienceDev/
├── python_scripts/             # Raw Python script templates (.py files)
│   ├── titanic_eda.py          # Exploratory Data Analysis demo (Seaborn dataset)
│   ├── housing_regression.py   # Regression model demo (XGBoost & Plotly)
│   └── brain_tumor_segment.py  # Deep learning segmentation demo (PyTorch & Matplotlib)
├── scripts/
│   └── compile_notebooks.py    # Stateful compiler converting .py scripts to .ipynb
├── public/
│   └── notebooks/              # Compiled fully-executed notebooks (.ipynb JSONs)
├── src/
│   ├── components/             # React components (simulators, viewers, dashboards)
│   │   ├── D3Network.tsx
│   │   ├── NotebookViewer.tsx
│   │   ├── GradientDescent.tsx
│   │   └── TelemetryDashboard.tsx
│   ├── data/
│   │   └── projects.ts         # Automatically generated projects registry metadata
│   ├── App.tsx                 # Main layout structure & navigation links
│   └── index.css               # Design system rules, grid backgrounds, & glass classes
└── package.json                # Project scripts & npm package definitions
```

---

## 🛠️ Getting Started

To run the portfolio website and execute scripts locally:

### Prerequisites
Make sure you have **Node.js (v18+)** and **Python (v3.10+)** installed on your system.

### 1. Install Dependencies

**Node.js Packages:**
```bash
npm install
```

**Python Libraries:**
```bash
pip install matplotlib seaborn plotly xgboost pandas scikit-learn pyyaml torch
```

### 2. Compile Python Scripts
Compile your python scripts to execute them and regenerate the projects list dynamically:
```bash
python scripts/compile_notebooks.py
```

### 3. Run Development Server
```bash
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 📝 How to Add a New Project

Adding a new project is fully automated:

1. Create a new `.py` script inside the `python_scripts/` directory (e.g. `python_scripts/my_model.py`).
2. Add a YAML metadata block at the top specifying details:
   ```python
   # ---
   # title: "My Custom Classifier"
   # description: "Supervised classification comparison on tabular data."
   # category: "ML"                         # Must be "EDA", "ML", or "AI"
   # tags: ["Scikit-Learn", "Python"]
   # date: "2026-06-25"
   # metrics:
   #   Accuracy: "96.4%"
   #   F1-Score: "0.925"
   # ---
   ```
3. Separate cells using `# %%` (for python code) and `# %% [markdown]` (for comments/descriptions).
4. Run `python scripts/compile_notebooks.py` to compile the script and automatically rebuild the frontend cards list!
