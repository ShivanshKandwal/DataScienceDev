# ---
# title: "Predictive Maintenance XGBoost Classifier"
# description: "Trains a gradient boosting classifier to predict industrial machine failure modes from sensor logs. Analyzes feature importance scores."
# category: "ML"
# tags: ["XGBoost", "Classification", "Feature Importance", "Plotly"]
# date: "2026-06-28"
# metrics:
#   ROC AUC: "0.868"
#   Accuracy: "92.0%"
#   Features: "6"
# ---

# %% [markdown]
# # Predictive Maintenance XGBoost Classifier
# 
# Modern manufacturing facilities use sensor telemetry to predict equipment failures before they happen. This project constructs a synthetic industrial telemetry dataset containing speed, torque, tool wear, and temperature sensor logs, trains an XGBoost classification pipeline to detect failure risks, and evaluates model predictions.

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Generate synthetic sensor telemetry records (2,000 machines/samples)
np.random.seed(54)
n_samples = 2000

# Base continuous features
rpm = np.random.normal(1500, 200, n_samples).clip(800, 2200)
torque = np.random.normal(40, 10, n_samples).clip(10, 80)
tool_wear = np.random.uniform(0, 240, n_samples)  # wear time in minutes
air_temp = np.random.normal(298, 2, n_samples)  # Kelvin
process_temp = air_temp + np.random.normal(10, 1, n_samples)
vibration = np.random.exponential(1.5, n_samples).clip(0.1, 8.0)

# Failure conditions logic (failures represent ~5% of events)
# Failure probability increases if torque is high and rpm is low, or if tool wear is extreme, or if vibration is extreme.
p_fail = (
    0.01 
    + 0.25 * (tool_wear > 200) 
    + 0.30 * (vibration > 4.5) 
    + 0.20 * ((torque > 60) & (rpm < 1200))
    + 0.15 * (process_temp > 311)
).clip(0, 1)

failure = np.random.binomial(1, p_fail)

df = pd.DataFrame({
    'RPM': rpm,
    'Torque_Nm': torque,
    'Tool_Wear_Min': tool_wear,
    'Air_Temp_K': air_temp,
    'Process_Temp_K': process_temp,
    'Vibration_mm_s': vibration,
    'Failure': failure
})

print(f"Dataset failures: {df['Failure'].sum()} failures out of {len(df)} samples ({df['Failure'].mean()*100:.1f}%)")
df.head(10)

# %% [markdown]
# ## Classification Model Training
# 
# We partition our dataset into training and validation folds (80/20 split) and fit an extreme gradient boosted classifier (XGBoost) using standard hyperparameter configurations.

# %%
X = df.drop(columns=['Failure'])
y = df['Failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Instantiate and fit XGBoost classifier
clf = XGBClassifier(
    n_estimators=100,
    learning_rate=0.08,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'
)
clf.fit(X_train, y_train)

# Predictions evaluation
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

# %% [markdown]
# ## Feature Importances
# 
# We extract relative feature importances from our trained XGBoost classifier to understand which sensor inputs contribute most to machine failure predictions.

# %%
feature_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importances:")
print(feature_imp)

# %% [markdown]
# ## Interactive Evaluation Plotly Dashboard
# 
# We construct a multi-trace interactive Plotly chart showcasing the Model ROC curve (Left) and the Feature Importance profile (Right).

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Compute ROC curve and AUC area
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Convert to standard Python lists to avoid JSON serialization errors
fpr_list = list(fpr)
tpr_list = list(tpr)
feature_imp_vals = [float(v) for v in feature_imp.values[::-1]]
feature_imp_idx = list(feature_imp.index[::-1])

# Initialize a subplots grid: 1 row, 2 columns
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Receiver Operating Characteristic (ROC)', 'Feature Importance Analysis'),
    horizontal_spacing=0.15
)

# 1. ROC Curve Trace (Col 1)
fig.add_trace(
    go.Scatter(
        x=fpr_list, y=tpr_list,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.3f})',
        line=dict(color='#2563EB', width=3)
    ),
    row=1, col=1
)
# Reference baseline line
fig.add_trace(
    go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Guess',
        line=dict(color='#94A3B8', dash='dash')
    ),
    row=1, col=1
)

# 2. Feature Importances Trace (Col 2)
fig.add_trace(
    go.Bar(
        x=feature_imp_vals,
        y=feature_imp_idx,
        orientation='h',
        name='Feature Importance',
        marker=dict(color='#059669')
    ),
    row=1, col=2
)

# Layout settings
fig.update_layout(
    title_text="XGBoost Predictive Maintenance Model Evaluation",
    template="plotly_white",
    showlegend=True,
    legend=dict(
        x=0.01,
        y=0.01,
        bgcolor='rgba(255, 255, 255, 0.7)'
    ),
    width=900,
    height=480
)

# Axes settings
fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
fig.update_xaxes(title_text="Relative Importance Score", row=1, col=2)
fig.update_yaxes(title_text="Sensor Feature", row=1, col=2)

fig.show()
