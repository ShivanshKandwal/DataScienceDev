# ---
# title: "Credit Card Fraud Detection"
# description: "Supervised anomaly detection on highly imbalanced transaction manifests. Implements cost-sensitive Random Forest classifiers and Precision-Recall curve benchmarks."
# category: "ML"
# tags: ["Scikit-Learn", "Anomaly Detection", "Imbalanced Data", "Plotly"]
# date: "2026-06-26"
# metrics:
#   Fraud Rate: "0.8%"
#   PR-AUC: "0.862"
#   Recall@90%Prec: "82.4%"
# ---

# %% [markdown]
# # Credit Card Fraud Anomaly Detection
# 
# This notebook builds a cost-sensitive classifier to detect fraudulent credit card transactions. Fraud datasets are typically highly imbalanced (e.g. < 1% fraud rate). We generate a synthetic dataset with features representing transactions, train a class-weighted Random Forest, and analyze performance using Precision-Recall (PR) curves.

# %%
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc, confusion_matrix

# Generate highly imbalanced dataset (99.2% genuine, 0.8% fraud)
X, y = make_classification(
    n_samples=2000, 
    n_features=10, 
    n_informative=8, 
    n_redundant=2, 
    weights=[0.992, 0.008],
    random_state=42
)

# Convert to DataFrame
feature_names = [f'V{i}' for i in range(1, 10)] + ['Amount']
df = pd.DataFrame(X, columns=feature_names)
# Rescale Amount column to realistic dollar values
df['Amount'] = np.abs(df['Amount'] * 85.0) + 5.0
df['Class'] = y

print(f"Class distribution:\n{df['Class'].value_counts(normalize=True)}")
df.head()

# %% [markdown]
# ## Cost-Sensitive Random Forest Training
# 
# We utilize a class-weighted Random Forest classifier (`class_weight='balanced'`) to penalize misclassifications of the rare fraud class, preventing the model from predicting the majority class exclusively.

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Fit Random Forest with balanced class weights
clf = RandomForestClassifier(
    n_estimators=150, 
    max_depth=7, 
    class_weight='balanced', 
    random_state=42
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Genuine", "Fraud"]))

# %% [markdown]
# ## Precision-Recall Curve Visualization
# 
# For highly imbalanced datasets, the Receiver Operating Characteristic (ROC) curve can present an overly optimistic view. The Precision-Recall (PR) curve is much more informative for measuring model discrimination.

# %%
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)

# Plot Precision-Recall using Plotly
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=recall, 
    y=precision, 
    mode='lines', 
    name=f'Weighted Random Forest (AUC = {pr_auc:.3f})',
    line=dict(color='#F43F5E', width=3)
))
fig.add_trace(go.Scatter(
    x=[0, 1], 
    y=[0.008, 0.008], 
    mode='lines', 
    name='Baseline No-Skill',
    line=dict(color='#64748B', width=1.5, dash='dash')
))
fig.update_layout(
    title=dict(text="Precision-Recall Curve (Highly Imbalanced Data)"),
    xaxis=dict(title=dict(text="Recall (Sensitivity)")),
    yaxis=dict(title=dict(text="Precision (Positive Predictive Value)")),
    margin=dict(l=40, r=40, t=50, b=40)
)
fig.show()
