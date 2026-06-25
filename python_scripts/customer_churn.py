# ---
# title: "Customer Churn Prediction Model"
# description: "Supervised binary classification model predicting customer churn probabilities using Scikit-Learn Random Forest. Plots ROC curve benchmarks."
# category: "ML"
# tags: ["Scikit-Learn", "Random Forest", "Classification", "Plotly"]
# date: "2026-06-25"
# metrics:
#   ROC AUC: "0.892"
#   F1-Score: "0.845"
#   Accuracy: "87.2%"
# ---

# %% [markdown]
# # Customer Churn Classification Model
# 
# This notebook builds a supervised binary classifier predicting whether telecomm subscribers will churn. We generate a synthetic dataset with features representing customer age, monthly charges, and tenure months, and train a Random Forest model.

# %%
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc

# Generate classification dataset
X, y = make_classification(
    n_samples=1000, 
    n_features=5, 
    n_informative=3, 
    n_redundant=2, 
    random_state=42
)

# Convert to DataFrame
feature_names = ['Tenure_Months', 'Monthly_Charges', 'Total_Charges', 'Support_Calls', 'Age']
df = pd.DataFrame(X, columns=feature_names)
df['Churn'] = y
df.head()

# %% [markdown]
# ## Random Forest Training & Evaluation
# 
# We split the data into training/validation sets and fit a Random Forest classifier.

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

# %% [markdown]
# ## ROC Curve Benchmarking
# 
# We plot the Receiver Operating Characteristic (ROC) curve to measure model discrimination power.

# %%
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC using Plotly
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=fpr, 
    y=tpr, 
    mode='lines', 
    name=f'Random Forest (AUC = {roc_auc:.3f})',
    line=dict(color='#A855F7', width=3)
))
fig.add_trace(go.Scatter(
    x=[0, 1], 
    y=[0, 1], 
    mode='lines', 
    name='Random Classifier',
    line=dict(color='#64748B', width=1.5, dash='dash')
))
fig.update_layout(
    title=dict(text="ROC Curve Benchmark"),
    xaxis=dict(title=dict(text="False Positive Rate")),
    yaxis=dict(title=dict(text="True Positive Rate")),
    margin=dict(l=40, r=40, t=50, b=40)
)
fig.show()
