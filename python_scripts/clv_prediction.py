# ---
# title: "Customer Lifetime Value (CLV) Prediction"
# description: "Predicts customer lifetime valuations using regularized linear estimators. Analyzes buying frequency and relationship tenure weights."
# category: "ML"
# tags: ["Scikit-Learn", "Regression", "CLV", "Plotly"]
# date: "2026-07-02"
# metrics:
#   R² Score: "0.834"
#   MAE: "$1.64k"
#   Dataset Size: "1200"
# ---

# %% [markdown]
# # Customer Lifetime Value (CLV) Prediction
# 
# Customer Lifetime Value (CLV) represents the total revenue a business expects to earn from a customer. Predicting CLV helps target high-value customers and optimize marketing costs. This project builds a regularized Ridge regression model on synthetic customer purchase behaviors to predict lifetime value.

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# Generate synthetic dataset (1,200 customers)
np.random.seed(88)
n_customers = 1200

tenure_months = np.random.randint(1, 60, n_customers)
purchase_frequency = np.random.normal(4, 1.5, n_customers).clip(0.5, 12)
avg_order_value = np.random.normal(65, 20, n_customers).clip(10, 200)
support_tickets = np.random.poisson(1.5, n_customers)

# Formula for CLV with non-linear noise
clv = (
    tenure_months * purchase_frequency * avg_order_value * 0.95
    - support_tickets * 15.0
    + np.random.normal(0, 150, n_customers)
).clip(50, None)

df = pd.DataFrame({
    'Tenure_Months': tenure_months,
    'Purchase_Frequency': purchase_frequency,
    'Avg_Order_Value': avg_order_value,
    'Support_Tickets': support_tickets,
    'CLV': clv
})

df.head(10)

# %% [markdown]
# ## Regression Model Fitting
# 
# We split the data into training (80%) and testing (20%) partitions, standardize continuous features, and fit a regularized Ridge regression estimator.

# %%
X = df.drop(columns=['CLV'])
y = df['CLV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Ridge regressor
model = Ridge(alpha=5.0)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Regression Performance Metrics:")
print(f"R² Score:  {r2:.4f}")
print(f"MAE:       ${mae:,.2f}")

# %% [markdown]
# ## Predictions Scatter Plot with Plotly
# 
# We plot actual CLV vs. predicted values for the test partition, adding a reference line marking perfect prediction.

# %%
# Convert numpy arrays to standard lists for json serialization compatibility
y_test_list = list(y_test)
y_pred_list = list(y_pred)

fig = go.Figure()

# Scatter of actual vs predicted
fig.add_trace(go.Scatter(
    x=y_test_list,
    y=y_pred_list,
    mode='markers',
    name=f'Predictions (R²: {r2:.3f})',
    marker=dict(color='#8B5CF6', opacity=0.6, size=6)
))

# Perfect alignment diagonal line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
fig.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode='lines',
    name='Ideal Prediction',
    line=dict(color='#EF4444', dash='dash', width=2)
))

fig.update_layout(
    title='CLV Model Predictions: Predicted vs Actual Values',
    xaxis_title='Actual Lifetime Value ($)',
    yaxis_title='Predicted Lifetime Value ($)',
    legend_title='Series',
    template='plotly_white',
    width=650,
    height=480
)

fig.show()
