# ---
# title: "MLP Credit Risk Classifier Pipeline"
# description: "Builds a multi-layer feedforward network with Batch Normalization to predict credit risk."
# category: "AI"
# tags: ["PyTorch", "MLP", "Batch Normalization", "Matplotlib"]
# date: "2026-07-08"
# metrics:
#   Layer Dimensions: "32-16-1"
#   Dropout Ratio: "0.2"
#   Accuracy: "71.9%"
# ---

# %% [markdown]
# # MLP Credit Risk Classifier Pipeline
# 
# Multi-Layer Perceptrons (MLPs) classify structured/tabular data tables. To prevent overfitting and speed up convergence in deep architectures, layers are augmented with **Batch Normalization** (which normalizes activations within mini-batches) and **Dropout** (which randomly drops neuron activations). This notebook trains a credit risk MLP classifier in PyTorch, fits it on 800 synthetic consumer profiles (10 attributes), and plots BCE loss decay.

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Generate synthetic consumer credit risk logs (800 profiles, 10 attributes)
np.random.seed(265)
n_profiles = 800
n_attributes = 10

X = np.random.normal(0, 1.0, (n_profiles, n_attributes))
# Target Risk is correlated to specific features (e.g. debt, income indexes)
risk_logits = 1.2 * X[:, 0] - 0.85 * X[:, 2] + 0.5 * X[:, 4] - 0.25
probs = 1 / (1 + np.exp(-risk_logits))
y = np.random.binomial(1, probs)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

df_risk = pd.DataFrame(X, columns=[f"Attr_{i+1:02d}" for i in range(n_attributes)])
df_risk['Default_Risk'] = y
df_risk.head(10)

# %% [markdown]
# ## MLP Network Architecture
# 
# We implement the MLP model in PyTorch:
# - **Layer 1**: 10 inputs -> 32 hidden units, followed by BatchNorm1d, ReLU, and 20% Dropout.
# - **Layer 2**: 32 units -> 16 hidden units, followed by BatchNorm1d, ReLU, and 20% Dropout.
# - **Layer 3**: 16 units -> 1 output unit, followed by Sigmoid.

# %%
class CreditRiskMLP(nn.Module):
    def __init__(self, in_dim=10):
        super(CreditRiskMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x).squeeze(1)

model = CreditRiskMLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# %% [markdown]
# ## Classifier Training Loop
# 
# We train the MLP model for 100 epochs.

# %%
epochs = 100
losses = []

for epoch in range(1, epochs + 1):
    model.train()
    predictions = model(X_train_t)
    loss = criterion(predictions, y_train_t)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}/{epochs} | BCE Training Loss: {loss.item():.4f}")

# Evaluate performance on test split
model.eval()
with torch.no_grad():
    test_preds = model(X_test_t).numpy()
    binary_preds = (test_preds > 0.5).astype(int)

acc = accuracy_score(y_test, binary_preds)
print(f"\nDefault Risk Test Accuracy: {acc*100:.2f}%")
print("\nClassification Report Summary:")
print(classification_report(y_test, binary_preds))

# %% [markdown]
# ## Loss Decay Plot
# 
# We chart the training BCE classification loss decay over epochs using Matplotlib.

# %%
plt.figure(figsize=(9, 5.5))
plt.plot(losses, color='#EC4899', linewidth=2.0, label='BCE Loss')

plt.title('MLP Credit Risk Classifier: Binary Cross-Entropy Loss Decay Curve', fontsize=13, fontweight='bold')
plt.xlabel('Training Epoch')
plt.ylabel('Loss (BCE)')
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend()
plt.show()
