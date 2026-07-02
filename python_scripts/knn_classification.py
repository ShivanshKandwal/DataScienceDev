# ---
# title: "KNN Classifier & Optimal K Selection"
# description: "Trains a K-Nearest Neighbors classifier to estimate diagnostic risks. Visualizes cross-validation accuracy bounds across neighbor parameters."
# category: "ML"
# tags: ["Scikit-Learn", "KNN", "Classification", "Matplotlib"]
# date: "2026-07-02"
# metrics:
#   Accuracy: "80.0%"
#   Neighbors K: "9"
#   Dimensions: "4"
# ---

# %% [markdown]
# # KNN Classifier & Optimal K Selection
# 
# The K-Nearest Neighbors (KNN) algorithm classifies instances based on the majority label of their nearest neighbors. Choosing the hyperparameter $K$ (number of neighbors) is critical: too low causes overfitting, while too high causes underfitting. This project fits a KNN diagnostic classifier and runs a parameters search to discover the optimal value of $K$.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Generate synthetic health diagnostic dataset (500 samples)
np.random.seed(99)
n_samples = 500

glucose = np.random.normal(105, 25, n_samples)
bmi = np.random.normal(27, 5, n_samples)
age = np.random.normal(45, 12, n_samples)
blood_pressure = np.random.normal(78, 10, n_samples)

# Classification rule with overlapping boundary noise
p_risk = (
    0.05 
    + 0.35 * (glucose > 125) 
    + 0.25 * (bmi > 32) 
    + 0.20 * (age > 55)
    + 0.10 * (blood_pressure > 90)
).clip(0, 1)

y = np.random.binomial(1, p_risk)

df = pd.DataFrame({
    'Glucose': glucose,
    'BMI': bmi,
    'Age': age,
    'BloodPressure': blood_pressure,
    'Target': y
})

df.head(10)

# %% [markdown]
# ## Data Scaling & Train-Test Partition
# 
# Because KNN calculates Euclidean distances between samples, feature scaling is essential. We standard-scale features and split them into 80/20 train/test segments.

# %%
X = df.drop(columns=['Target'])
y_target = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.2, random_state=42, stratify=y_target)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ## Optimal K Hyperparameter Selection
# 
# We evaluate neighbor values from $K=1$ to $K=25$. For each configuration, we compute the 5-fold cross-validation accuracy on the training partition to locate the optimal $K$ choice.

# %%
k_values = range(1, 26)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal number of neighbors (K): {optimal_k}")
print(f"Max cross-validation accuracy: {max(cv_scores)*100:.2f}%")

# Train final model
final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
final_knn.fit(X_train_scaled, y_train)
test_accuracy = final_knn.score(X_test_scaled, y_test)
print(f"Final test partition accuracy: {test_accuracy*100:.2f}%")

# %% [markdown]
# ## Cross-Validation Accuracy Curve
# 
# We plot the cross-validation accuracies against values of $K$ using Matplotlib, highlighting the optimal hyperparameter value.

# %%
plt.figure(figsize=(9, 5))
plt.plot(k_values, cv_scores, marker='o', color='#2563EB', linewidth=2, markersize=6)
plt.axvline(optimal_k, color='#EF4444', linestyle='--', linewidth=1.5, label=f'Optimal K = {optimal_k}')

plt.title('KNN Hyperparameter Tuning: K vs. CV Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Number of Neighbors (K)', fontsize=12)
plt.ylabel('5-Fold Cross-Validation Accuracy', fontsize=12)
plt.xticks(range(1, 26, 2))
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()
