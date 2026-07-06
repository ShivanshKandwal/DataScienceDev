# ---
# title: "Naive Bayes Text Likelihood Classification"
# description: "Trains a Multinomial Naive Bayes text classifier on synthetic vocabulary bag-of-words. Visualizes likelihood feature parameters."
# category: "ML"
# tags: ["Scikit-Learn", "Naive Bayes", "Classification", "NLP", "Matplotlib"]
# date: "2026-07-06"
# metrics:
#   Vocabulary Size: "100"
#   Accuracy: "100.0%"
#   Review Count: "400"
# ---

# %% [markdown]
# # Naive Bayes Text Likelihood Classification
# 
# Naive Bayes classifiers utilize Bayes' theorem under the conditional independence assumption. In Natural Language Processing (NLP), Multinomial Naive Bayes estimates text likelihoods based on word occurrences. This notebook creates a bag-of-words dataset containing 400 reviews across 100 vocabulary coordinates, trains Naive Bayes, and visualizes top class-likelihood indicator words using Matplotlib.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Define mock vocabulary words: index 0-9 positive indicators, index 10-19 negative indicators
vocab = [
    'excellent', 'amazing', 'perfect', 'love', 'great', 'awesome', 'enjoy', 'recommend', 'fantastic', 'delight',
    'awful', 'terrible', 'waste', 'worst', 'hate', 'boring', 'poor', 'bad', 'disappoint', 'annoying'
] + [f"word_{i}" for i in range(20, 100)] # Neutral filler words

np.random.seed(92)
n_reviews = 400

# Construct bag-of-words count vectors [400 reviews, 100 features]
X = np.random.poisson(0.15, (n_reviews, 100))
y = np.random.binomial(1, 0.5, n_reviews) # 1 = Positive, 0 = Negative

# Inject indicator counts depending on labels
for idx in range(n_reviews):
    label = y[idx]
    if label == 1:
        # Increase frequency of positive words (index 0-9)
        X[idx, :10] += np.random.poisson(1.2, 10)
    else:
        # Increase frequency of negative words (index 10-19)
        X[idx, 10:20] += np.random.poisson(1.2, 10)

df_text = pd.DataFrame(X, columns=vocab)
df_text['Sentiment'] = y
df_text.head(10)

# %% [markdown]
# ## Classifier Fitting & Log Likelihoods
# 
# We split reviews into 80/20 train/test segments, fit a Multinomial Naive Bayes estimator, and extract the log likelihood parameters of each word feature.

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Naive Bayes Text Classification Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Extract log probabilities for positive features: shape [2 classes, 100 features]
# Difference in log probabilities highlights class indicators
feature_importance = model.feature_log_prob_[1] - model.feature_log_prob_[0]

# %% [markdown]
# ## Top Word Indicators Visualization
# 
# We sort features by their log likelihood difference and plot the top 10 positive indicators vs. the top 10 negative indicators in a Matplotlib horizontal bar chart.

# %%
# Sort vocab features
sorted_indices = np.argsort(feature_importance)
top_neg_indices = sorted_indices[:10]
top_pos_indices = sorted_indices[-10:]

# Combine indicators
plot_indices = np.concatenate((top_neg_indices, top_pos_indices))
plot_words = [vocab[idx] for idx in plot_indices]
plot_scores = feature_importance[plot_indices]

# Set color codes
colors = ['#EF4444' if score < 0 else '#3B82F6' for score in plot_scores]

plt.figure(figsize=(10, 6.5))
plt.barh(plot_words, plot_scores, color=colors, edgecolor='black', alpha=0.8)

plt.title('Naive Bayes Feature Log-Likelihood Ratio: Positive vs. Negative Indicators', fontsize=13, fontweight='bold')
plt.xlabel('Log-Likelihood Difference (Positive Class - Negative Class)', fontsize=11)
plt.ylabel('Vocabulary Word Feature', fontsize=11)
plt.axvline(0.0, color='black', linestyle='-', linewidth=1.2)
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()
