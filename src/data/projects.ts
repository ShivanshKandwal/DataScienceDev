export interface ProjectData {
  id: string;
  title: string;
  description: string;
  category: 'EDA' | 'ML' | 'AI';
  notebookUrl: string;
  dashboardUrl?: string;
  dashboardImage?: string;
  tags: string[];
  date: string;
  metrics: { [key: string]: string };
}

export const projectsData: ProjectData[] = [
  {
    id: 'air_quality_decomposition',
    title: "Air Quality Time-Series & Seasonal Decomposition",
    description: "Simulates daily air quality indices and performs classical additive seasonal trend decomposition, profiling monthly variation boxplots.",
    category: 'EDA',
    notebookUrl: './notebooks/air_quality_decomposition.ipynb',
    tags: ["Time Series", "EDA", "Seaborn", "Matplotlib"],
    date: '2026-06-28',
    metrics: {
     
      "Days Analyzed": "1095",
      "Monthly Peak": "142.5",
      "Trend Coefficient": "+2.4 / yr"

    }
  },
  {
    id: 'brain_tumor_segment',
    title: "Brain Tumor MRI Segmentation U-Net",
    description: "Pixel-level segmentation of MRI scans using a custom PyTorch U-Net architecture. Employs dice loss coefficient optimizations and data augmentation matrices.",
    category: 'AI',
    notebookUrl: './notebooks/brain_tumor_segment.ipynb',
    dashboardImage: 'https://images.unsplash.com/photo-1559757175-5700dde675bc?auto=format&fit=crop&w=800&q=80',
    tags: ["PyTorch", "U-Net", "CNN", "Deep Learning"],
    date: '2026-06-25',
    metrics: {
     
      "IoU Acc": "96.8%",
      "Dice Coeff": "0.942",
      "Epochs": "120"

    }
  },
  {
    id: 'customer_churn',
    title: "Customer Churn Prediction Model",
    description: "Supervised binary classification model predicting customer churn probabilities using Scikit-Learn Random Forest. Plots ROC curve benchmarks.",
    category: 'ML',
    notebookUrl: './notebooks/customer_churn.ipynb',
    tags: ["Scikit-Learn", "Random Forest", "Classification", "Plotly"],
    date: '2026-06-25',
    metrics: {
     
      "ROC AUC": "0.892",
      "F1-Score": "0.845",
      "Accuracy": "87.2%"

    }
  },
  {
    id: 'dqn_cartpole',
    title: "Deep Q-Network (DQN) CartPole Simulation",
    description: "Implements a custom CartPole reinforcement learning physics simulator and trains a PyTorch DQN agent with experience replay and target networks.",
    category: 'AI',
    notebookUrl: './notebooks/dqn_cartpole.ipynb',
    tags: ["DQN", "Deep RL", "PyTorch", "Matplotlib"],
    date: '2026-06-27',
    metrics: {
     
      "State Space": "4",
      "Action Space": "2",
      "Avg Reward": "185.4"

    }
  },
  {
    id: 'fraud_detection',
    title: "Credit Card Fraud Detection",
    description: "Supervised anomaly detection on highly imbalanced transaction manifests. Implements cost-sensitive Random Forest classifiers and Precision-Recall curve benchmarks.",
    category: 'ML',
    notebookUrl: './notebooks/fraud_detection.ipynb',
    tags: ["Scikit-Learn", "Anomaly Detection", "Imbalanced Data", "Plotly"],
    date: '2026-06-26',
    metrics: {
     
      "Fraud Rate": "0.8%",
      "PR-AUC": "0.862",
      "Recall@90%Prec": "82.4%"

    }
  },
  {
    id: 'house_valuation',
    title: "House Valuation Regression Modeling",
    description: "Evaluates property prices using Ridge Regression and Random Forest. Performs feature scaling, handles multicollinearity, and visualizes residuals using interactive Plotly scatter plots.",
    category: 'ML',
    notebookUrl: './notebooks/house_valuation.ipynb',
    tags: ["Scikit-Learn", "Regression", "Random Forest", "Plotly"],
    date: '2026-06-27',
    metrics: {
     
      "R\u00b2 Score": "0.892",
      "RMSE": "$32.4k",
      "Dataset Size": "1500"

    }
  },
  {
    id: 'housing_regression',
    title: "Housing Market Prices Regression Model",
    description: "Predictive pricing utilizing Random Forest and XGBoost Regressor. Implements cross-validated hyperparameter tuning and recursive feature elimination.",
    category: 'ML',
    notebookUrl: './notebooks/housing_regression.ipynb',
    dashboardUrl: 'https://share.streamlit.io/house-regression-model',
    dashboardImage: 'https://images.unsplash.com/photo-1560518883-ce09059eeffa?auto=format&fit=crop&w=800&q=80',
    tags: ["XGBoost", "Scikit-Learn", "Random Forest", "Plotly"],
    date: '2026-06-23',
    metrics: {
     
      "R\u00b2 Score": "0.915",
      "CV Folds": "10",
      "RMSE": "$14.2k"

    }
  },
  {
    id: 'iris_clustering',
    title: "Iris Flowers Clustering & PCA",
    description: "Dimensionality reduction using PCA and cluster partitioning using K-Means on the classic Iris dataset. Visualizes 2D projection clusters.",
    category: 'EDA',
    notebookUrl: './notebooks/iris_clustering.ipynb',
    tags: ["Scikit-Learn", "PCA", "K-Means", "Matplotlib"],
    date: '2026-06-25',
    metrics: {
     
      "Clusters": "3",
      "PCA Variance": "92.4%",
      "Samples": "150"

    }
  },
  {
    id: 'lstm_sentiment',
    title: "LSTM Text Sentiment Classifier",
    description: "Trains a PyTorch Recurrent Neural Network (LSTM) with token embedding layers to categorize sentiment polarity of synthetic reviews.",
    category: 'AI',
    notebookUrl: './notebooks/lstm_sentiment.ipynb',
    tags: ["PyTorch", "LSTM", "NLP", "Matplotlib"],
    date: '2026-06-29',
    metrics: {
     
      "Vocab Size": "30",
      "Hidden Dim": "16",
      "Accuracy": "95.0%"

    }
  },
  {
    id: 'maze_qlearning',
    title: "Reinforcement Learning Q-Maze Solver",
    description: "Tabular Q-learning agent training in a stateful Gridworld maze environment. Visualizes cumulative episode rewards and the learned path arrow grid.",
    category: 'AI',
    notebookUrl: './notebooks/maze_qlearning.ipynb',
    tags: ["Q-Learning", "Reinforcement Learning", "Pathfinding", "Matplotlib"],
    date: '2026-06-26',
    metrics: {
     
      "Grid World": "6x6",
      "Episodes": "300",
      "Path Steps": "10"

    }
  },
  {
    id: 'mnist_classifier',
    title: "MNIST Handwritten Digits CNN",
    description: "Building a convolutional neural network in PyTorch to classify handwritten digits. Visualizes predicted digit masks and weights.",
    category: 'AI',
    notebookUrl: './notebooks/mnist_classifier.ipynb',
    tags: ["PyTorch", "CNN", "Deep Learning", "Matplotlib"],
    date: '2026-06-25',
    metrics: {
     
      "Test Acc": "99.1%",
      "Epochs": "5",
      "Parameters": "21.8k"

    }
  },
  {
    id: 'movie_recommender',
    title: "Collaborative Filtering Movie Recommender",
    description: "Implements user-based collaborative filtering from scratch using cosine similarities. Generates rating prediction heatmaps and recommendation bars.",
    category: 'ML',
    notebookUrl: './notebooks/movie_recommender.ipynb',
    tags: ["Recommendation", "Collaborative Filtering", "Cosine Similarity", "Plotly"],
    date: '2026-06-29',
    metrics: {
     
      "User Count": "100",
      "Movie Count": "20",
      "Average Rating": "3.62"

    }
  },
  {
    id: 'predictive_maintenance',
    title: "Predictive Maintenance XGBoost Classifier",
    description: "Trains a gradient boosting classifier to predict industrial machine failure modes from sensor logs. Analyzes feature importance scores.",
    category: 'ML',
    notebookUrl: './notebooks/predictive_maintenance.ipynb',
    tags: ["XGBoost", "Classification", "Feature Importance", "Plotly"],
    date: '2026-06-28',
    metrics: {
     
      "ROC AUC": "0.868",
      "Accuracy": "92.0%",
      "Features": "6"

    }
  },
  {
    id: 'social_network_graph',
    title: "Social Network Centrality & Link Prediction",
    description: "Analyzes network structures using NetworkX. Computes degree/betweenness centralities and suggests friendships via Jaccard link predictions.",
    category: 'EDA',
    notebookUrl: './notebooks/social_network_graph.ipynb',
    tags: ["NetworkX", "Graph Analysis", "Link Prediction", "Matplotlib"],
    date: '2026-06-29',
    metrics: {
     
      "Nodes (Users)": "45",
      "Edges (Friendships)": "86",
      "Avg Centrality": "0.09"

    }
  },
  {
    id: 'stock_analysis',
    title: "Stock Market Technical Analysis & Indicators",
    description: "Simulates stock price trends and calculates critical technical indicators (SMA, EMA, RSI) for quantitative analysis and feature engineering.",
    category: 'EDA',
    notebookUrl: './notebooks/stock_analysis.ipynb',
    tags: ["Time Series", "Technical Indicators", "Finance", "Matplotlib"],
    date: '2026-06-27',
    metrics: {
     
      "Time Range": "252 Days",
      "Max Price": "$248.5",
      "Final RSI": "58.2"

    }
  },
  {
    id: 'svm_boundaries',
    title: "SVM Decision Boundaries & Hyperparameter Tuning",
    description: "Trains Support Vector Classifiers with varying kernel parameters (Linear, Poly, RBF) on non-linear dataset. Plots 2x2 decision boundary grid layouts.",
    category: 'ML',
    notebookUrl: './notebooks/svm_boundaries.ipynb',
    tags: ["Scikit-Learn", "SVM", "Classification", "Matplotlib"],
    date: '2026-06-28',
    metrics: {
     
      "Accuracy": "94.5%",
      "Support Vectors": "84",
      "Kernels Tested": "3"

    }
  },
  {
    id: 'temperature_anomalies',
    title: "Global Temperature Anomalies & Volcanic Activity Correlation",
    description: "Simulates historical temperature deviations and models temporary volcanic aerosol cooling effects. Generates double-axis timeline Plotly plots.",
    category: 'EDA',
    notebookUrl: './notebooks/temperature_anomalies.ipynb',
    tags: ["EDA", "Correlation", "Climate Analysis", "Plotly"],
    date: '2026-06-28',
    metrics: {
     
      "Time Span": "150 Years",
      "Correlation Index": "-0.42",
      "Anomalies Recorded": "150"

    }
  },
  {
    id: 'text_topic_modeling',
    title: "Product Reviews Topic Modeling & Sentiment",
    description: "NLP pipeline combining sentiment classification and unsupervised topic modeling on customer feedback. Implements TF-IDF vectorization and NMF decomposition.",
    category: 'EDA',
    notebookUrl: './notebooks/text_topic_modeling.ipynb',
    tags: ["NLP", "TF-IDF", "Topic Modeling", "Seaborn"],
    date: '2026-06-26',
    metrics: {
     
      "Reviews": "100",
      "Topics": "4",
      "Vocabulary": "148"

    }
  },
  {
    id: 'titanic_eda',
    title: "Titanic Survival Exploratory Data Analysis",
    description: "Data-mining passenger manifest records to extract survival patterns. Features extensive categorical data engineering, missing value imputations, and seaborn density plots.",
    category: 'EDA',
    notebookUrl: './notebooks/titanic_eda.ipynb',
    tags: ["Python", "Pandas", "Seaborn", "Matplotlib"],
    date: '2026-06-21',
    metrics: {
     
      "Records": "891",
      "Features": "12",
      "Correlations": "0.78"

    }
  },
  {
    id: 'transfer_learning_resnet',
    title: "Transfer Learning ResNet Image Classifier",
    description: "Simulates transfer learning using a PyTorch classifier head on top of pre-extracted ResNet features. Plots training curves and prediction probability bars.",
    category: 'AI',
    notebookUrl: './notebooks/transfer_learning_resnet.ipynb',
    tags: ["PyTorch", "Transfer Learning", "Image Classification", "Matplotlib"],
    date: '2026-06-29',
    metrics: {
     
      "Input Shape": "512 Features",
      "Classes": "3",
      "Accuracy": "88.5%"

    }
  }
];
