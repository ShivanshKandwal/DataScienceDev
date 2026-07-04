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
    id: 'a2c_cartpole',
    title: "Advantage Actor-Critic (A2C) Agent Simulator",
    description: "Trains an Advantage Actor-Critic (A2C) agent in PyTorch to stabilize a simplified cartpole. Charts convergence progression.",
    category: 'AI',
    notebookUrl: './notebooks/a2c_cartpole.ipynb',
    tags: ["PyTorch", "Actor-Critic", "A2C", "Matplotlib"],
    date: '2026-07-04',
    metrics: {
     
      "Actor Layers": "2",
      "Critic Layers": "2",
      "Convergence Ep": "160"

    }
  },
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
    id: 'autoencoder_anomaly',
    title: "Autoencoder Anomaly Detection Pipeline",
    description: "Builds a PyTorch Autoencoder to reconstruct normal sensor telemetry. Pinpoints anomalous spikes based on reconstruction loss thresholds.",
    category: 'AI',
    notebookUrl: './notebooks/autoencoder_anomaly.ipynb',
    tags: ["PyTorch", "Autoencoder", "Anomaly Detection", "Matplotlib"],
    date: '2026-07-02',
    metrics: {
     
      "Input Size": "64",
      "Latent Dim": "8",
      "Precision": "91.2%"

    }
  },
  {
    id: 'bi_lstm_returns',
    title: "Bidirectional LSTM Sequence Return Forecaster",
    description: "Trains a Bidirectional LSTM in PyTorch to forecast sequential asset returns. Compares actual test signals against prediction bounds.",
    category: 'AI',
    notebookUrl: './notebooks/bi_lstm_returns.ipynb',
    tags: ["PyTorch", "LSTM", "Bidirectional", "Matplotlib"],
    date: '2026-07-03',
    metrics: {
     
      "Hidden Size": "32",
      "Sequence Length": "10",
      "Test Loss (MSE)": "0.024"

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
    id: 'clv_prediction',
    title: "Customer Lifetime Value (CLV) Prediction",
    description: "Predicts customer lifetime valuations using regularized linear estimators. Analyzes buying frequency and relationship tenure weights.",
    category: 'ML',
    notebookUrl: './notebooks/clv_prediction.ipynb',
    tags: ["Scikit-Learn", "Regression", "CLV", "Plotly"],
    date: '2026-07-02',
    metrics: {
     
      "R\u00b2 Score": "0.834",
      "MAE": "$1.64k",
      "Dataset Size": "1200"

    }
  },
  {
    id: 'cohort_retention',
    title: "Customer Cohort Retention Heatmap Analysis",
    description: "Analyzes user signups and retention trends across monthly cohorts. Generates SaaS retention decay heatmaps using Seaborn.",
    category: 'EDA',
    notebookUrl: './notebooks/cohort_retention.ipynb',
    tags: ["EDA", "SaaS", "Cohort Analysis", "Seaborn"],
    date: '2026-07-02',
    metrics: {
     
      "User Cohorts": "12 Months",
      "Total Users": "4500",
      "Month-12 Avg": "24.2%"

    }
  },
  {
    id: 'color_quantization',
    title: "K-Means Image Color Quantization",
    description: "Compresses color dimensions of simulated RGB arrays using K-Means clustering. Compares original vs quantized representations.",
    category: 'ML',
    notebookUrl: './notebooks/color_quantization.ipynb',
    tags: ["Scikit-Learn", "K-Means", "Quantization", "Matplotlib"],
    date: '2026-07-02',
    metrics: {
     
      "Cluster Counts": "16, 8, 4",
      "Image Dimensions": "64x64x3",
      "Quantized MSE": "0.012"

    }
  },
  {
    id: 'continuous_a2c',
    title: "Continuous Action Space A2C Tracking Agent",
    description: "Trains a continuous Gaussian A2C reinforcement learning agent to hold coordinate targets in a 1D physics sandbox. Charts tracking errors.",
    category: 'AI',
    notebookUrl: './notebooks/continuous_a2c.ipynb',
    tags: ["PyTorch", "A2C", "Continuous Control", "Matplotlib"],
    date: '2026-07-04',
    metrics: {
     
      "State Dim": "2",
      "Latent Hidden": "32",
      "Convergence Ep": "150"

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
    id: 'decision_tree',
    title: "Decision Tree Classification & Overfitting Boundaries",
    description: "Trains Decision Tree classifiers to map non-linear spaces. Compares boundary complexity and overfitting limits across tree depth bounds.",
    category: 'ML',
    notebookUrl: './notebooks/decision_tree.ipynb',
    tags: ["Scikit-Learn", "Decision Tree", "Classification", "Matplotlib"],
    date: '2026-07-03',
    metrics: {
     
      "Max Depth": "2, 4, None",
      "Features": "2",
      "Accuracy": "89.2%"

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
    id: 'flight_routes',
    title: "Global Flight Path Connections Map",
    description: "Models flight networks across global coordinates. Renders 3D orthographic globe connections using Plotly Scattergeo.",
    category: 'EDA',
    notebookUrl: './notebooks/flight_routes.ipynb',
    tags: ["EDA", "Geospatial", "Network Graph", "Plotly"],
    date: '2026-07-02',
    metrics: {
     
      "Route Count": "12",
      "Cities Connected": "6",
      "Max Range": "12k km"

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
    id: 'funnel_analysis',
    title: "Web Clickstream Funnel Conversion Analysis",
    description: "Models user journeys and conversion funnel decay rates from web log sequences. Builds interactive Plotly funnels.",
    category: 'EDA',
    notebookUrl: './notebooks/funnel_analysis.ipynb',
    tags: ["EDA", "Clickstream", "Conversion Funnel", "Plotly"],
    date: '2026-07-02',
    metrics: {
     
      "Funnel Steps": "4",
      "Conversion Rate": "3.2%",
      "Session Count": "10k"

    }
  },
  {
    id: 'game_of_life',
    title: "Conway's Game of Life Cellular Automata",
    description: "Simulates Conway's Game of Life on a 50x50 grid. Tracks structural transformations (gliders, blinkers) over generations.",
    category: 'EDA',
    notebookUrl: './notebooks/game_of_life.ipynb',
    tags: ["Simulation", "Cellular Automata", "Data Visualization", "Matplotlib"],
    date: '2026-07-02',
    metrics: {
     
      "Grid Size": "50x50",
      "Generations": "50",
      "Structures": "Gliders"

    }
  },
  {
    id: 'gan_1d',
    title: "Generative Adversarial Network (GAN) 1D Learning",
    description: "Builds Generator and Discriminator neural networks in PyTorch to replicate 1D distributions. Compares real vs generated density curves.",
    category: 'AI',
    notebookUrl: './notebooks/gan_1d.ipynb',
    tags: ["GAN", "PyTorch", "Generative AI", "Matplotlib"],
    date: '2026-07-02',
    metrics: {
     
      "Gen Layers": "3",
      "Disc Layers": "3",
      "Epochs": "50"

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
    id: 'isolation_forest',
    title: "Isolation Forest Anomaly & Outlier Mapping",
    description: "Constructs an Isolation Forest estimator to identify structural anomalies. Visualizes coordinate outliers and score contour gradients.",
    category: 'ML',
    notebookUrl: './notebooks/isolation_forest.ipynb',
    tags: ["Scikit-Learn", "Isolation Forest", "Anomaly Detection", "Matplotlib"],
    date: '2026-07-04',
    metrics: {
     
      "Inlier Count": "350",
      "Outlier Count": "50",
      "Accuracy (F1)": "90.4%"

    }
  },
  {
    id: 'knn_classification',
    title: "KNN Classifier & Optimal K Selection",
    description: "Trains a K-Nearest Neighbors classifier to estimate diagnostic risks. Visualizes cross-validation accuracy bounds across neighbor parameters.",
    category: 'ML',
    notebookUrl: './notebooks/knn_classification.ipynb',
    tags: ["Scikit-Learn", "KNN", "Classification", "Matplotlib"],
    date: '2026-07-02',
    metrics: {
     
      "Accuracy": "80.0%",
      "Neighbors K": "9",
      "Dimensions": "4"

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
    id: 'pca_reduction',
    title: "Principal Component Analysis (PCA) Dimension Reduction",
    description: "Decomposes 4D feature vectors into 2 principal components. Maps class separations and evaluates cumulative explained variances.",
    category: 'ML',
    notebookUrl: './notebooks/pca_reduction.ipynb',
    tags: ["Scikit-Learn", "PCA", "Dimensionality Reduction", "Matplotlib"],
    date: '2026-07-04',
    metrics: {
     
      "Input Dim": "4",
      "Reduced Dim": "2",
      "Explained Var": "91.6%"

    }
  },
  {
    id: 'population_pyramid',
    title: "Demographic Population Pyramid Analysis",
    description: "Models demographic distributions split by age cohorts and gender. Generates mirrored horizontal bar charts using Plotly.",
    category: 'EDA',
    notebookUrl: './notebooks/population_pyramid.ipynb',
    tags: ["EDA", "Demographics", "Population Pyramid", "Plotly"],
    date: '2026-07-04',
    metrics: {
     
      "Age Groups": "10",
      "Cohort Width": "10 Years",
      "Total Pop": "100k"

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
    id: 'reinforce_cartpole',
    title: "Policy Gradient REINFORCE Agent Simulator",
    description: "Trains a Policy Gradient (REINFORCE) reinforcement learning agent in PyTorch to stabilize a simplified cartpole. Plots reward progression.",
    category: 'AI',
    notebookUrl: './notebooks/reinforce_cartpole.ipynb',
    tags: ["PyTorch", "Policy Gradient", "REINFORCE", "Matplotlib"],
    date: '2026-07-03',
    metrics: {
     
      "State Dim": "4",
      "Action Dim": "2",
      "Convergence Ep": "180"

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
    id: 'taxi_qlearning',
    title: "Q-Learning Taxi Navigation Solver",
    description: "Trains a tabular Q-learning reinforcement learning agent to route a taxi in a 5x5 grid world to pick up and drop off riders.",
    category: 'AI',
    notebookUrl: './notebooks/taxi_qlearning.ipynb',
    tags: ["Q-Learning", "Reinforcement Learning", "Grid World", "Matplotlib"],
    date: '2026-07-02',
    metrics: {
     
      "State Space": "125",
      "Action Space": "6",
      "Convergence Ep": "150"

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
  },
  {
    id: 'vae_latent',
    title: "Variational Autoencoder (VAE) Latent Clustering",
    description: "Builds a Variational Autoencoder (VAE) in PyTorch to map digit representations to a 2D latent space. Visualizes clustered latent regions.",
    category: 'AI',
    notebookUrl: './notebooks/vae_latent.ipynb',
    tags: ["PyTorch", "VAE", "Generative AI", "Matplotlib"],
    date: '2026-07-03',
    metrics: {
     
      "Latent Dim": "2",
      "Loss (ELBO)": "42.15",
      "Batch Size": "64"

    }
  },
  {
    id: 'violin_distributions',
    title: "Multi-Group Violin Probability Density Analysis",
    description: "Models probability density profiles across categorical groups. Generates annotated multi-group Violin plots using Seaborn.",
    category: 'EDA',
    notebookUrl: './notebooks/violin_distributions.ipynb',
    tags: ["EDA", "Statistical Visualization", "Violin Plot", "Seaborn"],
    date: '2026-07-04',
    metrics: {
     
      "Target Groups": "4",
      "Sample Size": "800",
      "Width Parameter": "0.8"

    }
  }
];
