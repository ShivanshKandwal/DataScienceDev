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
    id: 'a2c_continuous_control',
    title: "A2C Continuous Control Target Positioner",
    description: "Configures continuous controls using Gaussian output coordinate dimensions to hold vector targets in PyTorch.",
    category: 'AI',
    notebookUrl: './notebooks/a2c_continuous_control.ipynb',
    tags: ["PyTorch", "A2C", "Continuous Control", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "State Vector Size": "4",
      "Action Output Size": "2",
      "Convergence Ep": "160"

    }
  },
  {
    id: 'a2c_taxi_navigation',
    title: "A2C Gridworld Taxi Navigation Policy",
    description: "Upgrades tabular taxi agents to neural Actor-Critic approximation models in PyTorch.",
    category: 'AI',
    notebookUrl: './notebooks/a2c_taxi_navigation.ipynb',
    tags: ["PyTorch", "A2C", "Gridworld Navigation", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "State Dim": "2",
      "Hidden Layers": "2",
      "Convergence Ep": "140"

    }
  },
  {
    id: 'adaboost_regression',
    title: "AdaBoost Ensemble Regressor Predictions",
    description: "Fits boosting ensemble models to resolve noisy non-linear functions. Compares base vs. AdaBoost predictions in Matplotlib.",
    category: 'ML',
    notebookUrl: './notebooks/adaboost_regression.ipynb',
    tags: ["Scikit-Learn", "AdaBoost", "Ensemble", "Matplotlib"],
    date: '2026-07-05',
    metrics: {
     
      "Base Estimators": "50",
      "Learning Rate": "0.1",
      "R\u00b2 Score": "0.934"

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
    id: 'arima_forecasting',
    title: "Time-Series ARIMA Forecast Analysis",
    description: "Evaluates order params (p,d,q) to configure Autoregressive Integrated Moving Average sequence predictors.",
    category: 'ML',
    notebookUrl: './notebooks/arima_forecasting.ipynb',
    tags: ["Forecasting", "ARIMA", "Time Series", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Sequence Data Points": "365",
      "Order Settings": "(2,1,1)",
      "Test MAPE": "3.55%"

    }
  },
  {
    id: 'athlete_radar',
    title: "Athlete Multi-Dimensional Attribute Radar Profiles",
    description: "Maps multivariate athlete performance attributes. Builds filled polar radar chart overlays using Plotly.",
    category: 'EDA',
    notebookUrl: './notebooks/athlete_radar.ipynb',
    tags: ["EDA", "Multivariate Analysis", "Radar Chart", "Plotly"],
    date: '2026-07-05',
    metrics: {
     
      "Dimensions": "6",
      "Profiles Compared": "2",
      "Performance Index": "0-100"

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
    id: 'cnn_1d_ecg',
    title: "1D CNN ECG Sequence Classifier",
    description: "Implements a 1D CNN containing pooling layers to identify cardiac anomalies in ECG sequences.",
    category: 'AI',
    notebookUrl: './notebooks/cnn_1d_ecg.ipynb',
    tags: ["PyTorch", "1D CNN", "ECG Analysis", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Channels": "1",
      "Kernel Width": "5",
      "Anomaly Recall": "100.0%"

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
    id: 'collaborative_filtering',
    title: "Collaborative Filtering Item Similarity Recommender",
    description: "Builds item-to-item similarity matrices from purchase behaviors to calculate product recommendation outputs.",
    category: 'ML',
    notebookUrl: './notebooks/collaborative_filtering.ipynb',
    tags: ["Recommendation Engines", "Item-Similarity", "Cosine Metric", "Plotly"],
    date: '2026-07-08',
    metrics: {
     
      "User Database": "800",
      "Catalog Items": "120",
      "MAP Score": "0.185"

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
    id: 'conditional_vae',
    title: "Conditional VAE Coordinates Data Generator",
    description: "Implements a conditional VAE (CVAE) feeding labels to Latent dimensions to generate specific category grids.",
    category: 'AI',
    notebookUrl: './notebooks/conditional_vae.ipynb',
    tags: ["PyTorch", "Generative AI", "CVAE", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Latent Variables": "2",
      "Digit Classes": "3",
      "ELBO Loss": "0.104"

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
    id: 'correlation_heatmap',
    title: "Statistical Correlation Matrix Heatmap",
    description: "Models Pearson correlation matrices across synthetic variables. Renders annotated correlation heatmaps using Seaborn.",
    category: 'EDA',
    notebookUrl: './notebooks/correlation_heatmap.ipynb',
    tags: ["EDA", "Correlation Matrix", "Heatmap", "Seaborn"],
    date: '2026-07-06',
    metrics: {
     
      "Features": "6",
      "Sample Size": "500",
      "Max Correlation": "0.85"

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
    id: 'customer_journey_sunburst',
    title: "Customer Journey Path Sequence Sunburst Diagram",
    description: "Maps sequential website navigation paths to checkout as a hierarchical Sunburst chart in Plotly.",
    category: 'EDA',
    notebookUrl: './notebooks/customer_journey_sunburst.ipynb',
    tags: ["Clickstream", "Sunburst", "Customer Journeys", "Plotly"],
    date: '2026-07-08',
    metrics: {
     
      "Path Depth": "4 rings",
      "Sequence Rows": "3000",
      "Top Path Conversion": "18.2%"

    }
  },
  {
    id: 'dbscan_density',
    title: "DBSCAN Density-Based Spatial Clustering",
    description: "Clusters points based on density constraints (Eps, MinSamples), identifying arbitrary-shaped clusters and noise points.",
    category: 'ML',
    notebookUrl: './notebooks/dbscan_density.ipynb',
    tags: ["Clustering", "DBSCAN", "Density Clusters", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Epsilon Range": "0.35",
      "Minimum Core Points": "5",
      "Clustered Groups": "3"

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
    id: 'denoising_autoencoder',
    title: "Denoising Autoencoder Pattern Restorer",
    description: "Adds Gaussian noise to synthetic pixel arrays and trains deep autoencoders to reconstruct clean image grids.",
    category: 'AI',
    notebookUrl: './notebooks/denoising_autoencoder.ipynb',
    tags: ["PyTorch", "Denoising Autoencoder", "Image Compression", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Input Dimensions": "64",
      "Latent Dimensions": "8",
      "Noise Scale": "0.25"

    }
  },
  {
    id: 'double_dqn_learning',
    title: "DQN with Double Q-Learning (Double DQN)",
    description: "Implements Double DQN in PyTorch, decoupling actions selection from targets calculation.",
    category: 'AI',
    notebookUrl: './notebooks/double_dqn_learning.ipynb',
    tags: ["Double DQN", "Reinforcement Learning", "Target Stability", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "States Bins": "4",
      "Actions": "2",
      "Convergence Ep": "80"

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
    id: 'dqn_prioritized_replay',
    title: "DQN with Prioritized Experience Replay",
    description: "Modifies DQN replay queues to sample transitions scaled by absolute temporal difference (TD) errors in PyTorch.",
    category: 'AI',
    notebookUrl: './notebooks/dqn_prioritized_replay.ipynb',
    tags: ["DQN", "Prioritized Replay", "Reinforcement Learning", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Replay Size": "1000",
      "Priority Weight": "0.6",
      "Convergence Ep": "90"

    }
  },
  {
    id: 'dqn_target_network',
    title: "Target Network DQN CartPole Balancer",
    description: "Implements a Deep Q-Network (DQN) with a periodic target network update in PyTorch to stabilize a cartpole. Charts reward convergence.",
    category: 'AI',
    notebookUrl: './notebooks/dqn_target_network.ipynb',
    tags: ["PyTorch", "DQN", "Target Network", "Matplotlib"],
    date: '2026-07-06',
    metrics: {
     
      "State Dim": "4",
      "Replay Size": "1000",
      "Convergence Ep": "100"

    }
  },
  {
    id: 'dueling_dqn_learning',
    title: "DQN Dueling Architecture (Dueling DQN)",
    description: "Splits network heads to approximate State-Value V(s) and Action-Advantage A(s,a) streams independently.",
    category: 'AI',
    notebookUrl: './notebooks/dueling_dqn_learning.ipynb',
    tags: ["Dueling DQN", "DQN", "Reinforcement Learning", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Value Layers": "2",
      "Advantage Layers": "2",
      "Convergence Ep": "85"

    }
  },
  {
    id: 'ecommerce_funnel',
    title: "E-Commerce Conversion Funnel Heatmap Matrix",
    description: "Tracks conversion decays along different checkout stages across mobile OS, desktop browser, and app configurations.",
    category: 'EDA',
    notebookUrl: './notebooks/ecommerce_funnel.ipynb',
    tags: ["Product Analytics", "Funnel Matrix", "Heatmap", "Seaborn"],
    date: '2026-07-08',
    metrics: {
     
      "Funnel Levels": "4",
      "Traffic Channels": "6",
      "Conversions": "4.8%"

    }
  },
  {
    id: 'elasticnet_penalty_progression',
    title: "ElasticNet Regression L1 vs. L2 Penalty Progression",
    description: "Evaluates coefficient shrinkages and sparsity levels under varying combinations of ElasticNet L1/L2 weights.",
    category: 'ML',
    notebookUrl: './notebooks/elasticnet_penalty_progression.ipynb',
    tags: ["ElasticNet", "Regularization", "Sparsity", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Alpha Ranges": "0.01-10.0",
      "Active Coefficients": "17/20",
      "R\u00b2 Score": "0.958"

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
    id: 'gan_grid_pattern',
    title: "GAN 2D Grid Pattern Synthesizer",
    description: "Trains Generator/Discriminator modules to construct synthetic 2D grid coordinates.",
    category: 'AI',
    notebookUrl: './notebooks/gan_grid_pattern.ipynb',
    tags: ["GAN", "PyTorch", "Generative AI", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Generator Layers": "3",
      "Discriminator Layers": "3",
      "Epochs": "80"

    }
  },
  {
    id: 'global_earthquakes',
    title: "Global Earthquakes Scatterglobe Visualization",
    description: "Parses historical earthquake coordinates and plots magnitudes/depths onto a 3D spinning Scattergeo globe.",
    category: 'EDA',
    notebookUrl: './notebooks/global_earthquakes.ipynb',
    tags: ["Geospatial", "3D Globe", "Seismology", "Plotly"],
    date: '2026-07-08',
    metrics: {
     
      "Events": "1000",
      "Max Magnitude": "8.2",
      "Depth Range": "0-700km"

    }
  },
  {
    id: 'global_gdp_bubble',
    title: "GDP vs. Life Expectancy Global Bubble Analysis",
    description: "Models global health and wealth indicators. Renders Gapminder-style interactive bubble charts using Plotly.",
    category: 'EDA',
    notebookUrl: './notebooks/global_gdp_bubble.ipynb',
    tags: ["EDA", "Global Metrics", "Bubble Chart", "Plotly"],
    date: '2026-07-05',
    metrics: {
     
      "Countries": "150",
      "Indicators": "GDP, Life Exp",
      "Year Reference": "2025"

    }
  },
  {
    id: 'gmm_covariance',
    title: "GMM Expectation-Maximization Covariance Clustering",
    description: "Fits Gaussian Mixture Models (GMM) and plots covariance probability ellipses.",
    category: 'ML',
    notebookUrl: './notebooks/gmm_covariance.ipynb',
    tags: ["Unsupervised", "GMM", "Expectation-Maximization", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Components": "3",
      "Sample Size": "500",
      "Covariance Type": "Full"

    }
  },
  {
    id: 'gradient_boosting_importance',
    title: "Gradient Boosting Classifier Feature Importances",
    description: "Fits gradient boosting estimators to classify churn risk and ranks feature splits coordinates importances.",
    category: 'ML',
    notebookUrl: './notebooks/gradient_boosting_importance.ipynb',
    tags: ["Gradient Boosting", "Classification", "Feature Importance", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Trees Count": "100",
      "Max Depth": "5",
      "Accuracy": "86.9%"

    }
  },
  {
    id: 'gru_forecaster',
    title: "GRU Sequential Sensor Forecaster",
    description: "Builds a Gated Recurrent Unit (GRU) model in PyTorch to forecast sequential time-series sensor metrics.",
    category: 'AI',
    notebookUrl: './notebooks/gru_forecaster.ipynb',
    tags: ["PyTorch", "GRU", "Sequence Forecasting", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Sequence Length": "15",
      "Hidden Size": "32",
      "Test Loss (MSE)": "0.232"

    }
  },
  {
    id: 'hierarchical_dendrogram',
    title: "Hierarchical Agglomerative Clustering Dendrogram",
    description: "Fits hierarchical agglomerative clustering on customer features and visualizes the tree-like dendrogram linkage boundaries.",
    category: 'ML',
    notebookUrl: './notebooks/hierarchical_dendrogram.ipynb',
    tags: ["Clustering", "Hierarchical Linkage", "Dendrogram", "Scipy"],
    date: '2026-07-08',
    metrics: {
     
      "Sample Size": "250",
      "Linkage Metric": "Ward",
      "Optimal Clusters": "4"

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
    id: 'label_propagation',
    title: "Semi-Supervised Label Propagation Classifier",
    description: "Spreads classification labels from a small set of annotated points to unlabeled nodes using similarity graphs.",
    category: 'ML',
    notebookUrl: './notebooks/label_propagation.ipynb',
    tags: ["Semi-Supervised", "Label Propagation", "Graph Similarity", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Labeled Points Count": "25",
      "Unlabeled Points Count": "475",
      "Label Accuracy": "98.3%"

    }
  },
  {
    id: 'lsh_neighbors',
    title: "Locality Sensitive Hashing Nearest Neighbor Search",
    description: "Performs high-dimensional document search comparisons using LSH coordinates hashing signatures.",
    category: 'ML',
    notebookUrl: './notebooks/lsh_neighbors.ipynb',
    tags: ["Nearest Neighbor", "LSH", "Search Analytics", "Scikit-Learn"],
    date: '2026-07-08',
    metrics: {
     
      "Database Size": "2000 Docs",
      "Query Hash Size": "8 bits",
      "Search Speedup": "12x"

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
    id: 'lstm_seq2seq_autoencoder',
    title: "LSTM Sequence-to-Sequence Autoencoder",
    description: "Builds an LSTM-based sequence-to-sequence autoencoder in PyTorch to reconstruct variable-length time-series curves.",
    category: 'AI',
    notebookUrl: './notebooks/lstm_seq2seq_autoencoder.ipynb',
    tags: ["PyTorch", "LSTM Autoencoder", "Sequence Modeling", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Encoder LSTM Layers": "1",
      "Latent Dimension": "16",
      "Reconstruction MSE": "0.047"

    }
  },
  {
    id: 'lstm_text_generation',
    title: "Character-Level LSTM Generative Text Pipeline",
    description: "Trains a character-level LSTM generator in PyTorch on a text corpus. Evaluates convergence cross-entropy decay curve logs.",
    category: 'AI',
    notebookUrl: './notebooks/lstm_text_generation.ipynb',
    tags: ["PyTorch", "LSTM", "Generative AI", "NLP", "Matplotlib"],
    date: '2026-07-06',
    metrics: {
     
      "Vocabulary Size": "28",
      "Hidden Units": "64",
      "Final Loss": "0.04"

    }
  },
  {
    id: 'market_basket_chord',
    title: "Retail Basket Association Matrix Chord Diagram",
    description: "Simulates cross-purchase associations and plots correlation relationships on a circular node layout in Plotly.",
    category: 'EDA',
    notebookUrl: './notebooks/market_basket_chord.ipynb',
    tags: ["Market Basket Analysis", "Chord Visual", "Relationships Matrix", "Plotly"],
    date: '2026-07-08',
    metrics: {
     
      "Products": "10",
      "Link Counts": "41",
      "Confidence Min": "0.20"

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
    id: 'mlp_credit_risk',
    title: "MLP Credit Risk Classifier Pipeline",
    description: "Builds a multi-layer feedforward network with Batch Normalization to predict credit risk.",
    category: 'AI',
    notebookUrl: './notebooks/mlp_credit_risk.ipynb',
    tags: ["PyTorch", "MLP", "Batch Normalization", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Layer Dimensions": "32-16-1",
      "Dropout Ratio": "0.2",
      "Accuracy": "71.9%"

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
    id: 'molecular_structure_3d',
    title: "Molecular Structure Valence Orbital Visualization",
    description: "Plots 3D structural coordinate spheres and bonds of organic molecules (e.g. methane, benzene) in interactive spaces.",
    category: 'EDA',
    notebookUrl: './notebooks/molecular_structure_3d.ipynb',
    tags: ["Cheminformatics", "3D Visualization", "Molecular Modeling", "Plotly"],
    date: '2026-07-08',
    metrics: {
     
      "Atoms": "12",
      "Bonds": "15",
      "Coordinate System": "3D Cart"

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
    id: 'multi_output_chain',
    title: "Multi-Output Chain Regression Predictions",
    description: "Trains chain regressors to predict correlated target dimensions concurrently (e.g. multiple air pollutant metrics).",
    category: 'ML',
    notebookUrl: './notebooks/multi_output_chain.ipynb',
    tags: ["Regression", "Multi-Output", "Chain Estimator", "Scikit-Learn"],
    date: '2026-07-08',
    metrics: {
     
      "Target Variables": "3",
      "Feature Vectors": "8",
      "Mean R\u00b2": "0.543"

    }
  },
  {
    id: 'naive_bayes_text',
    title: "Naive Bayes Text Likelihood Classification",
    description: "Trains a Multinomial Naive Bayes text classifier on synthetic vocabulary bag-of-words. Visualizes likelihood feature parameters.",
    category: 'ML',
    notebookUrl: './notebooks/naive_bayes_text.ipynb',
    tags: ["Scikit-Learn", "Naive Bayes", "Classification", "NLP", "Matplotlib"],
    date: '2026-07-06',
    metrics: {
     
      "Vocabulary Size": "100",
      "Accuracy": "100.0%",
      "Review Count": "400"

    }
  },
  {
    id: 'one_class_svm',
    title: "One-Class SVM Outlier Profiler",
    description: "Fits a One-Class Support Vector Machine boundary to identify coordinate outliers inside noisy telemetry sets.",
    category: 'ML',
    notebookUrl: './notebooks/one_class_svm.ipynb',
    tags: ["One-Class SVM", "Outlier Detection", "Security Analytics", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Inlier Sample Count": "400",
      "Detection Accuracy (F1)": "63.3%",
      "Outlier Index": "8.5%"

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
    id: 'pcr_pipeline',
    title: "Principal Component Regression Pipeline",
    description: "Builds a PCA reduction linear regression pipeline (PCR) in Scikit-Learn.",
    category: 'ML',
    notebookUrl: './notebooks/pcr_pipeline.ipynb',
    tags: ["Dimension Reduction", "Linear Pipeline", "PCR", "Scikit-Learn"],
    date: '2026-07-08',
    metrics: {
     
      "Raw Predictors": "25",
      "Principal Components": "5",
      "RMSE": "0.313"

    }
  },
  {
    id: 'pls_regression',
    title: "Partial Least Squares (PLS) Regression Predictions",
    description: "Decomposes collinear predictors coordinates into latent variables projections to forecast target outputs.",
    category: 'ML',
    notebookUrl: './notebooks/pls_regression.ipynb',
    tags: ["PLS", "Collinearity", "Regression", "Scikit-Learn"],
    date: '2026-07-08',
    metrics: {
     
      "Predictors": "30",
      "Latent Variables": "3",
      "Test R\u00b2": "0.978"

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
    id: 'q_cartpole',
    title: "Tabular Q-Learning Discretized CartPole Balancer",
    description: "Trains a tabular Q-learning agent on discretized 4D states to stabilize a cartpole simulator. Charts convergence rewards.",
    category: 'AI',
    notebookUrl: './notebooks/q_cartpole.ipynb',
    tags: ["Q-Learning", "Discretization", "Reinforcement Learning", "Matplotlib"],
    date: '2026-07-05',
    metrics: {
     
      "State Bins": "162",
      "Action Space": "2",
      "Convergence Ep": "200"

    }
  },
  {
    id: 'random_forest_oob',
    title: "Random Forest Out-of-Bag Error Convergence",
    description: "Visualizes OOB error rate curves decaying over boosting iteration step coordinates.",
    category: 'ML',
    notebookUrl: './notebooks/random_forest_oob.ipynb',
    tags: ["Ensemble", "Random Forest", "OOB Error", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Max Trees Count": "250",
      "Min Leaf Size": "4",
      "OOB Stability Ep": "120"

    }
  },
  {
    id: 'rbf_network_approximator',
    title: "Radial Basis Function Approximator Network",
    description: "Trains a custom PyTorch layer evaluating Gaussian radial basis coordinates to interpolate complex curves.",
    category: 'AI',
    notebookUrl: './notebooks/rbf_network_approximator.ipynb',
    tags: ["PyTorch", "RBF Layer", "Function Approximation", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Centroids count": "10",
      "Variance Metric": "Gamma",
      "Test R\u00b2": "0.985"

    }
  },
  {
    id: 'real_estate_choropleth',
    title: "Real Estate Price Index Geo-Choropleth Map",
    description: "Maps state-level real estate valuation variations using regional choropleth US boundary projections.",
    category: 'EDA',
    notebookUrl: './notebooks/real_estate_choropleth.ipynb',
    tags: ["Geospatial", "Choropleth Map", "Real Estate", "Plotly"],
    date: '2026-07-08',
    metrics: {
     
      "Regional Codes": "50",
      "Pricing Average": "$385k",
      "Year Growth": "+4.2%"

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
    id: 'rfm_3d_scatter',
    title: "Customer Recency, Frequency, Monetary (RFM) Segmentation 3D Scatter",
    description: "Groups consumers along RFM axes and plots them in a 3D coordinate space to highlight high-value users.",
    category: 'EDA',
    notebookUrl: './notebooks/rfm_3d_scatter.ipynb',
    tags: ["Marketing Analytics", "RFM", "3D Scatter", "Plotly"],
    date: '2026-07-08',
    metrics: {
     
      "Customer Count": "1500",
      "Dimensions": "3",
      "High-Value Ratio": "14.5%"

    }
  },
  {
    id: 'ridge_lasso_path',
    title: "Ridge vs. Lasso Coefficient Shrinkage Paths",
    description: "Plots Lasso (L1) zeroing paths against Ridge (L2) asymptotic decays across log regularizer alphas.",
    category: 'ML',
    notebookUrl: './notebooks/ridge_lasso_path.ipynb',
    tags: ["Regularization", "Ridge", "Lasso", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Features": "15",
      "Alpha Range": "10^-4 to 10^3",
      "L1 Sparsity Count": "6"

    }
  },
  {
    id: 'rnn_signal_classifier',
    title: "RNN Binary Sequence Signal Classifier",
    description: "Trains a recurrent neural network to classify sequence coordinates as normal vs. anomalous.",
    category: 'AI',
    notebookUrl: './notebooks/rnn_signal_classifier.ipynb',
    tags: ["PyTorch", "RNN", "Sequence Classification", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "State Vector": "2",
      "Sequence Steps": "20",
      "Loss (BCE)": "0.08"

    }
  },
  {
    id: 'server_streamgraph',
    title: "Server RAM & CPU Usage Streamgraph Tracker",
    description: "Tracks dynamic server cluster resource usage stacks over 24-hour time windows in streamgraph formats.",
    category: 'EDA',
    notebookUrl: './notebooks/server_streamgraph.ipynb',
    tags: ["SysOps", "Telemetry", "Streamgraph", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Server Nodes": "5",
      "Timestamps": "288",
      "Max Capacity": "64GB"

    }
  },
  {
    id: 'set_intersection_venn',
    title: "Venn Diagram Set-Intersection Analysis",
    description: "Models customer subscription overlaps using area-proportional sets boundaries.",
    category: 'EDA',
    notebookUrl: './notebooks/set_intersection_venn.ipynb',
    tags: ["Set Theory", "Venn Diagram", "Intersection Metrics", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Dimensions": "3 Sets",
      "Overlaps": "7 Areas",
      "Intersection %": "12.8%"

    }
  },
  {
    id: 'siamese_distance_learner',
    title: "Siamese Metric Learner Contrastive Network",
    description: "Trains Siamese twin networks using Contrastive Loss to estimate feature distances between similar/dissimilar samples in PyTorch.",
    category: 'AI',
    notebookUrl: './notebooks/siamese_distance_learner.ipynb',
    tags: ["PyTorch", "Siamese Network", "Contrastive Loss", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Input Dimensions": "16",
      "Latent Dimensions": "4",
      "Accuracy": "94.2%"

    }
  },
  {
    id: 'silhouette_analysis',
    title: "K-Means Clustering & Silhouette Analysis",
    description: "Applies K-Means clustering to synthetic blob coordinates. Evaluates optimal cluster counts using Silhouette metrics.",
    category: 'ML',
    notebookUrl: './notebooks/silhouette_analysis.ipynb',
    tags: ["Scikit-Learn", "K-Means", "Clustering", "Silhouette Score", "Matplotlib"],
    date: '2026-07-06',
    metrics: {
     
      "Target Clusters": "3",
      "Silhouette Score": "0.793",
      "Bins Count": "2-8"

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
    id: 'spectral_clustering',
    title: "Spectral Clustering Graph Partitioning Bounds",
    description: "Applies Spectral Clustering to partition complex concentric geometric structures. Compares K-Means vs Spectral results in Matplotlib.",
    category: 'ML',
    notebookUrl: './notebooks/spectral_clustering.ipynb',
    tags: ["Scikit-Learn", "Spectral Clustering", "Unsupervised", "Matplotlib"],
    date: '2026-07-05',
    metrics: {
     
      "Target Clusters": "2",
      "Sample Size": "400",
      "Gamma Parameter": "15.0"

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
    id: 'student_boxplots',
    title: "Multi-Variable Boxplot Score Analysis",
    description: "Compares distribution statistics (quartiles, outliers) of performance metrics across multiple student categories.",
    category: 'EDA',
    notebookUrl: './notebooks/student_boxplots.ipynb',
    tags: ["Statistical Visuals", "Boxplot", "Outliers", "Seaborn"],
    date: '2026-07-08',
    metrics: {
     
      "Test Groups": "5",
      "Student Count": "1200",
      "Outlier Ratio": "1.75%"

    }
  },
  {
    id: 'style_transfer_optimizer',
    title: "Neural Style Transfer Matrix Optimizer",
    description: "Implements feature representations loss optimization between a content array, a style array, and a target array.",
    category: 'AI',
    notebookUrl: './notebooks/style_transfer_optimizer.ipynb',
    tags: ["Generative AI", "Optimization Loop", "Neural Style Transfer", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Target Matrix Size": "32x32",
      "Loss Weight Style": "10^3",
      "Epochs Count": "60"

    }
  },
  {
    id: 'supply_chain_sankey',
    title: "Supply Chain Logistics Network Flow Diagram",
    description: "Generates flow transition Sankey diagrams mapping supply chains from supplier plants to warehouse distribution nodes.",
    category: 'EDA',
    notebookUrl: './notebooks/supply_chain_sankey.ipynb',
    tags: ["Network Graph", "Logistics", "Sankey Diagram", "Plotly"],
    date: '2026-07-08',
    metrics: {
     
      "Supply Nodes": "8",
      "Total Capacity": "25k Units",
      "Efficiency": "94.2%"

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
    id: 'svm_margin_search',
    title: "Support Vector Machine (SVM) Hyperparameter Margin Search",
    description: "Trains SVM models with varying RBF gamma parameters, mapping margin decisions boundaries and support vectors.",
    category: 'ML',
    notebookUrl: './notebooks/svm_margin_search.ipynb',
    tags: ["SVM", "Kernel Trick", "Hyperparameter Search", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Gamma Parameter": "0.1, 1.0, 10.0",
      "Support Vector Ratio": "18.5%",
      "Accuracy": "91.2%"

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
    id: 'temperature_hexbins',
    title: "Climate Temperature Anomaly Hexbin Density Map",
    description: "Aggregates coordinates measurements of global weather deviations using hexbin density grids.",
    category: 'EDA',
    notebookUrl: './notebooks/temperature_hexbins.ipynb',
    tags: ["Climatology", "Density Map", "Hexbin", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Coordinates Points": "5000",
      "Bins": "30",
      "Anomaly Limits": "-3.5C to +4.0C"

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
    id: 'traffic_violin_speed',
    title: "Vehicle Traffic Speed Violin Map Analysis",
    description: "Evaluates highway traffic velocity distributions across days of the week using Seaborn Violin plots.",
    category: 'EDA',
    notebookUrl: './notebooks/traffic_violin_speed.ipynb',
    tags: ["Transportation", "Violin Plot", "Distribution Analytics", "Seaborn"],
    date: '2026-07-08',
    metrics: {
     
      "Sensor Locations": "4",
      "Passes Checked": "8000",
      "Speed Limit": "65mph"

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
    id: 'transformer_char_classifier',
    title: "Transformer Encoder Character sequence Classifier",
    description: "Builds a transformer encoder model featuring multi-head self-attention to classify character sequences.",
    category: 'AI',
    notebookUrl: './notebooks/transformer_char_classifier.ipynb',
    tags: ["Transformer", "Attention Mechanism", "NLP", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Attention Heads": "4",
      "Encoder Layers": "2",
      "Accuracy": "98.3%"

    }
  },
  {
    id: 'tsne_projection',
    title: "t-SNE Dimensionality Reduction Projections",
    description: "Projects high-dimensional variables down to a 2D space using t-SNE in Scikit-Learn.",
    category: 'ML',
    notebookUrl: './notebooks/tsne_projection.ipynb',
    tags: ["Dimensionality Reduction", "t-SNE", "Manifold Learning", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Input Features": "10",
      "Reduced Features": "2",
      "Perplexity Parameter": "30"

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
  },
  {
    id: 'vocabulary_wordcloud',
    title: "Text Vocabulary Frequency WordCloud Analysis",
    description: "Compiles text datasets and renders area-scaled word occurrences clouds to highlight text topic profiles.",
    category: 'EDA',
    notebookUrl: './notebooks/vocabulary_wordcloud.ipynb',
    tags: ["NLP", "WordCloud", "Exploratory Text Analysis", "Matplotlib"],
    date: '2026-07-08',
    metrics: {
     
      "Doc Count": "250",
      "Vocabulary Bins": "500",
      "Top Word Count": "140"

    }
  },
  {
    id: 'web_traffic_polar',
    title: "Web Traffic Hourly Activity Radial Polar Chart",
    description: "Maps web traffic request volumes across 24-hour cycles onto polar radar axes to pinpoint server peaks.",
    category: 'EDA',
    notebookUrl: './notebooks/web_traffic_polar.ipynb',
    tags: ["Web Telemetry", "Polar Chart", "Chronological Analytics", "Plotly"],
    date: '2026-07-08',
    metrics: {
     
      "Sample Size": "24h",
      "Server Load Index": "0-100",
      "Peak Hours": "19:00-22:00"

    }
  }
];
