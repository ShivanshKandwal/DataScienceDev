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
  }
];
