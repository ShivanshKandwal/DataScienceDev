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
