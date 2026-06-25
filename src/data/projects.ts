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
    id: 'titanic-eda',
    title: 'Titanic Survival Exploratory Data Analysis',
    description: 'Data-mining passenger manifest records to extract survival patterns. Features extensive categorical data engineering, missing value imputations, and seaborn density plots.',
    category: 'EDA',
    notebookUrl: './notebooks/titanic_eda.ipynb',
    tags: ['Python', 'Pandas', 'Seaborn', 'Matplotlib'],
    date: '2026-06-21',
    metrics: {
      "Records": "891",
      "Features": "12",
      "Correlations": "0.78"
    }
  },
  {
    id: 'house-regression',
    title: 'Housing Market Prices Regression model',
    description: 'Predictive pricing utilizing Random Forest and XGBoost Regressor. Implements cross-validated hyperparameter tuning and recursive feature elimination.',
    category: 'ML',
    notebookUrl: './notebooks/housing_regression.ipynb',
    dashboardUrl: 'https://share.streamlit.io/house-regression-model',
    dashboardImage: 'https://images.unsplash.com/photo-1560518883-ce09059eeffa?auto=format&fit=crop&w=800&q=80',
    tags: ['XGBoost', 'Scikit-Learn', 'Random Forest', 'Plotly'],
    date: '2026-06-23',
    metrics: {
      "R² Score": "0.915",
      "CV Folds": "10",
      "RMSE": "$14.2k"
    }
  },
  {
    id: 'brain-segmentation',
    title: 'Brain Tumor MRI Segmentation U-Net',
    description: 'Pixel-level segmentation of MRI scans using a custom PyTorch U-Net architecture. Employs dice loss coefficient optimizations and data augmentation matrices.',
    category: 'AI',
    notebookUrl: './notebooks/brain_tumor_segment.ipynb',
    dashboardImage: 'https://images.unsplash.com/photo-1559757175-5700dde675bc?auto=format&fit=crop&w=800&q=80',
    tags: ['PyTorch', 'U-Net', 'CNN', 'Deep Learning'],
    date: '2026-06-25',
    metrics: {
      "IoU Acc": "96.8%",
      "Dice Coeff": "0.942",
      "Epochs": "120"
    }
  }
];
