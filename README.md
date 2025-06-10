# 🫀 Heart Failure Prediction System

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

*A machine learning approach to predict heart failure outcomes using clinical features*

[📊 Demo](#demo) • [🚀 Quick Start](#quick-start) • [📖 Documentation](#documentation) • [🤝 Contributing](#contributing)

</div>

---

## 🎯 Overview

This project implements a comprehensive machine learning pipeline to predict heart failure outcomes using clinical patient data. The system compares multiple classification algorithms and selects the best performing model based on recall score - prioritizing the identification of patients at risk.

### ✨ Key Features

- 🔬 **Multi-Algorithm Comparison**: Random Forest, Logistic Regression, KNN, SVM, and Decision Tree
- 📊 **Comprehensive Analysis**: Correlation analysis, feature visualization, and model evaluation
- 🎯 **Medical-Focused Metrics**: Emphasis on recall to minimize false negatives
- 💾 **Model Persistence**: Automated saving of the best performing model
- 📈 **Interactive Visualizations**: Correlation heatmaps and pair plotting

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/heart-failure-prediction.git
   cd heart-failure-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset

The model uses the Heart Failure Clinical Dataset containing 13 clinical features:

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Patient age | Numerical |
| `anaemia` | Haemoglobin level decrease | Boolean |
| `creatinine_phosphokinase` | CPK enzyme level | Numerical |
| `diabetes` | Diabetes presence | Boolean |
| `ejection_fraction` | Heart pumping efficiency | Numerical |
| `high_blood_pressure` | Hypertension | Boolean |
| `platelets` | Platelet count | Numerical |
| `serum_creatinine` | Creatinine level | Numerical |
| `serum_sodium` | Sodium level | Numerical |
| `sex` | Patient gender | Boolean |
| `smoking` | Smoking status | Boolean |
| `time` | Follow-up period | Numerical |
| `DEATH_EVENT` | Target variable | Boolean |

## 🚀 Quick Start

### Basic Usage

```python
import pandas as pd
from joblib import load

# Load the trained model
model_package = load('heartFailure_model.pkl')
model = model_package['model']
scaler = model_package['scaler']

# Make predictions
new_data = pd.DataFrame({
    'age': [65],
    'anaemia': [0],
    'creatinine_phosphokinase': [582],
    # ... include all features
})

# Scale and predict
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
probability = model.predict_proba(new_data_scaled)

print(f"Prediction: {prediction[0]}")
print(f"Risk Probability: {probability[0][1]:.2%}")
```

### Training Pipeline

```bash
# Run the complete training pipeline
python heart_failure_prediction.py
```

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 0.8500 | 0.7273 | 0.8000 | 0.7619 |
| Logistic Regression | 0.8167 | 0.6667 | 0.8000 | 0.7273 |
| K-Nearest Neighbors | 0.7833 | 0.6316 | 0.7200 | 0.6724 |
| Support Vector Machine | 0.8167 | 0.6842 | 0.7200 | 0.7018 |
| Decision Tree | 0.7833 | 0.6316 | 0.7200 | 0.6724 |

> 🎯 **Best Model**: Random Forest (selected based on highest recall score)

### Cross-Validation Results

- **Random Forest**: 0.8276 (±0.0891)
- **Logistic Regression**: 0.8138 (±0.0757)
- **KNN**: 0.7517 (±0.0816)
- **SVM**: 0.7931 (±0.0932)
- **Decision Tree**: 0.7379 (±0.1047)

## 📁 Project Structure

```
heart-failure-prediction/
├── 📊 data/
│   └── HeartFailureData.csv
├── 📓 notebooks/
│   └── heart_failure_analysis.ipynb
├── 🧠 models/
│   └── heartFailure_model.pkl
├── 📈 visualizations/
│   ├── correlation_matrix.png
│   └── feature_distribution.png
├── 📜 src/
│   └── heart_failure_prediction.py
├── 📋 requirements.txt
├── 🗒️ README.md
└── 📄 LICENSE
```

## 🔧 Key Features

### Data Analysis
- **Exploratory Data Analysis**: Comprehensive statistical analysis
- **Correlation Matrix**: Visual representation of feature relationships
- **Pair Plotting**: Distribution analysis by target variable

### Model Training
- **Multiple Algorithms**: Systematic comparison of 5 different models
- **Cross-Validation**: 5-fold stratified cross-validation
- **Hyperparameter Tuning**: Optimized model parameters
- **Medical Focus**: Recall-optimized for healthcare applications

### Evaluation Metrics
- **Accuracy**: Overall prediction correctness
- **Precision**: Positive prediction reliability
- **Recall**: Sensitivity to positive cases (prioritized)
- **F1-Score**: Balanced precision-recall metric

## 🔍 Usage Examples

### Data Exploration
```python
# Load and explore the dataset
df = pd.read_csv('data/HeartFailureData.csv')
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['DEATH_EVENT'].value_counts()}")
```

### Visualization
```python
# Generate correlation heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(16, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True)
plt.title('Feature Correlation Matrix')
plt.show()
```

## 🤝 Contributing

I welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset sourced from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+record)
- Built with [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), and [matplotlib](https://matplotlib.org/)
- Inspired by the need for better healthcare predictive analytics

## 📞 Contact

- **Author**: Bilal Rukundi
- **Email**: bilalrukundi1658@gmail.com
- **LinkedIn**: [LinkedIn Profile](www.linkedin.com/in/bilal-rukundi)
- **Project Link**: [https://github.com/Wayn-git/heart-failure-prediction](https://github.com/Wayn-git/heart-failure-prediction)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ for better healthcare outcomes

</div>
