# Data Preprocessing Tools

A comprehensive guide to essential data preprocessing techniques in machine learning using Python and scikit-learn. This repository contains a Jupyter notebook that demonstrates handling missing data, encoding categorical variables, splitting datasets, and feature scaling.

## Overview

This project covers the fundamental preprocessing steps required before training machine learning models. It walks through practical examples using a sample dataset with missing values and categorical features.

## Features

- **Missing Data Handling**: Imputation using mean strategy with `SimpleImputer`
- **Categorical Encoding**: 
  - One-Hot Encoding for independent variables using `OneHotEncoder`
  - Label Encoding for dependent variables using `LabelEncoder`
- **Train-Test Split**: Proper dataset splitting for model evaluation
- **Feature Scaling**: Standardization using `StandardScaler` for numerical features

## Dataset

The repository includes a sample dataset (`Data.csv`) with:
- **Features**: Country, Age, Salary
- **Target**: Purchased (Yes/No)
- **Characteristics**: Contains missing values for demonstration purposes

| Country | Age | Salary | Purchased |
|---------|-----|--------|-----------|
| France  | 44  | 72000  | No        |
| Spain   | 27  | 48000  | Yes       |
| Germany | 40  | NaN    | Yes       |
| Spain   | NaN | 52000  | No        |

## Technologies Used

- **Python 3.x**
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Scikit-learn**: Machine learning preprocessing tools
  - `SimpleImputer`
  - `OneHotEncoder`
  - `LabelEncoder`
  - `ColumnTransformer`
  - `StandardScaler`
  - `train_test_split`

## Installation

```bash
# Clone the repository
git clone https://github.com/lakumsaicharan/data-preprocessing-tools.git

# Navigate to the project directory
cd data-preprocessing-tools

# Install required packages
pip install numpy pandas matplotlib scikit-learn jupyter
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook "data preprocessing tools"
```

2. Run each cell sequentially to see the preprocessing steps in action

## Preprocessing Steps Demonstrated

### 1. Missing Data Imputation
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
```

### 2. Encoding Categorical Data
```python
# One-Hot Encoding for independent variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)

# Label Encoding for dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
```

### 3. Train-Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
```

### 4. Feature Scaling
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
```

## Learning Outcomes

After working through this notebook, you'll understand:
- Why preprocessing is crucial for machine learning models
- How to handle missing data appropriately
- When to use One-Hot Encoding vs Label Encoding
- The importance of feature scaling
- How to properly split data to avoid data leakage
- Best practices for preprocessing pipelines

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new preprocessing techniques
- Improve documentation
- Add more examples

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Sai Charan Lakum**
- GitHub: [@lakumsaicharan](https://github.com/lakumsaicharan)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/lakumsaicharan)

## Acknowledgments

- Scikit-learn documentation for comprehensive preprocessing guides
- Machine learning community for best practices

---

⭐ Star this repository if you find it helpful!
