# task5-decision-trees-randomforests
# Task 5 – Classification using Decision Tree and Random Forest

## Objective

The objective of this task is to implement and compare **Decision Tree** and **Random Forest** classification models on a healthcare dataset (Heart Disease). The goal is to evaluate their accuracy, understand overfitting, and analyze feature importance using ensemble methods.

## Dataset

- Name: Heart Disease Dataset
- Source: [Kaggle – Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- File Used: `heart.csv`
- Target Column: `target` (0 = No Heart Disease, 1 = Heart Disease Present)

This dataset includes features such as:
- `age`, `sex`, `cp` (chest pain type)
- `trestbps`, `chol` (blood pressure, cholesterol)
- `thalach`, `exang`, and other clinical attributes

## Tools and Libraries Used

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn (sklearn)


## Workflow Summary

### 1. Data Loading & Preprocessing
- The dataset was loaded using `pandas`.
- Categorical columns were encoded using one-hot encoding if needed.
- Feature matrix `X` and target `y` were defined.
- Data was split into 80% training and 20% testing sets.

### 2. Decision Tree Classifier
- Trained a `DecisionTreeClassifier` on the training data.
- Predicted outcomes on the test data.
- Evaluated using:
  - Accuracy
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1 Score)
- Visualized the decision tree using `plot_tree()` from `sklearn`.

### 3. Overfitting Analysis
- Trained a second decision tree with `max_depth=3` to reduce overfitting.
- Compared test accuracy with the default tree.

### 4. Random Forest Classifier
- Trained a `RandomForestClassifier` with 100 trees.
- Compared accuracy and classification report with the decision tree.
- Extracted **feature importances** to determine which clinical attributes were most influential in predicting heart disease.

### 5. Cross-Validation
- Performed 5-fold cross-validation for both classifiers.
- Reported average cross-validated accuracy for both models.

## Visualizations

- Confusion matrix heatmap (Decision Tree & Random Forest)
- Decision tree visualization
- Feature importance bar chart (Random Forest)
- Optional: ROC-AUC curve

