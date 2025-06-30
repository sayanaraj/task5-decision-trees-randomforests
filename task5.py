import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('heart.csv')

if df.select_dtypes(include='object').shape[1] > 0:
    df = pd.get_dummies(df, drop_first=True)

X = df.drop('target', axis=1)
y = df['target']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

y_pred_dt = dtree.predict(X_test)
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))
print("Accuracy:", accuracy_score(y_test, y_pred_dt))

sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Blues')
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(dtree, feature_names=X.columns, class_names=['0', '1'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

dtree_limited = DecisionTreeClassifier(max_depth=3, random_state=42)
dtree_limited.fit(X_train, y_train)
y_pred_dtl = dtree_limited.predict(X_test)
print("Decision Tree (Depth=3) Accuracy:", accuracy_score(y_test, y_pred_dtl))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title("Random Forest Feature Importance")
plt.show()

cv_dt = cross_val_score(dtree, X, y, cv=5)
cv_rf = cross_val_score(rf, X, y, cv=5)

print("Cross-Validation Accuracy (Decision Tree):", cv_dt.mean())
print("Cross-Validation Accuracy (Random Forest):", cv_rf.mean())
