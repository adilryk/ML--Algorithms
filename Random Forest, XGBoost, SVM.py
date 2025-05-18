# Adil Sahfique
#SP23-BAI-003

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from scipy.stats import randint

df = pd.read_csv('Telco-Customer-Churn.csv')

df = pd.get_dummies(df, drop_first=True)

X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)





# Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Grid Search Optimization
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train_resampled, y_train_resampled)

# Random Search Optimization
param_dist_rf = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}
random_search_rf = RandomizedSearchCV(estimator=rf, param_distributions=param_dist_rf, n_iter=50, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
random_search_rf.fit(X_train_resampled, y_train_resampled)





# XGBoost Classifier
xgboost = GradientBoostingClassifier(random_state=42)

# Grid Search Optimization
param_grid_xgboost = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 150, 200],
    'max_depth': [6, 8, 10],
    'subsample': [0.8, 0.9, 1.0]
}
grid_search_xgboost = GridSearchCV(estimator=xgboost, param_grid=param_grid_xgboost, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_xgboost.fit(X_train_resampled, y_train_resampled)

# Random Search Optimization
param_dist_xgboost = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': randint(100, 250),
    'max_depth': randint(5, 15),
    'subsample': [0.7, 0.8, 0.9],
}
random_search_xgboost = RandomizedSearchCV(estimator=xgboost, param_distributions=param_dist_xgboost, n_iter=50, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
random_search_xgboost.fit(X_train_resampled, y_train_resampled)





# Support Vector Machine (SVM)
svm = SVC(random_state=42)

# Grid Search Optimization
param_grid_svm = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1]
}
grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_svm.fit(X_train_resampled, y_train_resampled)

# Random Search Optimization
param_dist_svm = {
    'kernel': ['linear', 'rbf'],
    'C': randint(0.1, 10),
    'gamma': [0.01, 0.1, 1]
}
random_search_svm = RandomizedSearchCV(estimator=svm, param_distributions=param_dist_svm, n_iter=50, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
random_search_svm.fit(X_train_resampled, y_train_resampled)






# Evaluate models
rf_best = grid_search_rf.best_estimator_
rf_random_best = random_search_rf.best_estimator_

xgboost_best = grid_search_xgboost.best_estimator_
xgboost_random_best = random_search_xgboost.best_estimator_

svm_best = grid_search_svm.best_estimator_
svm_random_best = random_search_svm.best_estimator_





# Predictions
y_pred_rf = rf_best.predict(X_test_scaled)
y_pred_rf_random = rf_random_best.predict(X_test_scaled)

y_pred_xgboost = xgboost_best.predict(X_test_scaled)
y_pred_xgboost_random = xgboost_random_best.predict(X_test_scaled)

y_pred_svm = svm_best.predict(X_test_scaled)
y_pred_svm_random = svm_random_best.predict(X_test_scaled)



# Evaluation Metrics
print("Random Forest (Grid Search):")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))




# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix (Grid Search)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nRandom Forest (Random Search):")
print(classification_report(y_test, y_pred_rf_random))
print(confusion_matrix(y_test, y_pred_rf_random))





# Random Forest (Random Search) Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_rf_random), annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix (Random Search)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nXGBoost (Grid Search):")
print(classification_report(y_test, y_pred_xgboost))
print(confusion_matrix(y_test, y_pred_xgboost))





# XGBoost Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_xgboost), annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost Confusion Matrix (Grid Search)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nXGBoost (Random Search):")
print(classification_report(y_test, y_pred_xgboost_random))
print(confusion_matrix(y_test, y_pred_xgboost_random))

# XGBoost (Random Search) Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_xgboost_random), annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost Confusion Matrix (Random Search)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nSVM (Grid Search):")
print(classification_report(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))

# SVM Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix (Grid Search)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nSVM (Random Search):")
print(classification_report(y_test, y_pred_svm_random))
print(confusion_matrix(y_test, y_pred_svm_random))

# SVM (Random Search) Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_svm_random), annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix (Random Search)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
