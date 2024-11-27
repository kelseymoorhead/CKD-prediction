#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:01:10 2024

@author: kelseymoorhead
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle

# Step 1: Load and preprocess the dataset
file_path = '/Users/kelseymoorhead/Documents/ckd_normalized.csv'
df = pd.read_csv(file_path)

# Convert categorical columns to numerical
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Separate features and target variable
X = df.drop('Class', axis=1)
y = df['Class']

# Save the feature names
features_path = 'features.pkl'
with open(features_path, 'wb') as file:
    pickle.dump(X.columns.tolist(), file)
print(f"Features saved as '{features_path}'")

# Add noise to simulate real-world variability
np.random.seed(42)
X += np.random.normal(0, 0.05, X.shape)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training, validation, and test sets (40/30/30 split)
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.6, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalize numerical features
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_val = pd.DataFrame(scaler.transform(X_val), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Save the fitted scaler
scaler_path = 'scaler.pkl'
with open(scaler_path, 'wb') as file:
    pickle.dump(scaler, file)
print(f"Scaler saved as '{scaler_path}'")

# Step 2: Optimize the model with Grid Search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

# Extract the best model
best_model = grid_search.best_estimator_

# Save the model
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
print("Model saved as 'best_model.pkl'")

# Step 3: Adjust Prediction Threshold for Improved Recall
def adjusted_predictions(model, X, threshold=0.4):
    probabilities = model.predict_proba(X)[:, 1]  # Probability of being CKD
    return (probabilities >= threshold).astype(int)

# Step 4: Validate and test the model
y_val_pred = adjusted_predictions(best_model, X_val, threshold=0.4)
y_test_pred = adjusted_predictions(best_model, X_test, threshold=0.4)

# Calculate metrics
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_pred)

print("\nTest Set Metrics with Adjusted Threshold:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix on Test Set with Adjusted Threshold:")
print(conf_matrix)

# Step 5: Plot feature importances
feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
feature_importances_sorted = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importances_sorted.plot(kind='bar', color='skyblue')
plt.title('Feature Importances from Random Forest Model')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.show()

# Step 6: Simulate real-time data
def simulate_real_time_data(model, X_test, y_test, threshold=0.4):
    print("\nReal-Time Data Simulation:")
    total_samples = len(X_test)
    correct_predictions = 0

    start_time = time.time()
    for i in range(total_samples):
        X_sample = X_test.iloc[[i]]
        y_sample = y_test.iloc[i]

        y_pred = adjusted_predictions(model, X_sample, threshold=threshold)[0]
        if y_pred == y_sample:
            correct_predictions += 1

        print(f"Sample {i+1}/{total_samples}: True={y_sample}, Predicted={y_pred}")

    end_time = time.time()
    accuracy = correct_predictions / total_samples
    print(f"\nSimulation completed. Accuracy: {accuracy:.2f}")
    print(f"Processing time: {end_time - start_time:.2f} seconds for {total_samples} samples.")

simulate_real_time_data(best_model, X_test, y_test, threshold=0.4)

# Step 7: Cross-validation metrics
cv_scores = cross_val_score(best_model, X_resampled, y_resampled, cv=5, scoring='f1')
print("\nCross-Validation F1 Scores:", cv_scores)
print("Mean F1 Score:", cv_scores.mean())




