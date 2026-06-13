# ============================================================
# Diabetes Prediction using Support Vector Machine (SVM)
# Author: Kolli Upendra | PVPSIT | CSE (AI & ML) | 2024-2028
# Dataset: Pima Indians Diabetes Dataset
# ============================================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load Dataset
df = pd.read_csv('diabetes.csv')
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Shape:", df.shape)

# Step 3: Data Preprocessing
# Feature Selection
X = df.iloc[:, :-1]   # All columns except last (features)
y = df.iloc[:, -1]    # Last column (target: 0=No Diabetes, 1=Diabetes)

print("\nFeatures shape:", X.shape)
print("Target distribution:\n", y.value_counts())

# Step 4: Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Step 5: Build SVM Model (Linear Kernel)
model = SVC(kernel='linear', random_state=0)
model.fit(X_train, y_train)
print("\nModel trained successfully!")

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluation Metrics
print("\n--- Evaluation Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Diabetes Prediction (SVM Linear Kernel)')
plt.show()
