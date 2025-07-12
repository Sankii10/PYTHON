#THE FOLLOWING CODE IS EXECUTED BY MR.SANKET GAIKWAD
# ALGORITHM - KNN CLASSIFIER
# SOURCE :- MUSHROOM CLASSIFICATION (https://www.kaggle.com/datasets/uciml/mushroom-classification)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\PYTHON\\mushrooms.csv")
print(a.columns.tolist())

# Veil-type count and drop
print("\nVeil-type Count:")
print(a['veil-type'].value_counts())
a = a.drop('veil-type', axis=1)

# Check data info and missing values
print("\nDataset Info:")
print(a.info())
print("\nMissing Values Check:")
print(a.isnull().sum())

missing_value = a[a.isna().any(axis=1)]
if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")

# Label Encoding
le = LabelEncoder()
for col in a:
    a[col] = le.fit_transform(a[col])

# Class Distribution Before SMOTE
plt.figure(figsize=(6,4))
sns.countplot(x='class', data=a)
plt.title("Class Distribution Before SMOTE")
plt.xlabel("Class (0 = Edible, 1 = Poisonous)")
plt.ylabel("Count")
plt.show()

# Feature selection
x = a.drop('class', axis=1)
y = a['class']

# Standardization
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

# PCA with fixed components
pca = PCA(n_components=10)
x_pca = pca.fit_transform(x_scaled)

# Explained Variance by PCA
plt.figure(figsize=(8,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title("Cumulative Explained Variance by PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.show()

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=1)

# SMOTE on training set
smote = SMOTE(random_state=1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

# Class Distribution After SMOTE
plt.figure(figsize=(6,4))
sns.countplot(x=y_train_res)
plt.title("Class Distribution After SMOTE (Training Data)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# KNN Model
model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train_res, y_train_res)

# Predictions and evaluation
y_pred = model.predict(x_test)
print("\nEVALUATION")
print("ACCURACY SCORE:\n", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX:\n", confusion_matrix(y_test, y_pred))
print("CLASSIFICATION REPORT:\n", classification_report(y_test, y_pred))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Edible', 'Poisonous'], yticklabels=['Edible', 'Poisonous'])
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Actual vs Predict Comparison Plot
comparison = pd.DataFrame({'Actual': y_test, 'Predict': y_pred})
comparison.head(30).plot(kind='bar', figsize=(12,6))
plt.title("Actual vs Predicted (First 30 Observations)")
plt.xlabel("Index")
plt.ylabel("Class")
plt.xticks(rotation=0)
plt.legend(["Actual", "Predicted"])
plt.tight_layout()
plt.show()

# Cross-validation with StratifiedKFold
print("CROSS SCORE VALIDATION")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(model, x_pca, y, cv=skf)
print("Scores per fold:", scores)
print("Mean CV Score:", np.mean(scores))

# K value vs Accuracy Plot (Elbow Method)
cv_scores = []
k_range = range(1, 20, 2)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores_k = cross_val_score(knn, x_pca, y, cv=skf)
    cv_scores.append(np.mean(scores_k))

plt.figure(figsize=(8,5))
plt.plot(k_range, cv_scores, marker='o')
plt.title("K Value vs. Cross-Validation Accuracy")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("CV Accuracy")
plt.grid(True)
plt.show()

# OPTIONAL: 2D PCA Plot Colored by Class
pca_2 = PCA(n_components=2)
x_pca_2 = pca_2.fit_transform(x_scaled)
df_pca = pd.DataFrame(x_pca_2, columns=['PC1', 'PC2'])
df_pca['class'] = y.values

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='class', palette='Set1', alpha=0.7)
plt.title("2D PCA Scatter Plot by Class")
plt.show()
