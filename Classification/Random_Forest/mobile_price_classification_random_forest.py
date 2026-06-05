#THE CODE IS EXECUTED BY :- SANKET GAIKWAD
# ALGORITHM :- RANDOM FOREST CLASSIFIER
# SOURCE :- Mobile Price Classification(https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load train and test
x = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\PYTHON\\train.csv")
y = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\PYTHON\\test.csv")

x['is_train'] = 1
y['is_train'] = 0
y['price_range'] = np.nan  # ensure consistent column

# Combine both the dataset for consistent preprocessing
a = pd.concat([x, y], axis=0, ignore_index=True)
print(a.columns.tolist())

# Drop unnecessary columns
a = a.drop(['id'], axis=1)

# Split back to train and test
train_data = a[a['is_train'] == 1].drop(['is_train'], axis=1)
test_data = a[a['is_train'] == 0].drop(['is_train', 'price_range'], axis=1)

# Set target and features
X = train_data.drop('price_range', axis=1)
y = train_data['price_range'].astype(int)  # Ensure it's categorical for classifier

# Scaling
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# PCA (optional)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=1)

# Train model
model = RandomForestClassifier(n_estimators=150, criterion='entropy', class_weight='balanced', max_depth=4)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print(y_pred)

# Evaluation
print("\nEVALUATION")
print("ACCURACY SCORE:\n", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX:\n", confusion_matrix(y_test, y_pred))
print("CLASSIFICATION REPORT:\n", classification_report(y_test, y_pred))

# Cross-validation
print("\nCROSS VALIDATION")
scores = cross_val_score(model, X_pca, y, cv=10)
print("Scores:", scores)
print("Mean Score:", np.mean(scores))


#Correlation Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(pd.DataFrame(X).corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


#Explained Variance by PCA Components
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.tight_layout()
plt.show()


# Confusion Matrix (Heatmap)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()


