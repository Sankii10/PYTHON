# THE FOLLOWING CODE IS EXECUTED BY MR.SANKET GAIKWAD
# SOURCE :- INCENERATOR REAL ESTATE DATA
# ALGORITHM :- SVC(GRIDSEARCH)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\INTERNSHIP\\real_estate.csv")
print(a.columns.tolist())

# Use histplot instead of distplot (Deprecated)
print()
sns.histplot(a['price'], kde=True)
plt.title("DISTRIBUTION OF PRICE")
plt.show()

print()
sns.histplot(a['lprice'], kde=True)
plt.title("DISTRIBUTION OF LOG PRICE")
plt.show()

# Drop irrelevant or redundant columns
a = a.drop(['year', 'price', 'rprice', 'lrprice ', 'agesq', 'ldist', 'larea', 'lland',
            'y81nrinc', 'y81ldist', 'lintstsq', 'lintst'], axis=1)

# Feature Engineering
# Create interaction terms
a['area_baths'] = a['area'] * a['baths']
a['rooms_baths'] = a['rooms'] * a['baths']

# Log-transform skewed features
a['log_area'] = np.log1p(a['area'])
a['log_land'] = np.log1p(a['land'])

# Check target distribution
print()
print(a['lprice'].value_counts())

# Data overview
print()
print(a.info())

print()
print(a.isnull().sum())

missing_value = a[a.isna().any(axis=1)]
print()
if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")

# Split features and target
print()
x = a.drop('lprice', axis=1)
y = a['lprice']

# Standardize features
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

# PCA to reduce dimensionality
print()
pca = PCA(n_components=0.95)
x_pca = pca.fit_transform(x_scaled)

# Train-test split
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=1)

# Hyperparameter tuning using GridSearchCV
print()
param_grid = {
    'C': [1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1],
    'epsilon': [0.01, 0.1, 0.2]
}
grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5)
grid.fit(x_train, y_train)

print("Best Parameters:", grid.best_params_)

# Train model with best params
model = grid.best_estimator_
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)
print()
print("Predictions:\n", y_pred)

# Evaluation
print()
print("EVALUATION")
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2_val = r2_score(y_test, y_pred)

print("MEAN SQUARED ERROR:\n", mse)
print("MEAN ABSOLUTE ERROR:\n", mae)
print("R2_SCORE:\n", r2_val)

# Comparison table
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual': y_test, 'Predict': y_pred})
print(comparison)

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 4))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.show()

plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Predicted")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()

# Cross-validation
print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model, x_pca, y, cv=10)
print(scores)
print("THE MEAN SCORES:\n", np.mean(scores))
