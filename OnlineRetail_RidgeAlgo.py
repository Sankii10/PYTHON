"""
THE FOLLOWING CODE IS EXECUTED BY ...
NAME : SANKET GAIKWAD
DATASET : ONLINE RETAIL DATASET (KAGGLE)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA

a = pd.read_csv("C:\\Users\\anujg\\OneDrive\\Desktop\\MY DATA\\DATASET\\OnlineRetail.csv", encoding = 'latin1')
print(a.columns.tolist())

print()
a = a.drop(['InvoiceNo','InvoiceDate','StockCode'], axis = 1)

#FORMATION OF NEW COLUMN
a['TotalPrice'] = a['Quantity'] * a['UnitPrice']
print(a['TotalPrice'].head(10))

print()
print(a.info())

print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a[['Description','CustomerID']].isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")

print()
for col in ['CustomerID','Description']:
    a[col] = a[col].fillna(a[col].mode()[0])

print()
le = LabelEncoder()
for col in ['Description','Country']:
    a[col] = le.fit_transform(a[col])

print()
x = a.drop('TotalPrice', axis = 1)
y = a['TotalPrice']

print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

print()
pca = PCA(n_components = 0.95)
x_pca = pca.fit_transform(x_scaled)

print()
ridge_params = {'alpha': np.logspace(-3,3,10)}
ridge_cv = GridSearchCV(Ridge(), ridge_params, cv = 10, scoring = 'r2')
ridge_cv.fit(x_pca, y)

print()
print("BEST PARAMETERS")
print("THE BEST PARAMEETERS:-\n",ridge_cv.best_params_)
print("THE BEST INDEX:-\n", ridge_cv.best_index_)
print("THE BEST SCORE :-\n", ridge_cv.best_score_)

print()
model=  ridge_cv.best_estimator_
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size= 0.2,random_state = 1)

model.fit(x_train, y_train)

print()
y_pred = model.predict(x_test)
print("THE PREDICTIONS ARE:-\n", y_pred)

print()
print("EVALUATION")
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2_val = r2_score(y_test, y_pred)

print()
print("MEAN SQUARED ERROR :-\n", mse)
print("MEAN ABSOLUTE ERROR :-\n", mae)
print("R2_SCORE:-\n", r2_val)

print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores  = cross_val_score(model, x_pca, y, cv=10)
print("THE SCORES ARE :-\n", scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALIZATION

#ACTUAL VS PREDICTED SCATTERED PLOT
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.3, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Total Price')
plt.ylabel('Predicted Total Price')
plt.title('Actual vs Predicted Values (Ridge Regression)')
plt.grid(True)
plt.tight_layout()
plt.show()

#RESIDUAL DISTRIBUTION PLOT
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
sns.histplot(residuals, kde=True, bins=50, color='slateblue')
plt.xlabel('Residual (Actual - Predicted)')
plt.title('Residual Distribution')
plt.grid(True)
plt.tight_layout()
plt.show()


#Cross-Validation Scores Boxplot
plt.figure(figsize=(8,6))
sns.boxplot(data=scores, orient='h', color='skyblue')
plt.xlabel('Cross-Validation R² Scores')
plt.title('10-Fold Cross-Validation Results')
plt.tight_layout()
plt.show()
