# THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
# ALF+GORITHM USED :- RANDOM FOREST REGRESSOR
# SOURCE :- KAGGLE ( https://www.kaggle.com/datasets/rohitgrewal/restaurant-sales-data )


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\Sales.csv")
print(a.columns.tolist())


#CONVERTED THE DATE COLUMN TO A DATETIME FORMAT (WITH DAY-FIRST FORMAT) AND EXTRACTED THE YEAR AND MONTH AS SEPARATE FEATURES.
print()
a['Date'] = pd.to_datetime(a['Date'], dayfirst= True, errors = 'coerce')

a['Year'] = a['Date'].dt.year
a['Month'] = a['Date'].dt.month


#DROPPING THE UNWANTED FEATURES
print()
a = a.drop(['Order ID', 'Date'], axis = 1)


#CREATED A NEW COLUMN TOTALSALES BY MULTIPLYING PRICE AND QUANTITY TO REPRESENT THE TOTAL REVENUE PER ORDER.
print()
a['TotalSales'] = a['Price'] * a['Quantity']
print(a.columns.tolist())


#INFORMATION AND DESCRIPTION ABOUT THE DATA
print()
print(a.info())

print()
print(a.describe())

print()
print(a.head(10))


#TOTAL NUMBER OF DUPLICATES IN DATASET 
print()
print("TOTAL SUM OF DUPLICATES")
print(a.duplicated().sum())

#ROWWISE DUPLICATES IN DATA
print()
c = a[a.duplicated()]
print(c)


#DROPPING THE DUPLICATE VALUES AND ITS ROWS
a = a.drop_duplicates()


#TO FIND MISSING VALUES
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATSET !!!")


#CONVERTED THE SPECIFIED CATEGORICAL COLUMNS INTO DUMMY (ONE-HOT ENCODED) VARIABLES AND DROPPED THE FIRST CATEGORY TO AVOID MULTICOLLINEARITY.
print()
a = pd.get_dummies(a, columns = ['Product','Purchase Type', 'Payment Method', 'Manager', 'City'], drop_first = True)


#FEATURE SELECTION
print()
x = a.drop('TotalSales', axis = 1)
y = a['TotalSales']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components = 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLIT OF DATA IN TRAIN AND TEST
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state= 1)


#MODEL BUILDING
print()
model = RandomForestRegressor(n_estimators= 150, criterion='squared_error', max_depth = 4)
model.fit(x_train, y_train)

print()
y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2_val = r2_score(y_test, y_pred)

print()
print("MEAN SQUARED ERROR :-\n", mse)
print("MEAN ABSOLUTE ERROR:-\n", mae)
print("R2_SCORE:-\n", r2_val)


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model,x_pca, y, cv  =10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))



#DATA VISUALISATION

#Distribution of Total Sales
plt.figure(figsize=(8,5))
sns.histplot(a['TotalSales'], bins=30, kde=True)
plt.title('Distribution of Total Sales')
plt.xlabel('Total Sales')
plt.ylabel('Frequency')
plt.show()


#Feature Importance (from Random Forest)
importances = model.feature_importances_
features = [f'PC{i+1}' for i in range(x_pca.shape[1])]  # PCA components

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title('Feature Importance (PCA Components)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()


#Actual vs Predicted Scatter Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.title('Actual vs Predicted Total Sales')
plt.xlabel('Actual Total Sales')
plt.ylabel('Predicted Total Sales')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Line of perfect prediction
plt.show()


#Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
sns.histplot(residuals, bins=30, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()


#. Model Performance Bar Chart
metrics = {'MSE': mse, 'MAE': mae, 'R²': r2_val}
plt.figure(figsize=(6,4))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
plt.title('Model Performance Metrics')
plt.show()


#Cross-Validation Score Distribution
plt.figure(figsize=(8,6))
sns.boxplot(scores)
plt.title('Cross-Validation Score Distribution')
plt.xlabel('Cross-Validation Scores')
plt.show()


