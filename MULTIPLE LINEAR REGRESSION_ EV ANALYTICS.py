#THE FOLLOWING CODE IS EXECUTED BY :- MR. SANKET GAIKWAD
#ALGORITHM USED :- MULTIPLE LINEAR REGRESSION
#SOURCE :- Electric Vehicle Analytics Dataset ( https://www.kaggle.com/datasets/khushikyad001/electric-vehicle-analytics-dataset )



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\Ev.csv")
print(a.columns.tolist())


#DROPPING UNWANTED FEATURES
print()
a = a.drop(['Vehicle_ID', 'CO2_Saved_tons', 'Monthly_Charging_Cost_USD', 'Electricity_Cost_USD_per_kWh'], axis = 1)


#INFORMATION ABOUT DATA AND DESCRIBING OF DATA
print()
print(a.info())

print()
print(a.describe())


#TOTAL SUM OF DUPLICATE VALUES IN DATA
print()
print("TOTAL SUM OF DUPLICATED VALUES")
print(a.duplicated().sum())


#DUPLICATED VALUES ROW WISE
print()
c = a[a.duplicated()]
print(c)


#SEARCHING FOR MISSING VALUES
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#CONVERSION OF CATEGORICAL VALUES INTOP CONTINOUS VALUES :- ONE HOT ENCODING
print()
a = pd.get_dummies(a, columns = ['Make', 'Model', 'Region', 'Vehicle_Type', 'Usage_Type'], drop_first = True)


#FEATURE SELECTIONS
print()
x = a.drop('Resale_Value_USD', axis = 1)
y = a['Resale_Value_USD']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLIT OF TRAIN AND TEST DATA 
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state= 1)


#MODEL BUILDING
print()
model = LinearRegression(fit_intercept=True, n_jobs=-1)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2_val = r2_score(y_test, y_pred)

print()
print("MEAN SQUARED ERROR:-\n", mse)
print("MEAN ABSOLUTE ERROR:-\n", mae)
print("R2_SCORE:-\n", r2_val)


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model,x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALISATION

# 1. PCA EXPLAINED VARIANCE PLOT
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='blue')
plt.xlabel('NUMBER OF PRINCIPAL COMPONENTS')
plt.ylabel('CUMULATIVE EXPLAINED VARIANCE')
plt.title('PCA EXPLAINED VARIANCE RATIO')
plt.grid()
plt.show()

# 2. ACTUAL VS PREDICTED SCATTER PLOT
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.xlabel('ACTUAL RESALE VALUE')
plt.ylabel('PREDICTED RESALE VALUE')
plt.title('ACTUAL VS PREDICTED VALUES')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # DIAGONAL LINE
plt.show()

# 3. RESIDUAL PLOT
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.6, color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('PREDICTED VALUES')
plt.ylabel('RESIDUALS')
plt.title('RESIDUAL PLOT')
plt.show()

# 4. DISTRIBUTION OF RESIDUALS (NORMALITY CHECK)
plt.figure(figsize=(8,6))
sns.histplot(residuals, kde=True, color='green', bins=20)
plt.title('DISTRIBUTION OF RESIDUALS')
plt.xlabel('RESIDUALS')
plt.show()

# 5. CROSS-VALIDATION SCORES PLOT
plt.figure(figsize=(8,6))
plt.plot(range(1, len(scores)+1), scores, marker='o', linestyle='-', color='green')
plt.axhline(np.mean(scores), color='red', linestyle='--', label=f'MEAN SCORE: {np.mean(scores):.4f}')
plt.xlabel('FOLD')
plt.ylabel('R² SCORE')
plt.title('CROSS-VALIDATION SCORES')
plt.legend()
plt.grid()
plt.show()

# 6. CORRELATION HEATMAP OF ORIGINAL FEATURES (BEFORE PCA)
plt.figure(figsize=(12,8))
sns.heatmap(x.corr(), annot=False, cmap='coolwarm')
plt.title('CORRELATION HEATMAP OF ORIGINAL FEATURES')
plt.show()










