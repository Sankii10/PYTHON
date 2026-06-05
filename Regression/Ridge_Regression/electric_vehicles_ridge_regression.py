#THE FOLLOWING CODE IS EXECUTED BY :- MR. SANKET GAIKWAD
# ALGORITHM :- LINEAR REGRESSOR(RIDGE AND GRIDSEARCH)
# SOURCE :- Electric Vehicle Specs Dataset (2025)(https://www.kaggle.com/datasets/urvishahir/electric-vehicle-specifications-dataset-2025)



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\electric.csv")
print(a.columns.tolist())


#DROPPING UNWANTED COLUMNS 
print()
a = a.drop(['brand', 'model','fast_charge_port','number_of_cells'], axis = 1)


#INFORMATION ABOUT DATASET
print()
print(a.info())


#SEARCHING FOR NULL VALUES
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#HANDLING OF NULL VALUES
print()
for col in ['torque_nm', 'fast_charging_power_kw_dc','towing_capacity_kg', 'cargo_volume_l']:
    a[col] = a[col].fillna(a[col].median())


#LABEL ENCODER
print()
le = LabelEncoder()
for col in ['battery_type','drivetrain', 'segment','car_body_type']:
    a[col] = le.fit_transform(a[col])


#FEATURE SELECTION
print()
x = a.drop( 'range_km', axis = 1)
y = a[ 'range_km']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca = pca.fit_transform(x_scaled)


#SPLITTING OF DATA INTO TRAIN AND TEST
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state = 1)


#MODEL BUILDING
print()
ridge_params = {'alpha':np.logspace(3,-3,10)}
ridge_cv = GridSearchCV(Ridge(), ridge_params,cv = 10, scoring='r2')
ridge_cv.fit(x_pca, y)

print()
print("BEST PARAMETERS:-\n", ridge_cv.best_params_)
print("BEST INDEXES:-\n", ridge_cv.best_index_)
print("BEST SCORE:-\n", ridge_cv.best_score_)

print()
model = ridge_cv.best_estimator_
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
print("MEAN SQUARRED ERROR:-\n", mse)
print("MEAN ABSOLUTE ERROR:-\n", mae)
print("R2_SCORE:-\n", r2_val)


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("SCORES")
scores = cross_val_score(model,x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALIZATION

#Actual vs Predicted Plot (Scatter Plot)
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='royalblue', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', lw=2)
plt.xlabel('Actual Range (km)')
plt.ylabel('Predicted Range (km)')
plt.title('Actual vs Predicted Range using Ridge Regression')
plt.grid(True)
plt.show()


#Residual Plot
residuals = y_test - y_pred

plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, color='green', edgecolor='k')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Range (km)')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)
plt.show()


#Distribution of Residuals (Histogram + KDE)
sns.set(style="whitegrid")
plt.figure(figsize=(8,6))
sns.histplot(residuals, kde=True, bins=30, color='coral')
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


#Cross-Validation Scores (Bar Plot)
plt.figure(figsize=(8,6))
plt.bar(range(1, 11), scores, color='skyblue', edgecolor='black')
plt.axhline(y=np.mean(scores), color='red', linestyle='--', label='Mean CV Score')
plt.xlabel('Fold')
plt.ylabel('R² Score')
plt.title('10-Fold Cross Validation Scores')
plt.legend()
plt.show()
