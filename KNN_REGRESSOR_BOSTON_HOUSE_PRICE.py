# THE FOLLOWING CODE IS EXECUTED BY :- MR. SANKET GAIKWAD
# ALGORITHM :- KNN REGRESSOR 
# DATA SOURCE :- https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\PYTHON\\boston.csv")
print(a.columns.tolist())


##CHAS is a binary (0/1) categorical variable....If the count of 1 is very low (e.g., 35 out of 506), it means the variable is highly imbalanced.
print()
c = a['CHAS'].value_counts()
print(c)

#DROPPING UNWANTED COLUMNS
print()
a = a.drop(['CHAS','RAD', 'TAX'], axis = 1)


#DESCRIBING THE DATASET
print()
print(a.info())


#MISSING VALUES
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#FEATURE SELECTIONS
print()
x = a.drop('MEDV', axis = 1)
y = a['MEDV']


#STANDARD SCALER
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components=0.95)
x_pca =pca.fit_transform(x_scaled)

#SPLITTING OF DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state = 1)

#MODEL BUILDING
print()
model = KNeighborsRegressor(n_neighbors=4)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

#EVALUATIONS
print()
print("EVALUATION")
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2_val = r2_score(y_test, y_pred)

print()
print("MEAN SQUARRED ERROR:-\n", mse)
print("MEAN ABSOLUTE ERROR :-\n", mae)
print("R2_SCORE:-\n", r2_val)

print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model, x_pca, y, cv = 10)
print("THE SCORES ARE:-\n",scores)
print("THE MEAN SCORES :-\n", np.mean(scores))


#DATA VISUALIZATION
#Actual vs Predicted Plot

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', edgecolor='black')
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Actual vs Predicted MEDV - KNN Regressor")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.grid(True)
plt.show()


#Residual Plot
residuals = y_test - y_pred

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_pred, y=residuals, color='green')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted MEDV")
plt.ylabel("Residuals")
plt.title("Residual Plot - KNN Regressor")
plt.grid(True)
plt.show()

#Distribution of Actual vs Predicted
plt.figure(figsize=(8,6))
sns.kdeplot(y_test, label='Actual MEDV', shade=True)
sns.kdeplot(y_pred, label='Predicted MEDV', shade=True)
plt.title("Distribution: Actual vs Predicted MEDV")
plt.legend()
plt.show()

# Cross-Validation Scores Plot
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), scores, marker='o', linestyle='--', color='purple')
plt.axhline(np.mean(scores), color='red', linestyle=':')
plt.title("Cross-Validation Scores - KNN Regressor")
plt.xlabel("Fold")
plt.ylabel("R² Score")
plt.grid(True)
plt.show()
