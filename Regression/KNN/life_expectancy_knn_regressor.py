# THE FOLLOWING CODE IS EXECUTED BY :- MR. SANKET GAIKWAD
# SOURCE :- KAGGLE( LIFE EXPENTANCY DATASET)
# ALGORITHM USED IS KNN REGRESSOR

#########################################################################################################################################


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

a = pd.read_csv("C:\\Users\\anujg\\OneDrive\\Desktop\\MY DATA\\DATASET\\life_who.csv")
print(a.columns.tolist())

print()
print(a.info())

#FINDING MISSING VALUES
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#HANDLING MISSING VALUES
print()
for col in a.drop(['Country', 'Year', 'Status'], axis = 1).columns:
    a[col] = a[col].fillna(a[col].median())


#LABEL ENCODING
print()
le = LabelEncoder()
for col in ['Country','Status']:
    a[col] = le.fit_transform(a[col])


#FEATURE SELECTION
print()
x = a.drop('Life expectancy ', axis = 1)
y = a['Life expectancy ']


#STANDARD SCALER
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components = 0.95)
x_pca =pca.fit_transform(x_scaled)


#MODEL BUILDING
print()
model = KNeighborsRegressor(n_neighbors= 5)
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state= 1)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

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
scores = cross_val_score(model,x_pca, y, cv  =10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))



#DATA VISUALIZATION
# Scree Plot (Explained Variance by PCA Components)
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', color='green')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Explained Variance')
plt.grid(True)
plt.show()

#Actual vs Predicted Plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.xlabel("Actual Life Expectancy")
plt.ylabel("Predicted Life Expectancy")
plt.title("Actual vs Predicted Life Expectancy (KNN Regression)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.grid(True)
plt.show()


#Residual Plot (Errors)
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, color='purple', bins=30)
plt.xlabel("Residuals")
plt.title("Distribution of Residuals")
plt.grid(True)
plt.show()


#Cross-Validation Scores Bar Plot
plt.figure(figsize=(8, 5))
plt.bar(range(1, 11), scores, color='teal')
plt.xlabel("Fold")
plt.ylabel("Cross-Validation Score")
plt.title("Cross-Validation Scores for KNN Regressor")
plt.ylim(0, 1)
plt.grid(True)
plt.show()
