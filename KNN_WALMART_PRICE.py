#THE FOLLOWING CODE IS EXECUTED BY :- MR>SANKET GAIKWAD
#ALGORITHM USED :- KNN REGRESSOR 
#SOURCE :- Walmart Dataset ( https://www.kaggle.com/datasets/yasserh/walmart-dataset )



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#IMPORTING THE DATA 
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\Walmart.csv")
print(a.columns.tolist())


#CONVERTED THE DATE COLUMN FROM STRING FORMAT TO DATETIME OBJECTS IN PANDAS, HANDLING DAY-FIRST FORMAT AND INVALID DATES AS NaT.
print()
a['Date'] = pd.to_datetime(a['Date'], dayfirst= True, errors='coerce')

#EXTRACTED THE YEAR AND MONTH COMPONENTS FROM THE DATETIME COLUMN AND CREATED TWO NEW NUMERIC COLUMNS: YEAR AND MONTH.
a['Year'] = a['Date'].dt.year
a['Month'] = a['Date'].dt.month


#DROPPING NOT REQUIRED FEATURES
print()
a = a.drop(columns = ['Date'])


#INFORMATION ABOUT DATA
print()
print(a.info())


#DESCRIBING THE DATA
print()
print(a.describe())


#TOTAL NUMBER OF DUPLICATES
print()
print("TOTAL SUM OF DUPLICATES")
print(a.duplicated().sum())

#DUPLICATE VALUES ROW-WISE
print()
c = a[a.duplicated()]
print(c)


# SEARCHING FOR MISSING VALUES
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#CONVERTING CATEGORICAL VALUES INTO CONTINOUS - USE OF DUMMY VARIABLE
print()
a=  pd.get_dummies(a, columns = ['Store', 'Month'], drop_first= True)


#FEATURE SELECTION
print()
x = a.drop('Weekly_Sales', axis = 1)
y = a['Weekly_Sales']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLITING OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state= 1)


#MODEL BUILDING
print()
model = KNeighborsRegressor(n_neighbors=4, weights='distance', algorithm='auto')
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
print("R2-SCORE:-\n", r2_val)


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model, x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))



#DATA VISUALISATION

#Correlation Heatmap (Before PCA)
# Correlation heatmap
plt.figure(figsize=(12,6))
sns.heatmap(a.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


#PCA Explained Variance
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid()
plt.show()


#Actual vs Predicted Scatter Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.xlabel("Actual Weekly Sales")
plt.ylabel("Predicted Weekly Sales")
plt.title("Actual vs Predicted Sales (KNN)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)  # reference line
plt.show()


#Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8,5))
plt.scatter(y_pred, residuals, alpha=0.6, color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


#Distribution of Actual vs Predicted
plt.figure(figsize=(8,6))
sns.kdeplot(y_test, label="Actual", color='blue')
sns.kdeplot(y_pred, label="Predicted", color='orange')
plt.title("Distribution of Actual vs Predicted Sales")
plt.legend()
plt.show()


