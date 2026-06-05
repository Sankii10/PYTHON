#THE FOLLOWING CODE IS EXECUTED BY :- MR. SANKET GAIKWAD
#ALGORITHM USED :- KNN REGRESSOR
#SOURCE :- Medical Cost Personal Datasets ( https://www.kaggle.com/datasets/mirichoi0218/insurance )


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\insurance.csv")
print(a.columns.tolist())


#INFORMATION AND DECRIBING THE DATA
print()
print(a.info())

print()
print(a.describe())


#TO FIND TOTAL SUM OF DUPLICATED VALUES IN DATA
print()
print(a.duplicated().sum())


#TO SEARCH FOR DUPLICATED ROWS
print()
c = a[a.duplicated()]
print("THE DUPLICATED ROWS ARE :-\n", c)

#DROPPING DUPLICATED VALUES
a.drop_duplicates(inplace=True)


#SEARCHING FOR MISSING VALUES
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#CONVERTS CATEGORICAL COLUMNS 'SEX', 'SMOKER', AND 'REGION' INTO NUMERICAL COLUMNS USING ONE-HOT ENCODING AND DROPS THE FIRST CATEGORY TO AVOID REDUNDANCY.
print()
a = pd.get_dummies(a, columns=['sex','smoker','region'], drop_first=True)


#FEATURE SELECTIOn
print()
x = a.drop('charges', axis = 1)
y = a['charges']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLITING OF DATA IN TRAIN AND TEST
print()
x_train, x_test, y_train, y_test  = train_test_split(x_pca, y, test_size = 0.2, random_state= 1)


#MODEL BUILDING
print()
model = KNeighborsRegressor(n_neighbors=4,weights='distance', algorithm='auto')
model.fit(x_train, y_train)

y_pred =model.predict(x_test)
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
scores = cross_val_score(model, x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALISATION

#Correlation Heatmap
plt.figure(figsize=(10,7))
sns.heatmap(a.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


#Distribution of Target Variable (charges)
plt.figure(figsize=(8,5))
sns.histplot(y, bins=30, kde=True, color='skyblue')
plt.title("Distribution of Insurance Charges")
plt.xlabel("Charges")
plt.ylabel("Frequency")
plt.show()


#Scatter Plot: Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Charges")
plt.show()


#Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_pred, y=residuals, color='purple', alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Charges")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


#Distribution of Residuals
plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=30, kde=True, color='orange')
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.show()


