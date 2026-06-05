#THE FOLLOWING CIODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM :- SUPPORT VECTOR MACHINE ( SVR )
#SOURCE :- CALORIES BURNED (https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#LOADING OF BOTH CALORIES.CSV AND EXERCISE.CSV FILE
x = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\calories.csv")
y = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\exercise.csv")


# COMBINING/AMALGAMATING BOTH THE FILES IN ONE SINGLE FILE
print()
a = pd.concat([x,y], axis = 1)
print(a.columns.tolist())


#INFORMATION OF DATASET 
print()
print(a.info())


#SEARCHING MISSING VALUES
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#LABEL ENCODING
print()
le = LabelEncoder()
a['Gender'] = le.fit_transform(a['Gender'])


#FEATURE SELECTION
print()
x = a.drop('Calories', axis = 1)
y = a['Calories']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components = 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLIT OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state = 1)


#MODEL BUILDING
print()
model = SVR(kernel = 'rbf')
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
print("THE MEAN SCORE IS:-\n", np.mean(scores))


#DATA VISUALIZATION

#Heatmap for Correlation Matrix
plt.figure(figsize=(10,6))
sns.heatmap(a.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


#Distribution of the Target Variable (Calories)
plt.figure(figsize=(8,5))
sns.histplot(a['Calories'], bins=30, kde=True, color='green')
plt.title("Distribution of Calories")
plt.xlabel("Calories")
plt.ylabel("Frequency")
plt.show()


#Explained Variance by PCA Components
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Explained Variance")
plt.grid()
plt.show()


# Actual vs Predicted Scatter Plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color='blue')
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("Actual vs Predicted Calories (SVR)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
plt.show()




