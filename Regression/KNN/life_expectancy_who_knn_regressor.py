#THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
# ALGORITHM :- KNN REGRESSOR 
# SOURCE :- Life Expectancy (WHO) ( https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who )



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
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\life.csv")
print(a.columns.tolist())


#DROPPING UN-NECESSARY COLUMNS
print()
a = a.drop(['Population', 'Year', 'infant deaths', 'percentage expenditure'], axis = 1)


#INFORMATION OF DATA
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
for col in ['Life expectancy ', 'Adult Mortality', 'Alcohol', 'Hepatitis B', ' BMI ', 'Polio', 'Total expenditure', 'Diphtheria ', 'GDP', ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources', 'Schooling']:
    a[col] = a[col].fillna(a[col].median())


#LABEL ENCODING
print()
le = LabelEncoder()
for col in ['Country', 'Status']:
    a[col] = le.fit_transform(a[col])


#FEATURE SELECTION
print()
x = a.drop('Life expectancy ', axis = 1)
y = a['Life expectancy ']


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
model = KNeighborsRegressor(n_neighbors=4)
model.fit(x_train, y_train)

print()
y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
mse = mean_squared_error(y_test, y_pred)
mae =mean_absolute_error(y_test, y_pred)
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
print("CROSSS SCORE VALIDATION")
scores = cross_val_score(model,x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALISATION

#1. Predicted vs Actual Scatter Plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
plt.xlabel("Actual Life Expectancy")
plt.ylabel("Predicted Life Expectancy")
plt.title("Predicted vs Actual Values (KNN)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 45-degree line
plt.show()

# 2. Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(6,4))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Life Expectancy")
plt.ylabel("Residuals")
plt.title("Residual Plot (KNN)")
plt.show()

# 3. Distribution of Errors
plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=20, kde=True)
plt.xlabel("Prediction Error")
plt.title("Distribution of Errors (KNN)")
plt.show()

# 4. Cross-validation Scores Plot
plt.figure(figsize=(6,4))
plt.bar(range(1, len(scores)+1), scores)
plt.xlabel("Fold Number")
plt.ylabel("R² Score")
plt.title("Cross-validation Scores (KNN)")
plt.show()

