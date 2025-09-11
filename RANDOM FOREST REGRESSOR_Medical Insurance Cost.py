#THE CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- RANDOM FOREST REGRESSOR
#SOURCE :- https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#IMPORTING OF THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\insurance.csv")
print(a.columns.tolist())


#INFORMATION AND STATISTICAL DESCRIPTION OF THE DATA
print()
print(a.info())

print()
print(a.describe())


#TOTAL SUM OF DUPLICATED VALUES
print()
print("TOTAL SUM OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())


#ROW WISE DUPLICATED VALUES
print()
c = a[a.duplicated()]
print(c)


#DROPPING THE DUPLICATED VALUES
print()
print(a.drop_duplicates())


#SEARCHING FOR NULL VALUES
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


# SEPARATING CATEGORICAL AND CONTINUOUS COLUMNS FROM THE DATAFRAME
print()
cat = []
con = []

for i in a.columns:
    if a[i].dtypes == 'object':
        cat.append(i)
    else:
        con.append(i)

print()
print("CATEGORICAL COLUMNS:-\n", cat)
print("CONTINOUS VALUES :-\n", con)


#ONE HOT ENCODING
print()
a = pd.get_dummies(a, columns = ['sex', 'smoker', 'region'], drop_first= True)


# THIS PRINTS THE RELATIVE FREQUENCY DISTRIBUTION OF CHARGES COLUMN  
print()
print(a['charges'].value_counts(normalize=True))


# THIS PRINTS THE CORRELATION OF ALL NUMERIC COLUMNS WITH CHARGES SORTED IN DESCENDING ORDER  
print()
print(a.corr(numeric_only=True)['charges'].sort_values(ascending = False))


#FEATURES SELECTION
print()
x = a.drop('charges', axis = 1)
y = a['charges']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca =PCA(n_components= 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLITING OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state=1)


#MODEL BUILDING
print()
model = RandomForestRegressor(n_estimators=150, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', bootstrap=True, random_state=42, n_jobs=-1)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
print("MEAN SQUARED ERROR:-\n", mean_squared_error(y_test, y_pred))
print("MEAN ABSOLUTE ERROR:-\n", mean_absolute_error(y_test, y_pred))
print("R2_SCORE:-\n", r2_score(y_test, y_pred))


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


# 1. DISTRIBUTION OF TARGET VARIABLE (CHARGES)
plt.figure(figsize=(8,5))
sns.histplot(a['charges'], kde=True, bins=30, color='blue')
plt.title("Distribution of Insurance Charges")
plt.xlabel("Charges")
plt.ylabel("Frequency")
plt.show()

# 2. CORRELATION HEATMAP
plt.figure(figsize=(10,7))
sns.heatmap(a.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# 3. ACTUAL VS PREDICTED (SCATTER PLOT)
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') # perfect prediction line
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Charges")
plt.show()

# 4. RESIDUAL PLOT
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color="purple")
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Charges")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# 5. COMPARISON BAR PLOT (FIRST 25 RECORDS)
comparison = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
comparison = comparison.reset_index(drop=True)
comparison.head(25).plot(kind='bar', figsize=(12,6))
plt.title("Comparison of Actual vs Predicted Charges (First 25 Samples)")
plt.xlabel("Index")
plt.ylabel("Charges")
plt.xticks(rotation=0)
plt.show()


