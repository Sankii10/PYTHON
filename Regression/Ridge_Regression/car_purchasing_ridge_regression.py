#THE FOLLOWING CODE IS EXECUTED BY MR.SANKET GAIKWAD
#ALGORITHM :- LINEAR REGRESSOR(RIDGE AND GRIDSEARCHV)
#SOURCE :- https://www.kaggle.com/datasets/dev0914sharma/car-purchasing-model



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\car.csv")
print(a.columns.tolist())


#DROPING UN-REQUIRED FEATURES
print()
a = a.drop(['Customer Name', 'Customer e-mail', 'Country'], axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION OF DATA
print()
print(a.info())

print()
print(a.describe())


#TOTAL SUM OF DUPLICATED VALUES IN DATA
print()
print("TOTAL SUM OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())


#DISPLAY DUPLICATED VALUES ROWS
print()
c = a[a.duplicated()]
print(c)


#SEARCHING FOR MISSING VALUES IN DATA
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATSET !!!")


#SEGREGATING THE CATEGORICAL AND CONTINOUS VARIABLES(COLUMNS)
print()
cat = []
con = []

for i in a.columns:
    if a[i].dtypes == 'object':
        cat.append(i)
    else:
        con.append(i)

print()
print("CATEGORICAL VALUES :-\n", cat)
print("CONTINOUS VALUES :-\n", con)


#FEATURE SELECTION
print()
x = a.drop('Car Purchase Amount', axis = 1)
y = a['Car Purchase Amount']


# PRINTS THE RELATIVE FREQUENCY (NORMALIZED COUNTS) OF EACH UNIQUE VALUE IN "CAR PURCHASE AMOUNT"
print()
print(a['Car Purchase Amount'].value_counts(normalize = True))

# PRINTS THE CORRELATION VALUES (PEARSON) OF ALL NUMERIC COLUMNS WITH RESPECT TO "CAR PURCHASE AMOUNT", SORTED IN DESCENDING ORDER
print()
print(a.corr(numeric_only= True)['Car Purchase Amount'].sort_values(ascending = False))


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
x_train, x_test , y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state = 1)


#MODEL BULIDING AND IMPLEMENTATION
print()
ridge_params = {'alpha':np.logspace(3,-3,10)}
ridge_cv = GridSearchCV(Ridge(), ridge_params, cv = 10, scoring = 'r2')
ridge_cv.fit(x_train, y_train)

print()
print("BEST INDEXES :-\n", ridge_cv.best_index_)
print("BEST SCORE:-\n", ridge_cv.best_score_)
print("BEST PARAMETERS :-\n", ridge_cv.best_params_)

print()
model = ridge_cv.best_estimator_
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
print("MEAN SQUARED ERROR:-\n", mean_squared_error(y_test, y_pred))
print("MEAN ABSOLUTE ERROR :-\n", mean_absolute_error(y_test, y_pred))
print("R2_SCORE:-\n", r2_score(y_test, y_pred))


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)


#CROSS SCORE VALIDATION
print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model,x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))



#DATA VISUALISATION

# 1. CORRELATION HEATMAP
plt.figure(figsize=(10, 8))
sns.heatmap(a.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 2. PCA VARIANCE RATIO
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid()
plt.show()

# 3. ACTUAL vs PREDICTED SCATTERPLOT
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color="purple")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Car Purchase Amount")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# 4. RESIDUALS DISTRIBUTION
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True, color="teal")
plt.xlabel("Residuals")
plt.title("Distribution of Residuals")
plt.show()

# 5. RESIDUALS vs PREDICTED
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()

# 6. COMPARISON BARPLOT (ACTUAL vs PREDICTED for sample 25 points)
comparison_sample = comparison.head(25)
comparison_sample.plot(kind="bar", figsize=(12, 6))
plt.title("Actual vs Predicted (First 25 Samples)")
plt.ylabel("Car Purchase Amount")
plt.show()



