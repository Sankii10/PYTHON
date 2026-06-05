#THE CODE IS EXECUTED BY :- MR.SANKET GAIWKAD
#ALGORITHM USED :- LINEAR REGRESSION( RIDGE AND GRIDSEARCHV)
#SOURCE :- https://www.kaggle.com/datasets/isaaclopgu/coca-cola-stock-daily-updated


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score,  GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#IMPORTING DATA FROM DATASET FILE
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\cococola.csv")
print(a.columns.tolist())

# CREATING THE TARGET VARIABLE AS NEXT DAY CLOSE PRICE AND DROPPING ANY ROWS WITH NaN VALUES RESULTING FROM SHIFT
print()
a['Target'] = a['Close'].shift(-1)
a = a.dropna()


 # CONVERTING DATE TO DATETIME AND EXTRACTING YEAR AND MONTH
print()
a['Date'] = pd.to_datetime(a['Date'], utc=True, errors='coerce')

a['year'] = a['Date'].dt.year
a['month'] = a['Date'].dt.month


# DEFINING TIME SERIES CROSS-VALIDATION WITH 10 SPLITS
tscv = TimeSeriesSplit(n_splits=10)


#DROPPING UNREQUIRED FEATURES
print()
a = a.drop(['Date','ticker', 'name'], axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION OF DATA
print()
print(a.info())

print()
print(a.describe())


#SEARCHING FOR DUPLICATED VALUES IN DATA
print()
print("TOTAL SUM OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())


#DISPLAYING DUPLICATED VALUE ROWS IN DATA
print()
c = a[a.duplicated()]
print(c)


#SEARCHING FOR MISSING VALUES/NAN IN DATA
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUES FOUND IN DATASET !!!")


#SEGREGATING THE CATEGORICAL AND CONTINOUS VARIABLES
print()
cat = []
con = []

for i in a:
    if a[i].dtype == 'object':
        cat.append(i)
    else:
        con.append(i)

print()
print("CATEGORICAL VALUES :-\n", cat)
print("CONTINOUS VALUES :-\n", con)


# DISPLAYING NORMALIZED VALUE COUNTS OF TARGET COLUMN
print()
print(a['Target'].value_counts(normalize= True))


# DISPLAYING CORRELATION OF ALL NUMERIC COLUMNS WITH TARGET SORTED DESCENDING
print()
print(a.corr(numeric_only= True)['Target'].sort_values(ascending = False))


#FEATURES SELECTION
print()
x = a.drop('Target', axis = 1)
y = a['Target']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


# CALCULATING TRAIN SIZE AS 80% OF DATA AND SPLITTING FEATURES INTO TRAIN AND TEST SETS
print()
train_size = int(0.8 * len(x_scaled))
x_train, x_test = x_scaled[:train_size], x_scaled[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


#MODEL BUILDING AND IMPLEMENTATION
print()
ridge_params = {'alpha':np.logspace(3,-3,10)}
ridge_cv = GridSearchCV(Ridge(), ridge_params, cv = 10, scoring ='r2')
ridge_cv.fit(x_train, y_train)

print()
print('BEST PARAMETERS:-\n', ridge_cv.best_params_)
print("BEST SCORES :-\n", ridge_cv.best_score_)
print("BEST INDEX :-\n", ridge_cv.best_index_)

print()
model = ridge_cv.best_estimator_
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print('EVALUATION')
print("MEAN SQUARED ERROR:-\n", mean_squared_error(y_test, y_pred))
print("MEAN ASBOLUTE ERROR;-\n", mean_absolute_error(y_test, y_pred))
print("R2_SCORE :-\n", r2_score(y_test, y_pred))


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model, x_scaled, y, cv=tscv, scoring='r2')
print(scores)
print("MEAN SCORES :-\n", np.mean(scores))


#DATA VISUALISATION


# PLOT ACTUAL VS PREDICTED VALUES
plt.figure(figsize=(15,6))
plt.plot(y_test.values, label='Actual', color='blue')
plt.plot(y_pred, label='Predicted', color='red')
plt.title('ACTUAL VS PREDICTED CLOSE PRICES')
plt.xlabel('Index')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# SCATTER PLOT OF ACTUAL VS PREDICTED VALUES
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test.values, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # DIAGONAL LINE
plt.title('SCATTER PLOT OF ACTUAL VS PREDICTED')
plt.xlabel('Actual Close Price')
plt.ylabel('Predicted Close Price')
plt.show()

# RESIDUAL PLOT
residuals = y_test.values - y_pred
plt.figure(figsize=(10,6))
sns.histplot(residuals, bins=50, kde=True)
plt.title('RESIDUALS DISTRIBUTION')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# HEATMAP OF CORRELATION MATRIX
plt.figure(figsize=(10,8))
sns.heatmap(a.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('CORRELATION HEATMAP')
plt.show()

# PAIRPLOT FOR SELECTED FEATURES VS TARGET
selected_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'year', 'month', 'Target']
sns.pairplot(a[selected_features])
plt.suptitle('PAIRPLOT OF SELECTED FEATURES', y=1.02)
plt.show()



