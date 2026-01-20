#THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- LINEAR REGRESSOR : RIDGE AND GRIDSEARCHV
#SOURCE :- https://www.kaggle.com/datasets/nudratabbas/global-green-energy-pulse-real-time-renewable



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Data loading
a = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\Desktop\\My Data\\green.csv")
print(a.columns.tolist())

# Time conversion
a['time'] = pd.to_datetime(a['time'], format='mixed')
a.set_index('time', inplace=True)
a.sort_index(inplace=True)

print(a.index.inferred_freq)

# Drop static columns
a = a.drop(['city', 'latitude', 'longitude'], axis=1)

print()
print(a.info())
print()
print(a.describe())

# Duplicate check
print("\nDUPLICATED VALUES :-")
print(a.duplicated().sum())
print(a[a.duplicated()])

a = a.drop_duplicates()

# Missing value check
print()
print(a.isnull().sum())

missing_value = a[a.isna().any(axis=1)]
if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!")

# Identify categorical and continuous columns
cat = []
con = []

for i in a:
    if a[i].dtype == 'object':
        cat.append(i)
    else:
        con.append(i)

print("\nCATEGORICAL COLUMN :-\n", cat)
print("CONTINUOUS COLUMN :-\n", con)

# Create lag features
a['lag_1'] = a['green_score'].shift(1)
a['lag_24'] = a['green_score'].shift(24)
a['rolling_24'] = a['green_score'].rolling(24).mean()

a.dropna(inplace=True)

# Feature and target separation
x = a.drop(columns=['green_score'])
y = a['green_score']


# Displays the relative frequency (proportion) of each unique green_score value in the dataset
print()
print(a['green_score'].value_counts(normalize=True))


# Computes correlation of all numeric features with green_score and sorts them in descending order
print()
print(a.corr(numeric_only=True)['green_score'].sort_values(ascending=False))

# Time-aware train test split
x_train = x.iloc[:-24]
x_test  = x.iloc[-24:]

y_train = y.iloc[:-24]
y_test  = y.iloc[-24:]

# Scaling without data leakage
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled  = sc.transform(x_test)

# Ridge regression model
model = Ridge(alpha=1.0)
model.fit(x_train_scaled, y_train)

# Prediction for next 24 hours
y_pred = model.predict(x_test_scaled)

# Model evaluation
print("\nEVALUATION")
print("R2 Score :", r2_score(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("MAE :", mean_absolute_error(y_test, y_pred))

# Actual vs predicted comparison
print("\nCOMPARISON")
comparison = pd.DataFrame({'Actual': y_test.values,'Predicted': y_pred}, index=y_test.index)
print(comparison)
