#THE CODE IS EXECUTED BY :- SANKET GAIKWAD
#ALGORITHM :- GRADIENT BOOSTING REGRESSOR
#SOURCE :- wkaggle.com/datasets/ahmadrazakashif/housing-detail


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\Housing.csv")
print(a.columns.tolist())

print()
print(a.info())

print()
print(a.describe())

print()
print("TOTAL SUM OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())

print()
c = a[a.duplicated()]
print(c)

print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis=1)]
if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")

print()
cat = []
con = []

for i in a:
    if a[i].dtypes == 'object':
        cat.append(i)
    else:
        con.append(i)

print()
print("CATEGORICAL VALUES :-\n", cat)
print("CONTINUOUS VALUES :-\n", con)

print()
a = pd.get_dummies(a, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                               'airconditioning', 'prefarea', 'furnishingstatus'], drop_first=True)

print()
x = a.drop('price', axis=1)
y = a['price']

print()
y = np.log1p(y)

print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

print()
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

print()
# Gradient Boosting often outperforms Random Forest in tabular regression
model = GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)
model.fit(x_train, y_train)

print()
y_pred_log = model.predict(x_test)
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)
print(y_pred)

print()
print("EVALUATION")
print("MEAN SQUARED ERROR:-\n", mean_squared_error(y_test_actual, y_pred))
print("MEAN ABSOLUTE ERROR:-\n", mean_absolute_error(y_test_actual, y_pred))
print("R2_SCORE:-\n", r2_score(y_test_actual, y_pred))

print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual': y_test_actual, 'Predict': y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, x_scaled, y, cv=kf, scoring='r2')
print(scores)
print("THE MEAN R2 SCORE :-\n", np.mean(scores))
