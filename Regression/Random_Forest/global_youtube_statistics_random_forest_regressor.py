#THE CODE IS EXECUTED BY MR.SANKET GAIKWAD
#ALGORITHM USED :- RANDOM FOREST REGRESSOR
#SOURCE :- https://www.kaggle.com/datasets/nelgiriyewithana/global-youtube-statistics-2023


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

a = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\Desktop\\My Data\\youtube.csv", encoding='latin1')
print(a.columns.tolist())


#DROPING NOT REQUIRED FEATURES
print()
a = a.drop(['video views', 'video_views_rank', 'lowest_monthly_earnings', 'highest_monthly_earnings',
 'lowest_yearly_earnings', 'highest_yearly_earnings',
 'rank', 'Youtuber', 'Title', 'Abbreviation',
 'subscribers_for_last_30_days', 'created_date','country_rank','channel_type_rank'], axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION OF DATA
print()
print(a.info())

print()
print(a.describe())


#SEARCHING FOR DUPLICATED VALUES
print()
print("DUPLICATED VALUES")
print(a.duplicated().sum())

print()
c = a[a.duplicated()]
print()


#SEARCHING FOR MISSING VALUES
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUES FOUND IN DATASET !!")


#SEGREGATING IN CATEGORICAL AND CONTINOUS VALUES
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
print("CONTINOUS VALUES :-\n", con)


#REPLACING THE MISSING VALUES USING MODE, MEDIAN, MEAN
print()
for col in ['category', 'Country', 'channel_type', 'created_month']:
    a[col] = a[col].fillna(a[col].mode()[0])

for col in ['subscribers', 'uploads', 'video_views_for_the_last_30_days', 'created_year','Population', 'Urban_population']:
    a[col] = a[col].fillna(a[col].median())

for col in ['Gross tertiary education enrollment (%)', 'Unemployment rate','Latitude', 'Longitude']:
    a[col] = a[col].fillna(a[col].mean())


#ONE HOT ENCODING
print()
a = pd.get_dummies(a, columns =['category', 'Country', 'channel_type', 'created_month'], drop_first=True)


#FEATURE SELECTION
print()
x = a.drop('video_views_for_the_last_30_days', axis = 1)
y = np.log1p(a['video_views_for_the_last_30_days'])


#SPLIT OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=1)


# CALCULATES THE PROPORTIONAL DISTRIBUTION (RELATIVE FREQUENCY) OF VIDEO VIEWS IN THE LAST 30 DAYS
print()
print(a['video_views_for_the_last_30_days'].value_counts(normalize=True))


# COMPUTES AND SORTS THE CORRELATION OF ALL NUMERIC FEATURES WITH LAST 30 DAYS VIDEO VIEWS
print()
print(a.corr(numeric_only=True)['video_views_for_the_last_30_days'].sort_values(ascending=True))

#STANDARD SCALING
print()
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)


#MODEL BUILDING AND EXECUTION
print()
model = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=10, min_samples_leaf=5, max_features='sqrt', random_state=1, n_jobs=-1)
model.fit(x_train_sc, y_train)

y_pred = model.predict(x_test_sc)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
print("R2 SCORE :-\n", r2_score(y_test, y_pred))
print("MSE :-\n", mean_squared_error(y_test, y_pred))
print("MAE :-\n", mean_absolute_error(y_test, y_pred))


#COMPARISON
print()
print("COMPARISON")
comaprison =  pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comaprison)


#CROSS SCORE VALIDATION
print()
print('CROSS SCORE VALIDATION')
cv = KFold(n_splits =10, shuffle = True, random_state=1)

pipe = Pipeline([
    ('Scaler', StandardScaler()),
    ('model',RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=10, min_samples_leaf=5, max_features='sqrt', random_state=1, n_jobs=-1))
])

scores  =cross_val_score(pipe, x,y, cv = cv, scoring = 'r2')
print("SCORES:-\n", scores)
print("MEAN SCORES :-\n", np.mean(scores))


# DISTRIBUTION OF VIDEO VIEWS IN THE LAST 30 DAYS
plt.figure(figsize=(8,5))
sns.histplot(a['video_views_for_the_last_30_days'], bins=50, kde=True)
plt.title('DISTRIBUTION OF VIDEO VIEWS IN THE LAST 30 DAYS')
plt.xlabel('VIDEO VIEWS (LAST 30 DAYS)')
plt.ylabel('FREQUENCY')
plt.show()


# LOG-TRANSFORMED DISTRIBUTION OF VIDEO VIEWS
plt.figure(figsize=(8,5))
sns.histplot(np.log1p(a['video_views_for_the_last_30_days']), bins=50, kde=True)
plt.title('LOG TRANSFORMED DISTRIBUTION OF VIDEO VIEWS')
plt.xlabel('LOG(VIDEO VIEWS + 1)')
plt.ylabel('FREQUENCY')
plt.show()


# CORRELATION HEATMAP OF NUMERIC FEATURES
plt.figure(figsize=(10,8))
sns.heatmap(a.corr(numeric_only=True), cmap='coolwarm', linewidths=0.5)
plt.title('CORRELATION HEATMAP OF NUMERIC FEATURES')
plt.show()


# TOP 10 FEATURES MOST CORRELATED WITH TARGET
corr_values = a.corr(numeric_only=True)['video_views_for_the_last_30_days'].sort_values()
plt.figure(figsize=(8,5))
corr_values.tail(10).plot(kind='barh')
plt.title('TOP FEATURES CORRELATED WITH VIDEO VIEWS (LAST 30 DAYS)')
plt.xlabel('CORRELATION VALUE')
plt.show()


# ACTUAL VS PREDICTED VALUES SCATTER PLOT
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('ACTUAL VS PREDICTED VALUES')
plt.xlabel('ACTUAL LOG(VIDEO VIEWS)')
plt.ylabel('PREDICTED LOG(VIDEO VIEWS)')
plt.show()


# RESIDUAL DISTRIBUTION
residuals = y_test - y_pred
plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=40, kde=True)
plt.title('RESIDUAL DISTRIBUTION')
plt.xlabel('RESIDUALS')
plt.ylabel('FREQUENCY')
plt.show()


# RESIDUALS VS PREDICTED VALUES
plt.figure(figsize=(6,5))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title('RESIDUALS VS PREDICTED VALUES')
plt.xlabel('PREDICTED LOG(VIDEO VIEWS)')
plt.ylabel('RESIDUALS')
plt.show()


# FEATURE IMPORTANCE FROM RANDOM FOREST
importances = model.feature_importances_
feature_names = x.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(8,6))
feat_imp.head(15).plot(kind='barh')
plt.title('TOP FEATURE IMPORTANCE FROM RANDOM FOREST')
plt.xlabel('IMPORTANCE SCORE')
plt.gca().invert_yaxis()
plt.show()


# CROSS VALIDATION R2 SCORE DISTRIBUTION
plt.figure(figsize=(8,5))
sns.boxplot(scores)
plt.title('CROSS VALIDATION R2 SCORE DISTRIBUTION')
plt.xlabel('R2 SCORE')
plt.show()
