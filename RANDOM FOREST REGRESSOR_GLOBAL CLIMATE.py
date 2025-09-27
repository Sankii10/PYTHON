#THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- RANDOM FOREST REGRESSOR
#SOURCE :- https://www.kaggle.com/datasets/uom190346a/global-climate-events-and-economic-impact-dataset




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\global.csv")
print(a.columns.tolist())


# CONVERTED 'date' COLUMN TO DATETIME FORMAT AND EXTRACTED THE MONTH INTO A NEW COLUMN 'month'
print()
a['date'] = pd.to_datetime(a['date'], dayfirst=True, errors='coerce')

a['month'] = a['date'].dt.month


#DROPPING ALL UNREQUIRED FEATURES
print()
a = a.drop(['event_id', 'date', 'latitude', 'longitude', 'impact_per_capita', 'aid_percentage'], axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION ABOUT THE DATA
print()
print(a.info())

print()
print(a.describe())


#TOTAL SUM OF DUPLICATED VALUES
print()
print("TOTAL SUM OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())


#DISPLAYING THE DUPLICATED ROWS VALUES
print()
c = a[a.duplicated()]
print(c)


#FINDING THE MISSING VALUES IN DATA
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#HANDLING THE MISSING VALUES
print()
a['month'] = a['month'].fillna(a['month'].mode()[0])


#SEGREGRATING THE CATEGORICAL AND CONTINOUS VALUES 
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


#ONE HOT ENCODING
print()
a = pd.get_dummies(a, columns = ['country', 'event_type'], drop_first= True)


# PRINTED THE NORMALIZED VALUE COUNTS OF 'total_casualties' TO SEE THE PROPORTION OF EACH UNIQUE VALUE
print()
print(a['total_casualties'].value_counts(normalize= True))


# CALCULATED AND SORTED THE CORRELATION OF ALL NUMERIC COLUMNS WITH 'total_casualties' IN DESCENDING ORDER
print()
print(a.corr(numeric_only= True)['total_casualties'].sort_values(ascending = False))


#FEATURE SELECTION
print()
x = a.drop('total_casualties', axis = 1)
y = a['total_casualties']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLIT THE TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state = 1)


#MODEL BUILDING AND IMPLEMENTING
print()
model = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', bootstrap=True, random_state=42, n_jobs=-1)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
print("MEAN SQUARED ERROR:-\n", mean_squared_error(y_test, y_pred))
print("MEAN ABSOLUTE ERROR:-\n", mean_absolute_error(y_test, y_pred))
print("r2_SCORE:-\n", r2_score(y_test, y_pred))


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model,x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES :-\n", np.mean(scores))


#DATA VISUALISATION

# 1. DISTRIBUTION OF NUMERIC FEATURES
plt.figure(figsize=(15,10))
a[con].hist(bins=20, edgecolor='black', layout=(4,3))
plt.suptitle("DISTRIBUTION OF NUMERIC FEATURES")
plt.show()

# 2. DISTRIBUTION OF TARGET VARIABLE 'total_casualties'
plt.figure(figsize=(8,5))
sns.histplot(a['total_casualties'], bins=50, kde=True, color='orange')
plt.title("DISTRIBUTION OF TOTAL CASUALTIES")
plt.xlabel("Total Casualties")
plt.ylabel("Frequency")
plt.show()

# 3. HEATMAP OF CORRELATION
plt.figure(figsize=(15,10))
sns.heatmap(a.corr(numeric_only=True), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("CORRELATION HEATMAP")
plt.show()

# 4. TOP 10 FEATURES CORRELATED WITH 'total_casualties'
top_corr = a.corr(numeric_only=True)['total_casualties'].sort_values(ascending=False)[1:11]
plt.figure(figsize=(10,5))
sns.barplot(x=top_corr.values, y=top_corr.index, palette='viridis')
plt.title("TOP 10 FEATURES CORRELATED WITH TOTAL CASUALTIES")
plt.xlabel("Correlation")
plt.ylabel("Feature")
plt.show()

# 5. PCA EXPLAINED VARIANCE
pca_explained = pca.explained_variance_ratio_
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca_explained), marker='o', color='green')
plt.title("CUMULATIVE EXPLAINED VARIANCE BY PCA COMPONENTS")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.show()

# 6. ACTUAL VS PREDICTED TOTAL CASUALTIES
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.6, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title("ACTUAL VS PREDICTED TOTAL CASUALTIES")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# 7. RESIDUAL PLOT
residuals = y_test - y_pred
plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=50, kde=True, color='red')
plt.title("RESIDUALS DISTRIBUTION")
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.show()

# 8. FEATURE IMPORTANCE FROM RANDOM FOREST (ON ORIGINAL FEATURES)
# Extract original feature importance (before PCA)
model_orig = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_split=5,
                                   min_samples_leaf=2, max_features='sqrt', bootstrap=True,
                                   random_state=42, n_jobs=-1)
model_orig.fit(x, y)
importances = pd.Series(model_orig.feature_importances_, index=x.columns).sort_values(ascending=False)

plt.figure(figsize=(12,8))
sns.barplot(x=importances.values[:20], y=importances.index[:20], palette='cool')
plt.title("TOP 20 FEATURE IMPORTANCES FROM RANDOM FOREST")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()