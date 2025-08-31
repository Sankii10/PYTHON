#THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALFORITHM USD :- Logistic Regression, Random Forest, and XGBoost
#SOURCE :- https://www.kaggle.com/datasets/mohamadsallah5/english-premier-league-stats20212024



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# LOAD DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\mydata.csv")
print(a.columns.tolist())

# CONVERT DATE
a['date'] = pd.to_datetime(a['date'], dayfirst=True, errors='coerce')
a['year'] = a['date'].dt.year
a['month'] = a['date'].dt.month

# DROP UNNECESSARY COLUMNS
a = a.drop(['clock', 'stadium', 'attendance', 'Home Team', 'Away Team', 'Goals Home', 'Away Goals', 'links', 'date'], axis=1)

print()
print(a.info())
print()
print(a.describe())

# CHECK DUPLICATES
print("TOTAL DUPLICATES IN CODE:", a.duplicated().sum())
if a.duplicated().sum() > 0:
    print(a[a.duplicated()])

# CHECK MISSING VALUES
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis=1)]
if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")

print()
# HANDLE MISSING VALUES FOR YEAR & MONTH
for col in ['year', 'month']:
    a[col] = a[col].fillna(a[col].mode()[0])

# MAP TARGET VARIABLE (h = HOME WIN, d = DRAW, a = AWAY WIN)
a['class'] = a['class'].map({'h': 0, 'd': 1, 'a': 2})

# SPLIT X AND Y
x = a.drop('class', axis=1)
y = a['class']

# SCALE FEATURES
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

# TRAIN-TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42, stratify=y)

# HANDLE IMBALANCE USING SMOTE
smote = SMOTE(random_state=42)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

# 1. LOGISTIC REGRESSION MODEL

log_reg = LogisticRegression(
    multi_class='multinomial', solver='lbfgs', class_weight='balanced',
    max_iter=1000, penalty='l2', C=1.0, n_jobs=-1, random_state=42
)
log_reg.fit(x_train_res, y_train_res)
y_pred_lr = log_reg.predict(x_test)

print("\n===== LOGISTIC REGRESSION =====")
print("ACCURACY:", accuracy_score(y_test, y_pred_lr))
print("CONFUSION MATRIX:\n", confusion_matrix(y_test, y_pred_lr))
print("CLASSIFICATION REPORT:\n", classification_report(y_test, y_pred_lr))


# 2. RANDOM FOREST MODEL

model = RandomForestClassifier(
    n_estimators=500, max_depth=None, max_features='sqrt', class_weight='balanced_subsample',
    criterion='gini', min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1
)
# 500 TREES, DEEP TREES, FEATURE SUBSAMPLING, BALANCED CLASSES, GINI CRITERION
model.fit(x_train_res, y_train_res)
y_pred_rf = model.predict(x_test)

print("\n===== RANDOM FOREST =====")
print("ACCURACY:", accuracy_score(y_test, y_pred_rf))
print("CONFUSION MATRIX:\n", confusion_matrix(y_test, y_pred_rf))
print("CLASSIFICATION REPORT:\n", classification_report(y_test, y_pred_rf))

# FEATURE IMPORTANCE (RANDOM FOREST)
feature_importances = pd.Series(model.feature_importances_, index=x.columns)
feature_importances.sort_values(ascending=False).plot(kind='bar', figsize=(12, 5), title="Random Forest Feature Importance")
plt.show()


# 3. XGBOOST MODEL

xgb = XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
    objective='multi:softmax', num_class=3, random_state=42, use_label_encoder=False, eval_metric='mlogloss'
)
xgb.fit(x_train_res, y_train_res)
y_pred_xgb = xgb.predict(x_test)

print("\n===== XGBOOST =====")
print("ACCURACY:", accuracy_score(y_test, y_pred_xgb))
print("CONFUSION MATRIX:\n", confusion_matrix(y_test, y_pred_xgb))
print("CLASSIFICATION REPORT:\n", classification_report(y_test, y_pred_xgb))

# FEATURE IMPORTANCE (XGBOOST)
xgb_importances = pd.Series(xgb.feature_importances_, index=x.columns)
xgb_importances.sort_values(ascending=False).plot(kind='bar', figsize=(12, 5), title="XGBoost Feature Importance")
plt.show()


# CROSS-VALIDATION SCORES

print("\n===== STRATIFIED K-FOLD ACCURACY =====")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("Logistic Regression CV:", cross_val_score(log_reg, x_scaled, y, cv=cv, scoring='accuracy').mean())
print("Random Forest CV:", cross_val_score(model, x_scaled, y, cv=cv, scoring='accuracy').mean())
print("XGBoost CV:", cross_val_score(xgb, x_scaled, y, cv=cv, scoring='accuracy').mean())
