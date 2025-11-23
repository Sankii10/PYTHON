#THE CODE IS EXECUTED BY ME.SANKET GAIKWAD
#ALGORITHM USED :- RANDOM FOREST CLASSIFIER
#SOURCE :- https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE


#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\transactions.csv")
print(a.columns.tolist())


# CONVERTING THE TRANSACTION_TIME COLUMN FROM STRING FORMAT TO PYTHON DATETIME FORMAT WITH UTC TIMEZONE
print()
a['transaction_time'] = pd.to_datetime(a['transaction_time'], utc = True)
a['year'] = a['transaction_time'].dt.year
a['month'] = a['transaction_time'].dt.month


#DROPPING UNREQUIRED FEATURES
print()
a = a.drop(['transaction_id', 'user_id','total_transactions_user', 'avg_amount_user','transaction_time'], axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION OF THE DATA
print()
print(a.info())

print()
print(a.describe())


#FINDING FOR THE TOTAL DUPLICATED VALUES
print()
print("TOTAL SUM OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())

#DISPLAY THE DUPLICATED VALUES ROWS VALUES
print()
c = a[a.duplicated()]
print(c)


#SEARCHING FOR MISSING VALUES IN DATA
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#SEGREGATING THE CATEGORICAL AND CONTINOUS VALUES
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


#ONE HOT ENCODING
print()
a = pd.get_dummies(a, columns =['country', 'bin_country', 'channel', 'merchant_category'], drop_first=True )


#FEATURE SELECTION
print()
x = a.drop('is_fraud', axis = 1)
y =a['is_fraud']

# PRINTING THE PROPORTION (PERCENTAGE) OF EACH CLASS IN THE is_fraud COLUMN USING NORMALIZED VALUE COUNTS
print()
print(a['is_fraud'].value_counts(normalize= True))


# DISPLAYING THE CORRELATION OF ALL NUMERICAL FEATURES WITH THE TARGET VARIABLE is_fraud AND SORTING THEM IN DESCENDING ORDER TO IDENTIFY STRONGEST RELATIONSHIPS
print()
print(a.corr(numeric_only=True)['is_fraud'].sort_values(ascending = False))


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

#SPLITING OF DATA AS TRAIN AND TEST
print()
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state= 1)


print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res  =smote.fit_resample(x_train, y_train)


#MODEL BUILDING AND IMPLEMENTATION
print()
model = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', bootstrap=True, random_state=1)
model.fit(x_train_res, y_train_res)

y_pred_prob = model.predict_proba(x_test)[:,1]
y_pred = (y_pred_prob > 0.5).astype(int)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
print("ACCURACY SCORE:-\n", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX :-\n", confusion_matrix(y_test, y_pred))
print("CLASSIFICATION REPORT :-\n", classification_report(y_test, y_pred))


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
kf = KFold(n_splits = 10, shuffle = True, random_state= 1)
scores  =cross_val_score(model,x_scaled, y, cv = kf, scoring = 'accuracy')
print(scores)
print("MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALISATION

# 1. FRAUD DISTRIBUTION
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Fraud vs Non-Fraud Count")
plt.xlabel("is_fraud")
plt.ylabel("Count")
plt.show()

# 2. CORRELATION HEATMAP
plt.figure(figsize=(12,8))
sns.heatmap(a.corr(numeric_only=True), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()

# 3. FEATURE IMPORTANCE (RANDOM FOREST)
plt.figure(figsize=(14,6))
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = x.columns

plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
plt.title("Feature Importance from Random Forest")
plt.show()

# 4. BOX PLOTS FOR IMPORTANT NUMERIC FEATURES
numeric_cols = x.columns[:10]  # show first 10 numerical features

plt.figure(figsize=(14,10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(5, 2, i)
    sns.boxplot(data=a, x='is_fraud', y=col)
    plt.title(f"{col} vs is_fraud")
plt.tight_layout()
plt.show()

# 5. COUNT PLOTS FOR IMPORTANT CATEGORICAL FEATURES
categorical_cols = ['channel_web', 'channel_app', 'promo_used', 'three_ds_flag']

plt.figure(figsize=(14,10))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(2, 2, i)
    sns.countplot(data=a, x=col, hue='is_fraud')
    plt.title(f"{col} distribution by Fraud")
plt.tight_layout()
plt.show()

# 6. CONFUSION MATRIX HEATMAP
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 7. ROC CURVE
y_pred_prob = model.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_pred_prob):.3f}")
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()