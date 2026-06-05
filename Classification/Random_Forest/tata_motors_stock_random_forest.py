#THE FOLLOWING CODE IS EXECUTED BY :- MR. SANKET GAIKWAD
#ALGORITHM USED :- RANDOM FOREST CLASSIFIER
#SOURCE :- https://www.kaggle.com/datasets/jatinkalra17/tata-motors-stock-details1995-2025



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# LOAD DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\tatamotors.csv")
print(a.columns.tolist())

a['Date'] = pd.to_datetime(a['Date'], dayfirst=True, errors='coerce')
a['year'] = a['Date'].dt.year
a['month'] = a['Date'].dt.month

# TARGET VARIABLE
a['dailyreturn'] = a['Daily_Return_%'].apply(lambda x: "Up" if x > 0 else "Down")

a = a.drop(['Sr.No', 'Date', 'Symbol', 'PrevClose', 'Trades', 'Daily_Return_%'], axis=1)

print(a.info())
print(a.describe())

print("TOTAL SUM OF DUPLICATED VALUES")
print(a.duplicated().sum())

c = a[a.duplicated()]
print(c)

b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis=1)]
if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET")

# HANDLE MISSING VALUES
for col in ['MA_20', 'MA_50']:
    a[col] = a[col].fillna(a[col].mean())

for col in ['year', 'month']:
    a[col] = a[col].fillna(a[col].mode()[0])

# SPLIT CATEGORICAL / NUMERIC
cat = []
con = []
for i in a.columns:
    if a[i].dtype == 'object':
        cat.append(i)
    else:
        con.append(i)

print("CATEGORICAL VALUE :-\n", cat)
print("CONTINUOUS VALUE :-\n", con)

# LABEL ENCODING
le = LabelEncoder()
a['dailyreturn'] = le.fit_transform(a['dailyreturn'])

# FEATURES & TARGET
x = a.drop('dailyreturn', axis=1)
y = a['dailyreturn']

# SCALING
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

# TRAIN/TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=1, stratify=y)

# SMOTE BALANCING
smote = SMOTE(random_state=1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

# RANDOM FOREST MODEL
model = RandomForestClassifier(n_estimators=300, criterion='entropy', max_depth=None, min_samples_split=5, class_weight='balanced', random_state=1)
model.fit(x_train_res, y_train_res)

# PREDICTIONS
y_pred = model.predict(x_test)
print(y_pred)

# EVALUATION
print("EVALUATION")
print("ACCURACY SCORE:-\n", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX :-\n", confusion_matrix(y_test, y_pred))
print("CLASSIFICATION REPORT :-\n", classification_report(y_test, y_pred))

# COMPARISON
print("COMPARISON")
comparison = pd.DataFrame({'Actual': y_test, 'Predict': y_pred})
print(comparison)

# CROSS-VALIDATION
print("CROSS SCORE VALIDATION")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(model, x_scaled, y, cv=cv)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALISATION

# 1. CORRELATION HEATMAP
plt.figure(figsize=(10,8))
sns.heatmap(pd.DataFrame(x_scaled, columns=x.columns).corr(), cmap="coolwarm", annot=False)
plt.title("CORRELATION HEATMAP")
plt.show()

# 2. TARGET VARIABLE DISTRIBUTION
plt.figure(figsize=(6,4))
sns.countplot(x=y, palette="Set2")
plt.title("DISTRIBUTION OF DAILY RETURN (UP/DOWN)")
plt.xticks([0,1], ["Down","Up"])
plt.show()

# 3. FEATURE IMPORTANCE (RANDOM FOREST)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=np.array(x.columns)[indices], palette="viridis")
plt.title("FEATURE IMPORTANCE (RANDOM FOREST)")
plt.show()

# 4. CONFUSION MATRIX HEATMAP
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Down","Up"], yticklabels=["Down","Up"])
plt.title("CONFUSION MATRIX")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 5. CROSS-VALIDATION SCORE DISTRIBUTION
plt.figure(figsize=(8,4))
sns.boxplot(x=scores, color="lightblue")
plt.title("CROSS-VALIDATION SCORE DISTRIBUTION")
plt.xlabel("Accuracy")
plt.show()



