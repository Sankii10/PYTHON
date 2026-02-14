#THE CODE IS EXECUTED BY MR.SANKET GAIKWAD
#ALGORITHM USED :- RANDOM FOREST CLASSIFIER
#SOURCE :- https://www.kaggle.com/datasets/nudratabbas/patient-churn-prediction-dataset-for-healthcare



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.pipeline import Pipeline


#IMPORT THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\Desktop\\My Data\\patient.csv")
print(a.columns.tolist())


#DROPING THE UNREQUIRED VARIABLES
print()
a = a.drop(['PatientID','Last_Interaction_Date'], axis = 1)


#INFORMAION AND STATISTICAL DESCRIPTION OF DATA
print()
print(a.info())

print()
print(a.describe())


#DUPLICATED VALUES
print()
print("DUPLICATED VALUES :-\n")
print(a.duplicated().sum())

print()
c = a[a.duplicated()]
print(c)


#SEARCHING FOR MISSING VALUES
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!")


#SEGREGATION CATEGORICAL AND CONTINOUS VALUES
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


# CHECKING CLASS DISTRIBUTION (IMBALANCE) OF TARGET VARIABLE 'CHURNED' USING NORMALIZED VALUE COUNTS
print()
print(a['Churned'].value_counts(normalize=True))

print()
print(a.corr(numeric_only=True)['Churned'].sort_values(ascending=False))


# CALCULATING CORRELATION OF ALL NUMERICAL FEATURES WITH TARGET VARIABLE 'CHURNED' AND SORTING THEM IN DESCENDING ORDER
print()
a = pd.get_dummies(a, columns = ['Gender', 'State', 'Specialty', 'Insurance_Type'], drop_first=True)


#FEATURE SELECTION
print()
x = a.drop('Churned', axis = 1)
y = a['Churned']


#SPLIT OF DATA AS TRAIN AND TEST
print()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state= 1)


#STANDARD SCALER
print()
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_train_pca = pca.fit_transform(x_train_sc)
x_test_pca =pca.transform(x_test_sc)


#MODEL BUILDING AND IMPLEMENTATION
print()
model = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=8, min_samples_leaf=4, max_features='sqrt', class_weight='balanced', random_state=1, n_jobs=-1)
model.fit(x_train_pca, y_train)

y_pred_prob = model.predict_proba(x_test_pca)[:,1]
y_pred = (y_pred_prob > 0.5).astype(int)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
print("ACCURACY SCORE :-\n", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX :-\n", confusion_matrix(y_test, y_pred))
print("CLASSIFICATION REPORT :-\n", classification_report(y_test, y_pred))


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)


#CROSS SCORE VALIDATION
print()
print("CROSS SCORE VALIDATION")
cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state=1)

pipe = Pipeline([
    ('Standard Scaler', StandardScaler()),
    ('PCA', PCA(n_components=0.95)),
    ('model',RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=8, min_samples_leaf=4, max_features='sqrt', class_weight='balanced', random_state=1, n_jobs=-1))
])

print()
scores = cross_val_score(pipe, x,y, cv = cv, scoring = 'accuracy')
print(scores)
print("MEAN SCORES :-\n", np.mean(scores))


# VISUALIZATION SECTION

plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("CLASS DISTRIBUTION OF TARGET VARIABLE CHURNED")
plt.xlabel("CHURNED")
plt.ylabel("COUNT")
plt.show()


plt.figure(figsize=(12,8))
sns.heatmap(a.corr(numeric_only=True), cmap='coolwarm')
plt.title("CORRELATION HEATMAP OF NUMERICAL FEATURES")
plt.show()


plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("CONFUSION MATRIX HEATMAP")
plt.xlabel("PREDICTED")
plt.ylabel("ACTUAL")
plt.show()


plt.figure(figsize=(10,6))
importances = model.feature_importances_
indices = np.argsort(importances)[-15:]
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [x.columns[i] for i in indices])
plt.title("TOP 15 FEATURE IMPORTANCE FROM RANDOM FOREST")
plt.xlabel("IMPORTANCE SCORE")
plt.show()


from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc_score = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1])
plt.title("ROC CURVE")
plt.xlabel("FALSE POSITIVE RATE")
plt.ylabel("TRUE POSITIVE RATE")
plt.show()

print("ROC AUC SCORE :- ", auc_score)
