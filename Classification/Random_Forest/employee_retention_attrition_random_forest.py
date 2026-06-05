#THE CODE IS EXECUTED BY :- MR.SANKET TANAJI GAIKWAD
#ALGORITHM USED :- RANDOM FORWST CLASSIFIER
#SOURCE :- https://www.kaggle.com/datasets/ajinkyachintawar/employee-attrition-and-retention-analytics-dataset


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


#UPLOADING THE DATA 
a = pd.read_csv("C:\\Users\\ASUS\\Downloads\\HR-Employee-Attrition.csv")
print(a.columns.tolist())


# REMOVED NON-NUMERIC AGE VALUES AND CONVERTED AGE COLUMN TO INTEGER TYPE
print()
a = a[pd.to_numeric(a['Age'], errors = 'coerce').notnull()]

print()
a['Age'] = a['Age'].astype(int)


#DROPING THE UNWANTED FEATURES
print()
a = a.drop(['EmployeeCount', 'EmployeeNumber','Over18','StandardHours'], axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION OF DATA
print()
print(a.info())

print(a.describe())


#SEARCHING FOR TOTAL NUMBER OF DUPLICATED VALUES
print()
print(a.duplicated().sum())


#DISPLAYING DUPLICATED VALUES ROWS
print()
c=a[a.duplicated()]
print(c)


#SEARCHING FOR TOTAL NUMBER OF MISSING VALUES
print()
b = a.isnull().sum()
print(b)


#DISPLAYING OF MISSING VALUES 
print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATSET !!!")


#SEGREGATION OF CATEGORICAL AND CONTINOUS VALUES
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


#LABEL ENCODING FOR TARGET VARIABLE
print()
le = LabelEncoder()
a['Attrition'] = le.fit_transform(a['Attrition'])


#ONE HOT ENCODING
print()
a = pd.get_dummies(a, columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime'], drop_first=True)


# CHECKED CLASS DISTRIBUTION OF ATTRITION BY CALCULATING NORMALIZED PROPORTIONS
print()
print(a['Attrition'].value_counts(normalize=True))


# COMPUTED AND SORTED CORRELATION OF NUMERICAL FEATURES WITH ATTRITION TARGET VARIABLE
print()
print(a.corr(numeric_only=True)['Attrition'].sort_values(ascending=True))


#FEATURE SELECTION
print()
x = a.drop('Attrition', axis = 1)
y = a["Attrition"]


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca =PCA(n_components=0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLITING OF DATA AS TRAIN AND TEST
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=  0.2, random_state= 1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING AND IMPLEMENTATION
print()
model = RandomForestClassifier(n_estimators=600, max_depth=10, min_samples_split=8, min_samples_leaf=3, max_features='sqrt', bootstrap=True, random_state=1, n_jobs=-1)
model.fit(x_train_res, y_train_res)

y_pred_prob = model.predict_proba(x_test)[:,1]
y_pred = (y_pred_prob > 0.40).astype(int)
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

print()
print()
print("CROSS SCORE VALIDATION (SMOTE INSIDE FOLDS)")
kf = KFold(n_splits=10, shuffle=True, random_state=1)
pipeline = Pipeline([('smote', SMOTE(random_state=1)),('rf', model)])
scores = cross_val_score(pipeline, x_pca, y, cv=kf, scoring='f1')
print(scores)
print("MEAN SCORES :-\n", np.mean(scores))


#DATA VISUALISATION

# 1. ATTRITION CLASS DISTRIBUTION
plt.figure()
sns.countplot(x=y)
plt.title("ATTRITION CLASS DISTRIBUTION")
plt.xlabel("ATTRITION (0 = NO, 1 = YES)")
plt.ylabel("EMPLOYEE COUNT")
plt.show()


# 2. CORRELATION HEATMAP (TOP NUMERICAL FEATURES)
plt.figure(figsize=(10, 8))
corr = a.corr(numeric_only=True)
sns.heatmap(corr, cmap='coolwarm', linewidths=0.5)
plt.title("CORRELATION HEATMAP OF NUMERICAL FEATURES")
plt.show()


# 3. CONFUSION MATRIX HEATMAP
plt.figure()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("CONFUSION MATRIX - RANDOM FOREST")
plt.xlabel("PREDICTED LABEL")
plt.ylabel("ACTUAL LABEL")
plt.show()


