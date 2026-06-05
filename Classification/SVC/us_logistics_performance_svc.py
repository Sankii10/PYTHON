#THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- SVC
#SOURCE :- https://www.kaggle.com/datasets/shahriarkabir/us-logistics-performance-dataset


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from imblearn.combine import SMOTETomek

# READ DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\logistic.csv")
print(a.columns.tolist())

print()
# DROP UNNECESSARY COLUMNS
a = a.drop(['Shipment_ID', 'Shipment_Date', 'Delivery_Date'], axis=1)

print()
print(a.info())

print()
print(a.describe())

print()
print("TOTAL NUMBER OF DUPLICATES")
print(a.duplicated().sum())

print()
c = a[a.duplicated()]
print(c)

print()
b = a.isnull().sum()
print(b)

# CHECK IF ANY MISSING VALUES IN ROWS
missing_value = a[a.isna().any(axis=1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")

print()
# HANDLE MISSING VALUES IN COST COLUMN USING MEAN IMPUTATION
a['Cost'] = a['Cost'].fillna(a['Cost'].mean())

print()
# ONE HOT ENCODING FOR CATEGORICAL FEATURES
a = pd.get_dummies(a, columns=['Origin_Warehouse', 'Destination', 'Carrier'], drop_first=True)

# LABEL ENCODING FOR TARGET COLUMN
le = LabelEncoder()
a['Status'] = le.fit_transform(a['Status'])

print()
# SPLIT INTO FEATURES AND TARGET
x = a.drop('Status', axis=1)
y = a['Status']

print()
# FEATURE SCALING
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

print()
# PCA TO REDUCE DIMENSIONS (RETAIN 99% VARIANCE FOR BETTER PERFORMANCE)
pca = PCA(n_components=0.99)
x_pca = pca.fit_transform(x_scaled)

print()
# TRAIN TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=1)

print()
# USE SMOTETomek TO HANDLE CLASS IMBALANCE BETTER THAN SMOTE
smote_tomek = SMOTETomek(random_state=1)
x_train_res, y_train_res = smote_tomek.fit_resample(x_train, y_train)

print()
# GRID SEARCH FOR BEST HYPERPARAMETERS OF SVC
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf', 'poly']
}

grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=5, scoring='accuracy')
grid.fit(x_train_res, y_train_res)

print("BEST PARAMETERS FOUND:", grid.best_params_)

# USE BEST MODEL FROM GRID SEARCH
model = grid.best_estimator_
model.fit(x_train_res, y_train_res)

# PREDICTION
y_pred = model.predict(x_test)
print(y_pred)

print()
print("EVALUATION")
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

print()
print("ACCURACY SCORE:\n", acc)
print("MACRO F1-SCORE:\n", f1_score(y_test, y_pred, average='macro'))
print("CONFUSION MATRIX:\n", cm)
print("CLASSIFICATION REPORT:\n", clr)

# CONFUSION MATRIX HEATMAP
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("CONFUSION MATRIX")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual': y_test, 'Predict': y_pred})
print(comparison)

print()
print("CROSS VALIDATION SCORE")
scores = cross_val_score(model, x_pca, y, cv=10)
print(scores)
print("THE MEAN SCORE:\n", np.mean(scores))
