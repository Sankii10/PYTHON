"""
This code has been written by..
Name : SANKET GAIKWAD
DATASET :- WINE QUALITY DATASET(SOURCE :- KAGGLE)
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


#LOADING CSV FILE
a = pd.read_csv("C:\\Users\\anujg\\OneDrive\\Desktop\\MY DATA\\DATASET\\WineQT.csv")
print(a.columns.tolist())

# prints the count of each unique value in the quality column, 
# showing how many wines belong to each quality score.
print(a['quality'].value_counts())

print()
a = a.drop('Id', axis = 1)


#DATA PREPROCESSING
print()
print(a.info())

print()
b = a.isnull().sum()
print(b)

print()
missing_value= a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUES !!!")


# This line creates a new binary column quality_label where wines with quality less than 6 are labeled as 1 (low quality)
#  and those with quality 6 or higher are labeled as 0 (high quality).
print()
a['quality_label'] = a['quality'].apply(lambda x: 1 if x < 6 else 0)

print()
x = a.drop(['quality_label', 'quality'], axis = 1)
y = a['quality_label']

#STANDARD SCALER
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

print()
pca = PCA(n_components=0.95)
x_pca = pca.fit_transform(x_scaled)


#FEATURE SELECTION
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, stratify=y, random_state=1)


#SMOTE
print()
smote = SMOTE(random_state=1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

print()

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 4, 5, None],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced', None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}


#MODEL BUILDING
rf = RandomForestClassifier(random_state=1)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

grid_search.fit(x_train_res, y_train_res)

print("Best parameters found: ", grid_search.best_params_)

best_rf = grid_search.best_estimator_

# Prediction with tuned model
y_pred = best_rf.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

print()
print("ACCURACY SCORE:-\n", acc)
print("CONFUSION MATRIX:-\n", cm)
print("CLASSIFICATION REPORT:-\n", clr)

print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual': y_test, 'Predict': y_pred})
print(comparison)

print()
print("CROSS VALIDATION SCORE")
scores = cross_val_score(best_rf, x_pca, y, cv=10)
print("THE SCORES ARE :-\n", scores)
print("THE MEAN SCORE:-\n", np.mean(scores))


#DATA VISUALIZATION
#Feature Importance Plot
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importances (PCA components)")
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [f"PC{i+1}" for i in indices])
plt.xlabel('PCA Components')
plt.ylabel('Importance')
plt.show()


#Confusion Matrix Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['High Quality (0)', 'Low Quality (1)'], yticklabels=['High Quality (0)', 'Low Quality (1)'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix Heatmap')
plt.show()


#ROC CURVE
from sklearn.metrics import roc_curve, auc

y_prob = best_rf.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
