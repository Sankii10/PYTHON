""""
THIS CODE IS EXECUTED BY
NAME :- SANKET GAIKWAD
DATASET :- TITANIC DATASET (SOURCE:-KAGGLE)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE

a = pd.read_csv("C:\\Users\\anujg\\OneDrive\\Desktop\\MY DATA\\DATASET\\Titanic_Dataset.csv")
print(a.columns.tolist())

print()
a = a.drop(['PassengerId','Name','Ticket', 'Cabin'], axis = 1)

print()
print(a.info())


#DATA PREPROCESSING
print(a['Survived'].value_counts)

print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a[['Age','Embarked']].isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#HANDLING MISSING VALUES
print()
a['Age'] = a['Age'].fillna(a['Age'].median())
a['Embarked'] = a['Embarked'].fillna(a['Embarked'].mode()[0])


#LABEL ENCODING
print()
le = LabelEncoder()
for col in ['Sex','Embarked']:
    a[col] = le.fit_transform(a[col])


#CORRELATION HEATMAP AFTER ENCODING
plt.figure(figsize=(10, 8))
sns.heatmap(a.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()


#FEATURE SELECTION
print()
x = a.drop('Survived', axis = 1)
y = a['Survived']


#STANDARD SCALER
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components = 0.95)
x_pca = pca.fit_transform(x_scaled)

#PCA EXPLAINED VARIANCE RATIO
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle='-', label='0.95 Explained Variance')
plt.legend(loc='lower right')
plt.show()



#SPLITING OF TRAIN & TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, stratify=y, random_state= 1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model = SVC(kernel = 'rbf', class_weight='balanced', probability=True)
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION METRICS 
print()
print("EVALUATION")
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

print()
print("ACCURACY SCORE:-\n", acc)
print("CONFUSION MATRIX:-\n", cm)
print("CLASSIFICCATION REPORT :-\n", clr)


#CONFUSION MATRIX HEATMAP
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS VALIDATION SCORE")
scores = cross_val_score(model, x_pca, y, cv = 10)
print("THE SCORES:-\n", scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALIZATION
#ROC CURVE & AUC
y_prob = model.predict_proba(x_test)[:, 1] # Get probabilities for the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()



