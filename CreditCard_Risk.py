"""
THE FOLLOWING CODE IS WRITTEN BY :-
NAME :- SANKET GAIKWAD
DATASET :- CREDIT RISK DATASET(SOURCE :- KAGGLE)

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


a = pd.read_csv("C:\\Users\\anujg\\OneDrive\\Desktop\\MY DATA\\DATASET\\credit_risk_dataset.csv")
print(a.columns.tolist())

print()
a = a.drop('loan_percent_income', axis = 1)


#DATA PREPROCESSING
print()
print(a.info())

print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#HANDLING MISSING VALUE
print()
for col in ['person_emp_length','loan_int_rate']:
    a[col]= a[col].fillna(a[col].median())

print()
le = LabelEncoder()
for col in ['person_home_ownership', 'loan_intent','loan_grade', 'cb_person_default_on_file']:
    a[col] = le.fit_transform(a[col])

print()
x = a.drop('loan_status', axis = 1)
y = a['loan_status']

print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components = 0.95)
x_pca = pca.fit_transform(x_scaled)


#FEATURE SELECTION
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state = 1)

#SMOTE 
print()
smote = SMOTE(random_state = 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

#MODEL BUILDING
print()
model = SVC(kernel= 'rbf', class_weight= 'balanced')
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print(y_pred)

print()
print("EVALUATION")
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

print()
print("ACCURACY SCORE:-\n", acc)
print("CONFUSION MATRIX:-\n", cm)
print("CLASSIFICATION REPORT :-\n", clr)

print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE EVALUATION")
scores = cross_val_score(model, x_pca, y, cv = 10)
print("THE SCORES:-\n", scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#VISUALIZATION
# Distribution of Target Variable (Class Imbalance Check)
sns.countplot(x='loan_status', data=a)
plt.title('Distribution of Loan Status Before SMOTE')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.show()


#PCA Explained Variance Ratio Plot
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()


#Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Comparison of Actual vs Predicted Values
sample_comparison = comparison.sample(50).reset_index(drop=True)
plt.figure(figsize=(10, 4))
plt.plot(sample_comparison['Actual'], label='Actual', marker='o')
plt.plot(sample_comparison['Predict'], label='Predicted', marker='x')
plt.title('Sample Comparison of Actual vs Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Loan Status')
plt.legend()
plt.show()
