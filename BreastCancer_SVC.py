"""
THE BELOW CODE IS EXECUTED BY...
NAME :- SANKET GAIKWAD
DATASET :- BREAST CANCER DATASET ( SOURCE - KAGGLE)

"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

a = pd.read_csv("C:\\Users\\anujg\\OneDrive\\Desktop\\MY DATA\\DATASET\\Breast Cancer.csv")
print(a.columns.tolist())

print()
a = a.drop(['id','Unnamed: 32'], axis = 1)

print()
print(a.info())

# DATA PREPROCESSING

print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")

print()
le = LabelEncoder()
a['diagnosis'] = le.fit_transform(a['diagnosis'])


# FEATURE SELECTION
print()
x = a.drop('diagnosis', axis = 1)
y = a['diagnosis']

print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

print()
pca = PCA(n_components = 0.95)
x_pca = pca.fit_transform(x_scaled)


# SPLITING OF DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, stratify=y, random_state = 1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


# BUILDING OF MODEL
print()
model = SVC(kernel='rbf', class_weight='balanced')
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print("THE PREDICTIONS ARE :-\n", y_pred)

print()
print("EVALUATION")
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

print()
print("ACCURACY SCORE:-\n", acc)
print("CONFUSION MATRIX :-\n", cm)
print("CLASSIFICATION REPORT :-\n", clr)

print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS VALIDATION SCORE")
scores = cross_val_score(model, x_pca, y, cv  =10)
print("THE SCORES:-\n", scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


# DATA VISUALIZATION

# Exploratory Data Analysis (EDA) – Count Plot of Diagnosis
sns.countplot(x='diagnosis', data=a)
plt.title('Diagnosis Distribution (0 = Benign, 1 = Malignant)')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.show()


# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(a.corr(), cmap='coolwarm', annot=False)
plt.title('Feature Correlation Heatmap')
plt.show()


# Explained Variance from PCA
pca_components = PCA().fit(x_scaled)
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca_components.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Cumulative Explained Variance')
plt.grid(True)
plt.show()


# Confusion Matrix Visualization
import seaborn as sns
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Comparison Plot: Actual vs Predicted
comparison.reset_index(drop=True).head(30).plot(kind='bar', figsize=(12, 6))
plt.title('Actual vs Predicted (First 30 Observations)')
plt.xlabel('Index')
plt.ylabel('Diagnosis')
plt.legend(['Actual', 'Predicted'])
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

