#THE FOLLOWING CODE IS EXECUTED BY :-MR.SANKET GAIKWAD
#ALGORITHM USED :- SUPPORT VECTOR CLASSIFIER
#SOURCE :- KAGGLE(https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
#DATASET TOPIC :- IBM HR Analytics Employee Attrition & Performance



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


#IMPORTING DATA FROM CSV FILE
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\ibmemp.csv")
print(a.columns.tolist())


#INFORMATION ABOUT DATA
print()
print(a.info())


#SEARCHING FOR MISSING VALUES
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#LABEL ENCODING 
print()
le = LabelEncoder()
for col in ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'Attrition']:
    a[col]  = le.fit_transform(a[col])


#FEATURE SELECTION
print()
x = a.drop('Attrition', axis = 1)
y = a['Attrition']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca = pca.fit_transform(x_scaled)


#SPLIT OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size= 0.2, random_state= 1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model = SVC(kernel = 'rbf', class_weight='balanced')
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

print()
print("ACCURACY SCORE:-\n", cm)
print("CONFUSION MATRIX:-\n", cm)
print("CLASSIFICATION REPORT :-\n", clr)


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model,x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALIZATION

#Explained Variance from PCA
# PCA Variance Explained
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='purple')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.tight_layout()
plt.show()


#Confusion Matrix Heatmap
# Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Attrition', 'Attrition'], yticklabels=['No Attrition', 'Attrition'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


#Actual vs Predicted Bar Plot
# Bar plot for comparison
comparison['Match'] = comparison['Actual'] == comparison['Predict']
comparison['Index'] = range(len(comparison))

plt.figure(figsize=(12, 5))
sns.scatterplot(x='Index', y='Actual', data=comparison, label='Actual', marker='o')
sns.scatterplot(x='Index', y='Predict', data=comparison, label='Predicted', marker='x')
plt.title('Actual vs Predicted Attrition')
plt.legend()
plt.tight_layout()
plt.show()


#Cross-Validation Scores Boxplot
# Cross Validation Score Distribution
plt.figure(figsize=(8, 5))
sns.boxplot(scores, color='teal')
plt.title("Cross-Validation Scores Distribution")
plt.xlabel("Accuracy Score")
plt.grid(True)
plt.tight_layout()
plt.show()
