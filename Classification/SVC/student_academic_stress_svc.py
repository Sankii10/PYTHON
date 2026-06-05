# THE FOLLOWING CODE IS EXECUTED BY :- MR. SANKET GAIKWAD
# ALGORITHM USED :- SUPPORT VECTOR MACHINE ( SVC)
# SOURCE :- KAGGLE ( https://www.kaggle.com/datasets/poushal02/student-academic-stress-real-world-dataset )


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\academy.csv")
print(a.columns.tolist())


#DROPING THE UNWANTED FEATURE
print()
a = a.drop('Timestamp', axis = 1)


#INFORMATION ABOUT DATASET 
print()
print(a.info())


#DESCRIBING THE DATASET 
print()
print(a.describe())


#TOTAL NUMBER OF DUPLICATES
print()
print("TOTAL SUM OF DUPLICATES")
print(a.duplicated().sum())

#DUPLICATED VALUES ROW WISE
print()
c = a[a.duplicated()]
print(c)


#DROPPING THE DUPLICATES
print(a.drop_duplicates(inplace=True))


#SEARCHING FOR THE MISSING VALUES
print()
missing_values = a[a.isna().any(axis = 1)]

if not missing_values.empty:
    print(missing_values)
else:
    print("NO SUCH MISSING VALUES FOUND IN DATASET !!!")


#CONVERTING CATEGORICAL VALUES INTO DUMMY VALUES
print()
a = pd.get_dummies(a, columns = [ 'Your Academic Stage','Study Environment', 'What coping strategy you use as a student?', 'Do you have any bad habits like smoking, drinking on a daily basis?'], drop_first=True)


#FEATURE SELECTION
print()
x = a.drop('Rate your academic stress index ', axis = 1)
y = a['Rate your academic stress index ']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled  =sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components = 0.95)
x_pca = pca.fit_transform(x_scaled)


#SPLITING OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state = 1)

print()
smote = SMOTE(random_state= 1, k_neighbors=3)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model = SVC(kernel = 'rbf', class_weight='balanced')
model.fit(x_train_res, y_train_res)

print()
y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

print()
print("ACCURACY SCORE:-\n", acc)
print("CONFUSION MATRIX :-\n", cm)
print("CLASSIFICATION REPORT :-\n", clr)


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores  =cross_val_score(model,x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALISATION

#Class Distribution (Before & After SMOTE)
# Before SMOTE
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Class Distribution Before SMOTE")
plt.show()

# After SMOTE
plt.figure(figsize=(6,4))
sns.countplot(x=y_train_res)
plt.title("Class Distribution After SMOTE")
plt.show()


#PCA Explained Variance Ratio
plt.figure(figsize=(6,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Explained Variance Ratio")
plt.grid()
plt.show()


#Confusion Matrix (Heatmap)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


#Cross-Validation Score Distribution
plt.figure(figsize=(6,4))
plt.plot(range(1, len(scores)+1), scores, marker='o')
plt.axhline(np.mean(scores), color='red', linestyle='--', label=f"Mean = {np.mean(scores):.2f}")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Cross Validation Scores")
plt.legend()
plt.show()


#Comparison: Actual vs Predicted
plt.figure(figsize=(8,4))
plt.scatter(range(len(y_test)), y_test, label="Actual", alpha=0.7)
plt.scatter(range(len(y_pred)), y_pred, label="Predicted", alpha=0.7)
plt.legend()
plt.title("Actual vs Predicted Values")
plt.show()


