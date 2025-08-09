#THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- KNN CLASIFIER ALGORITHM
#SOURCE :- Adult Census Income( https://www.kaggle.com/datasets/uciml/adult-census-income )


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


#IMPORT THE DATA 
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\adult.csv")
print(a.columns.tolist())


#INFORMATION OF DATA 
print()
print(a.info())


#REPLACING '?' BY NAN IN DATASET BECAUSE WE HAVE '?' IN DATASET AND IT SHOWS NO MISSING VALUE BUT IF WE REPLACE '?' BY NAN WE GET TO KNOW THE MISSING VALUES
print()
a.replace('?', np.nan, inplace = True)


#FINDING MISSING VALUES
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#HANDLING MISSING VALUES
print()
for col in ['workclass','occupation','native.country']:
    a[col] = a[col].fillna(a[col].mode()[0])


#LABEL ENCODING
print()
le = LabelEncoder()
for col in ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'native.country', 'sex', 'income']:
    a[col] = le.fit_transform(a[col])


#FEATURE SELECTION
print()
x = a.drop('income', axis = 1)
y = a['income']


#STANDARD SCALER
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca = pca.fit_transform(x_scaled)


#SPLITING OF DATA IN TRAIN AND TEST
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state= 1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model = KNeighborsClassifier(n_neighbors=4)
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
print("ACCURACY SCORE:-\n", acc)
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

#Class Distribution Before and After SMOTE
# Before SMOTE
plt.figure(figsize=(6,4))
sns.countplot(x=y, palette="viridis")
plt.title("Class Distribution Before SMOTE")
plt.show()

# After SMOTE
plt.figure(figsize=(6,4))
sns.countplot(x=y_train_res, palette="viridis")
plt.title("Class Distribution After SMOTE")
plt.show()


# PCA Explained Variance Ratio
pca_full = PCA().fit(x_scaled)
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Explained Variance Ratio")
plt.grid(True)
plt.show()


# Confusion Matrix Heatmap
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


#Cross-Validation Score Distribution
plt.figure(figsize=(6,4))
plt.plot(range(1, len(scores)+1), scores, marker='o')
plt.axhline(np.mean(scores), color='r', linestyle='--', label=f"Mean: {np.mean(scores):.3f}")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Cross-Validation Scores")
plt.legend()
plt.grid(True)
plt.show()


