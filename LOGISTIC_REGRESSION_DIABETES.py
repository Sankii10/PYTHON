#THE FOLLOWING CODE IS EXECUTED BY :-
# SANKET GAIKWAD
# SOURCE :- KAGGLE (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

a = pd.read_csv("C:\\Users\\anujg\\OneDrive\\Desktop\\MY DATA\\DATASET\\diabetes.csv")
print(a.columns.tolist())


print()
print(a.info())

# MISSING VALUES DETECTION
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#FEATURE SELECTION
print()
x = a.drop('Outcome', axis = 1)
y = a['Outcome']


#STANDARD SCALER
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

#PRINCIPAL COMPONENT ANALYSIS
print()
pca =PCA(n_components = 0.95)
x_pca =pca.fit_transform(x_scaled)

#SPLIT TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state = 1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model = LogisticRegression()
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print(y_pred)

print()
print("EVALUATION")
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test,y_pred)
clr = classification_report(y_test, y_pred)

print("ACCURACY SCORE:-\n", acc)
print("CONFUSION MATRIX:-\n", cm)
print("CLASSSIFICATION REPORT :-\n", clr)

print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print("CROSS SCORE VALIDATION")
scores = cross_val_score(model, x_pca, y, cv = 10)
print("THE SCORES ARE:-\n", scores)
print("THE MEAN SCORE IS :-\n", np.mean(scores))


#DATA VISUALIZATION
#Correlation Heatmap (before PCA)
plt.figure(figsize=(10, 8))
sns.heatmap(a.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Diabetes Dataset")
plt.show()


# PCA Explained Variance Plot
pca_full = PCA()
pca_full.fit(x_scaled)
explained_variance = pca_full.explained_variance_ratio_

plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(explained_variance), marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Explained Variance')
plt.grid(True)
plt.show()


#Scatter Plot (First 2 Principal Components)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=y, palette='Set1')
plt.title("PCA - First Two Components")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title='Outcome')
plt.grid(True)
plt.show()


#Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
