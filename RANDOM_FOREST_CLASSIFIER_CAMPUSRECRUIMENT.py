# THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- RANDOM FOREST CLASSIFIER
#SOURCE :- Campus Recruitment ( https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement )



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\placement.csv")
print(a.columns.tolist())


#DROPPING THE UNNECESSARY FEATURE
print()
a = a.drop('sl_no', axis = 1)


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


#HANDLING MISSING VALUES
print()
a['salary'] = a['salary'].fillna(a['salary'].median())


#LABEL ENCODING
print()
le = LabelEncoder()
for col in ['gender', 'hsc_b', 'ssc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status']:
    a[col] = le.fit_transform(a[col])


#FEATURE SELECTIONS
print()
x = a.drop('status', axis = 1)
y = a['status']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLIT OF TRAIN AND TEST DATASET 
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y,test_size=0.2, random_state= 1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model = RandomForestClassifier(n_estimators=150, criterion='entropy', class_weight='balanced', max_depth=4)
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
print("CONFUSION MATRIX :-\n", cm)
print("CLASSIFICATION REPORT :-\n", clr)


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model,x_pca, y, cv  =10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALISATION

#Correlation Heatmap (to see feature relationships before PCA)
plt.figure(figsize=(12,6))
sns.heatmap(a.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.show()


#Class Distribution Before & After SMOTE
fig, ax = plt.subplots(1,2, figsize=(12,5))

sns.countplot(x=y, ax=ax[0], palette="Set2")
ax[0].set_title("Original Class Distribution")

sns.countplot(x=y_train_res, ax=ax[1], palette="Set1")
ax[1].set_title("Class Distribution After SMOTE")

plt.show()


#Explained Variance Ratio (PCA)
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance by PCA Components")
plt.grid()
plt.show()


#Confusion Matrix Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Placed","Placed"], yticklabels=["Not Placed","Placed"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#Feature Importance (from Random Forest)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.bar(range(x_pca.shape[1]), importances[indices])
plt.xlabel("PCA Components")
plt.ylabel("Importance Score")
plt.title("Feature Importance from Random Forest")
plt.show()



