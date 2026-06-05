#THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM :- LOGISTIC REGRESSION
#SOURCE :- Shopper's Behavior and Revenue( https://www.kaggle.com/datasets/subhajournal/shoppers-behavior-and-revenue )



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


#INPORTING DATA FILE
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\Shop.csv")
print(a.columns.tolist())


#DESCRIBING THE DATA
print()
print(a.describe())


#INFORMATION ABOUT DATA
print()
print(a.info())


#TO CHECK THE MISSING VALUES
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
le  = LabelEncoder()
for col in ['Month', 'VisitorType', 'Weekend', 'Revenue']:
    a[col] = le.fit_transform(a[col])


#FEATURE SELECTION
print()
x = a.drop('Revenue', axis = 1)
y = a['Revenue']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLITING OF DATA IN TRAIN AND TEST
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state = 1)

print()
smote = SMOTE(random_state=1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model  =LogisticRegression()
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
scores  =cross_val_score(model,x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALISATION

#Class Distribution Before and After SMOTE
# Before SMOTE
plt.figure(figsize=(6,4))
sns.countplot(x=y_train)
plt.title("Class Distribution Before SMOTE")
plt.xlabel("Revenue (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# After SMOTE
plt.figure(figsize=(6,4))
sns.countplot(x=y_train_res)
plt.title("Class Distribution After SMOTE")
plt.xlabel("Revenue (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()


#Explained Variance from PCA
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()


#Confusion Matrix (Heatmap)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Purchase','Purchase'], yticklabels=['No Purchase','Purchase'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Cross-validation Scores Boxplot
plt.figure(figsize=(6,4))
sns.boxplot(data=scores, orient='h', color='skyblue')
plt.title("Cross-Validation Accuracy Scores")
plt.xlabel("Accuracy")
plt.show()




