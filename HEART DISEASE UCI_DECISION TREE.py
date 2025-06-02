"""
THE FOLLOWING CODE IS EXECUTED BY :-
NAME :- SANKET GAIKWAD
DATASET :- HEART DISEASE UCI 
ALGORITHM :- DECISION TREE CLASSIFIER

"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

a = pd.read_csv("C:\\Users\\anujg\\OneDrive\\Desktop\\MY DATA\\DATASET\\heart_disease_uci.csv")
print(a.columns.tolist())

print()
a = a.drop('id', axis = 1)

# CONVERTED THE TARGET COLUMN 'num' INTO BINARY: 1 FOR PRESENCE AND 0 FOR ABSENCE OF HEART DISEASE
a['num'] = a['num'].apply(lambda x: 1 if x > 0 else 0)


#DATA-PREPROCESSING
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
    print("NO SUCH MISSING VALUE FOUND INTO DATASET !!!")


#HANDLING MISSING VALUES
print()
for col in ['trestbps', 'chol','thalch','oldpeak']:
    a[col] = a[col].fillna(a[col].median())

for col in ['fbs', 'restecg', 'exang','slope', 'ca', 'thal']:
    a[col] = a[col].fillna(a[col].mode()[0])


#LABEL ENCODING
print()
le = LabelEncoder()
for col in ['sex', 'dataset', 'cp','fbs', 'restecg','exang','slope','thal']:
    a[col] = le.fit_transform(a[col])

#FEATURE SELECTION
print()
x = a.drop('num', axis = 1)
y = a['num']

#STANDARD-SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca =pca.fit_transform(x_scaled)

#SPLITTING OF DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size= 0.2,stratify= y, random_state= 1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

#BUILDING MODEL
print()
model = DecisionTreeClassifier(criterion='entropy', max_depth=4, class_weight='balanced')
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print(y_pred)

print()
print("EVALUATION")
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

print()
print("ACCURACY SCORE :-\n", acc)
print("CONFUSION MATRIX:-\n", cm)
print("CLASSIFCIATION REPORT :-\n", clr)

print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model, x_pca, y, cv = 5)
print("THE SCORES:-\n", scores)
print("THE MEAN SCORES:-\n", np.mean(scores))



#VISUALIZATION
#CONFUSION <ATRIX
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


#PCA  PLOT
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', color='green')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Explained Variance')
plt.grid(True)
plt.show()


#DECISION TREE
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=[f'PC{i+1}' for i in range(x_pca.shape[1])], class_names=['No Disease', 'Disease'])
plt.title("Decision Tree (PCA Features)")
plt.show()


#CROSS VALIDATION SCORE
plt.figure(figsize=(6,4))
sns.boxplot(data=scores, orient='h', color='skyblue')
plt.scatter(scores, np.ones_like(scores), color='red', label='Fold Scores')
plt.xlabel('Accuracy')
plt.title('Cross-Validation Accuracy Scores')
plt.legend()
plt.grid(True)
plt.show()


#ACTUAL VS PREDICTED
comparison.reset_index(drop=True, inplace=True)
plt.figure(figsize=(12,4))
plt.plot(comparison['Actual'].values[:50], label='Actual', marker='o')
plt.plot(comparison['Predict'].values[:50], label='Predicted', marker='x')
plt.title('Actual vs Predicted (First 50 Test Cases)')
plt.xlabel('Index')
plt.ylabel('Heart Disease')
plt.legend()
plt.grid(True)
plt.show()

