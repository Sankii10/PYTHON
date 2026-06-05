#THE CODE IS EXECUTED BY : SANKET GAIKWAD
#ALGORITHM USED :- RANDOM FOREST CLASSIFIER


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE


#IMPORTING THE DATA FILE
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\MARKETING ANALYTICS\\Dataset\\bank.csv")
print(a.columns.tolist())


#DROPPING NOT REQUIRED FEATURE
print()
a = a.drop( 'Experience', axis = 1)

#INFORMATION AND STATISTICAL DESCRIPTION OF THE DATA
print()
print(a.info())

print()
print(a.describe())


#TO SEARCH OF SUM OF DUPLICATED VALUES 
print()
print("TOTAL SUM OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())


#DISPLAYING THE DUPLICATED VALUES ROWS 
print()
c = a[a.duplicated()]
print(c)


#DROPING THE DUPLICATED VALUES(ROWS)
print()
a = a.drop_duplicates()


#SEARCHING FOR MISSING VALUES IN DATA
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#SEGREGATING THE CATEGORICAL AND CONTINOUS VALUES 
print()
cat = []
con = []

for i in a:
    if a[i].dtypes == 'object':
        cat.append(i)
    else:
        con.append(i)

print()
print("CATEGORICAL VALUES :-\n",cat)
print("CONTINOUS VALUES :-\n", con)


#PRINTS THE PROPORTION OF EACH UNIQUE VALUE IN THE 'PERSONAL LOAN' COLUMN
print()
print(a['Personal Loan'].value_counts(normalize=True))


#PRINTS THE CORRELATION OF ALL NUMERIC COLUMNS WITH 'PERSONAL LOAN', SORTED IN DESCENDING ORDER
print()
print(a.corr(numeric_only=True)['Personal Loan'].sort_values(ascending=False))


#FEATURES SELECTION
print()
x = a.drop('Personal Loan', axis = 1)
y = a['Personal Loan']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLITING OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state= 1)

print()
smote = SMOTE(random_state=1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING AND IMPLEMENTING
print()
model = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=None, random_state=1)
model.fit(x_train_res, y_train_res)

y_predict_prob = model.predict_proba(x_test)[:,1]
y_pred = (y_predict_prob>=0.5).astype(int)


#EVALUATION
print()
print("EVALUATION")
print("ACCURACY SCORE :-\n", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX :-\n", confusion_matrix(y_test, y_pred))
print("CLASSIFICATION REPORT :-\n", classification_report(y_test, y_pred))
print("RECALL SCORE:-\n", recall_score(y_test, y_pred))
print("ROC SCORE:-\n", roc_auc_score(y_test, y_pred))


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model,x_pca, y, cv  =10)
print(scores)
print("THE MEAN SCORES :-\n", np.mean(scores))


#DATA VISUALISATION

# SETTING STYLE
sns.set(style="whitegrid")

#  DISTRIBUTION OF TARGET VARIABLE 'PERSONAL LOAN'
plt.figure(figsize=(6,4))
sns.countplot(x='Personal Loan', data=a)
plt.title('DISTRIBUTION OF PERSONAL LOAN')
plt.show()

#  COUNT PLOTS FOR CATEGORICAL VARIABLES
plt.figure(figsize=(15,6))
for i, col in enumerate(cat):
    plt.subplot(1, len(cat), i+1)
    sns.countplot(x=col, data=a, palette='pastel')
    plt.title(f'COUNT PLOT OF {col.upper()}')
plt.tight_layout()
plt.show()

#  HEATMAP OF CORRELATION MATRIX
plt.figure(figsize=(12,8))
sns.heatmap(a.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('CORRELATION MATRIX')
plt.show()

#  PCA EXPLAINED VARIANCE RATIO
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_)*100, marker='o', color='purple')
plt.xlabel('NUMBER OF COMPONENTS')
plt.ylabel('CUMULATIVE EXPLAINED VARIANCE (%)')
plt.title('PCA EXPLAINED VARIANCE')
plt.grid(True)
plt.show()

#  CLASS IMBALANCE VISUALIZATION AFTER SMOTE
plt.figure(figsize=(6,4))
sns.countplot(y_train_res)
plt.title('CLASS DISTRIBUTION AFTER SMOTE')
plt.show()


#  CONFUSION MATRIX HEATMAP
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('PREDICTED')
plt.ylabel('ACTUAL')
plt.title('CONFUSION MATRIX HEATMAP')
plt.show()

#  ROC CURVE
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_predict_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('FALSE POSITIVE RATE')
plt.ylabel('TRUE POSITIVE RATE')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()



