#THE CODE IS EXECUTED BY ME.SANKET GAIKWAD
#ALGORITHM USED :- SVC
#SOURCE :- https://www.kaggle.com/datasets/sudarshan24byte/online-food-dataset

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\Desktop\\My Data\\onlinefoods.csv")
print(a.columns.tolist())


#DROPING UNWANTED FEATURES
print()
a = a.drop(['Feedback', 'Unnamed: 12'], axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION
print()
print(a.info())

print()
print(a.describe())

#SEARCHING FOR DUPLICATED VALUES
print()
print("DUPLICATED VALUES")
print(a.duplicated().sum())

print()
c = a[a.duplicated()]
print(c)


#REMOVING DUPLICATED VALUES
print()
a = a.drop_duplicates()


#SEARCHING FOR MISSING VALUES
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!")


#SEGREGATION OF CATEGORICAL AND CONTINOUS VALUES
print()
cat = []
con = []

for i in a:
    if a[i].dtypes == 'object':
        cat.append(i)
    else:
        con.append(i)

print()
print("CATEGORICAL COLUMNS:-\n", cat)
print("CONTINOUS VALUES :-\n", con)


#ONE HOT ENCODING
print()
a = pd.get_dummies(a, columns = ['Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications'], drop_first=True)


#LABEL ENCODING FOR TARGET VARIABLE
le = LabelEncoder()
a['Output'] = le.fit_transform(a['Output'])


#FEATURES SELECTION
print()
x = a.drop('Output', axis = 1)
y = a['Output']


# CHECKS THE PROPORTION (PERCENTAGE DISTRIBUTION) OF EACH CLASS IN THE TARGET VARIABLE TO IDENTIFY CLASS IMBALANCE
print()
print(a['Output'].value_counts(normalize=True))


# COMPUTES AND SORTS THE CORRELATION OF ALL NUMERICAL FEATURES WITH THE TARGET VARIABLE TO IDENTIFY FEATURES MOST RELATED TO THE OUTPUT
print()
print(a.corr(numeric_only=True)['Output'].sort_values(ascending=True))


#SPLIT OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 1)


#STANDARD SCALER
print()
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca =PCA(n_components = 0.95)
x_train_pca =pca.fit_transform(x_train_sc)
x_test_pca =pca.transform(x_test_sc)


#SMOTE
print()
smote = SMOTE(random_state=1)
x_train_res, y_train_res = smote.fit_resample(x_train_pca, y_train)


#MODEL BUILDING AND IMPLEMENTATION
print()
model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
model.fit(x_train_res, y_train_res)

y_pred_prob = model.predict_proba(x_test_pca)[:,1]
y_pred = (y_pred_prob >= 0.45).astype(int)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
print("ACCURACY SCORE:-\n", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX :-\n", confusion_matrix(y_test, y_pred))
print("CLASSIFICATION REPORT :-\n",classification_report(y_test, y_pred))


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)


#CROSS SCORE VALIDATION
print()
print("CROSS SCORE VALIDATION")
kf = KFold(n_splits = 10, shuffle = True, random_state= 1)
pipe = Pipeline([
    ('Scaled', StandardScaler()),
    ('PCA', PCA(n_components=0.95)),
    ('smote', SMOTE(random_state=1)),
    ('model',SVC(kernel='rbf', C=10, gamma='scale', probability=True))
])

scores = cross_val_score(pipe,x, y, cv = kf, scoring = 'accuracy')
print(scores)
print("MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALIZATION
# CONFUSION MATRIX HEATMAP
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt='d', cmap='Blues')
plt.xlabel("PREDICTED LABEL")
plt.ylabel("ACTUAL LABEL")
plt.title("CONFUSION MATRIX")
plt.show()


# CLASS DISTRIBUTION BEFORE SMOTE
plt.figure(figsize=(5,4))
sns.countplot(x=y, palette="viridis")
plt.xlabel("OUTPUT CLASS")
plt.ylabel("COUNT")
plt.title("CLASS DISTRIBUTION BEFORE SMOTE")
plt.show()


# CLASS DISTRIBUTION AFTER SMOTE
plt.figure(figsize=(5,4))
sns.countplot(x=y_train_res, palette="magma")
plt.xlabel("OUTPUT CLASS")
plt.ylabel("COUNT")
plt.title("CLASS DISTRIBUTION AFTER SMOTE")
plt.show()


# CROSS VALIDATION ACCURACY SCORES
plt.figure(figsize=(7,4))
plt.plot(scores, marker='o')
plt.axhline(np.mean(scores), linestyle='--')
plt.xlabel("CV FOLD")
plt.ylabel("ACCURACY")
plt.title("CROSS VALIDATION ACCURACY ACROSS FOLDS")
plt.show()


# PREDICTED PROBABILITY DISTRIBUTION
plt.figure(figsize=(7,4))
plt.hist(y_pred_prob, bins=20)
plt.xlabel("PREDICTED PROBABILITY FOR CLASS 1")
plt.ylabel("COUNT")
plt.title("PREDICTED PROBABILITY DISTRIBUTION")
plt.show()


# ACTUAL VS PREDICTED COMPARISON
comparison_sorted = comparison.reset_index(drop=True)

plt.figure(figsize=(10,4))
plt.plot(comparison_sorted['Actual'], label='ACTUAL', marker='o')
plt.plot(comparison_sorted['Predict'], label='PREDICTED', marker='x')
plt.xlabel("TEST SAMPLE INDEX")
plt.ylabel("CLASS LABEL")
plt.title("ACTUAL VS PREDICTED OUTPUT")
plt.legend()
plt.show()
