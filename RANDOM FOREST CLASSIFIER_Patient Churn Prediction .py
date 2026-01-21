#THE CODE IS EXECUTED BY MR.SANKET GAIKWAD
#ALGORITHM USED : RANDOM FOREST CLASSIFIER
#SOURCE : https://www.kaggle.com/datasets/nudratabbas/patient-churn-prediction-dataset-for-healthcare


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline


#IMPORTING DATA 
a= pd.read_csv("C:\\Users\\ASUS\\OneDrive\\Desktop\\My Data\\patient.csv")
print(a.columns.tolist())


#REMOVING THE UNREQUIRED FEATURES
print()
a = a.drop(['PatientID', 'Last_Interaction_Date'], axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION OF DATASET
print()
print(a.info())

print()
print(a.describe())


#SEARCHING FOR DUPLICATED VALUES
print()
print("DUPLICATED VALUES :-\n")
print(a.duplicated().sum())

print()
c = a[a.duplicated()]
print(c)


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
    if a[i].dtype == 'object':
        cat.append(i)
    else:
        con.append(i)

print()
print("CATEGORICAL VALUES :-\n", cat)
print("CONTINOUS VALUES :-\n", con)


#ONE HOT ENCODING
print()
a = pd.get_dummies(a, columns = ['Gender', 'State', 'Specialty', 'Insurance_Type'], drop_first=True)


#FEATURE SELECTION
print()
x = a.drop('Churned', axis = 1)
y = a['Churned']


#SPLIT OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=  0.2, random_state= 1)

print()
print(a['Churned'].value_counts(normalize= True))

print()
print(a.corr(numeric_only=True)['Churned'].sort_values(ascending=False))

#STANDARD SCALER
print()
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_train_pca = pca.fit_transform(x_train_sc)
x_test_pca = pca.transform(x_test_sc)


#MODEL BUILDING AND INTERPRETATION
print()
model = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=10, min_samples_leaf=5, max_features='sqrt', class_weight='balanced', random_state=1)
model.fit(x_train_pca, y_train)

y_pred_prob = model.predict_proba(x_test_pca)[:,1]
y_pred = (y_pred_prob >= 0.5).astype(int)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
print("ACCURACY SCORE:-\n", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX :-\n", confusion_matrix(y_test, y_pred))
print("CLASSIFICATION REPORT :-\n", classification_report(y_test, y_pred))


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)


#CROSS SCORE VALIDATION USING PIPELINE
print()
print("CROSS SCORE VALIDATION")
kf = KFold(n_splits = 10, shuffle = True, random_state= 1)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('model', RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=10, min_samples_leaf=5, max_features='sqrt', class_weight='balanced', random_state=1))
])

scores = cross_val_score(pipe, x, y, cv=kf, scoring='accuracy')
print(scores)
print("MEAN SCORES :-\n", np.mean(scores))

# CREATE FIGURE WITH MULTIPLE SUBPLOTS
plt.figure(figsize=(18, 12))

# CLASS DISTRIBUTION
plt.subplot(2, 3, 1)
sns.countplot(x=y)
plt.title("CHURN CLASS DISTRIBUTION")
plt.xlabel("CHURNED")
plt.ylabel("COUNT")

# CORRELATION WITH TARGET
plt.subplot(2, 3, 2)
a.corr(numeric_only=True)['Churned'].sort_values().plot(kind='barh')
plt.title("CORRELATION OF FEATURES WITH CHURN")

# CONFUSION MATRIX HEATMAP
plt.subplot(2, 3, 3)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("CONFUSION MATRIX")
plt.xlabel("PREDICTED")
plt.ylabel("ACTUAL")


# PREDICTED PROBABILITY DISTRIBUTION
plt.subplot(2, 3, 5)
sns.histplot(y_pred_prob, bins=20, kde=True)
plt.title("PREDICTED CHURN PROBABILITY DISTRIBUTION")
plt.xlabel("PROBABILITY OF CHURN")

# CROSS VALIDATION SCORE DISTRIBUTION
plt.subplot(2, 3, 6)
sns.boxplot(x=scores)
plt.title("CROSS VALIDATION ACCURACY DISTRIBUTION")
plt.xlabel("ACCURACY")

# FINAL LAYOUT ADJUSTMENT
plt.tight_layout()
plt.show()