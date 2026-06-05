""" THE FOLLOWING CODE HAS BEEN CREATED BY:-
    SANKET GAIKWAD
    DATABASE :- RAINFALL PREDICTION USA"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

#UPLOAD THE FILE
a = pd.read_csv("C:\\Users\\anujg\\OneDrive\\Desktop\\MY DATA\\DATASET\\rainfall_prediction.csv")
print(a.columns.tolist())

print()
a = a.drop('Date', axis = 1)


#DATA PREPROCESSING
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
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")

#LABEL ENCODING
print()
le = LabelEncoder()
a['Location'] = le.fit_transform(a['Location'])

print()
x = a.drop('Rain Tomorrow', axis = 1)
y = a['Rain Tomorrow']

#STANDARD SCALING
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca = pca.fit_transform(x_scaled)

#FEATURE SELECTION
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state = 1)

print()
smote = SMOTE(random_state=1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

#BUILDING MODEL
print()
model = RandomForestClassifier(n_estimators= 150, criterion='entropy', max_depth=4, class_weight='balanced')
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print(y_pred)

print()
print("EVALUATION")
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

print()
print("ACCURACY SCORE:-\n", acc)
print("CONFUSION MATRIX:-\n", cm)
print("CLASSIFICATION REPORT :-\n", clr)

print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model, x_pca, y, cv = 10)
print("THE SCORES ARE :-\n", scores)
print("THE MEAN SCORES ARE :-\n", np.mean(scores))


#VISUALIZATION
sns.countplot(x='Rain Tomorrow', data=a)
plt.title('Distribution of Target Variable - Rain Tomorrow')
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

comparison_sample = comparison.sample(30, random_state=1)
comparison_sample.reset_index(drop=True).plot(kind='bar', figsize=(12, 6))
plt.title('Actual vs Predicted - Sample View')
plt.xlabel('Sample Index')
plt.ylabel('Rain Tomorrow')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), scores, marker='o')
plt.title('Cross-Validation Accuracy Scores (10-Fold)')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(range(1, 11))
plt.show()

