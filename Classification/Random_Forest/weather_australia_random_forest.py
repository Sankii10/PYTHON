#THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM :- RANDOM FOREST CLASSIFIER
#SOURCE:- RAIN IN AUSTRALIA (https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)


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
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\weatherAUS.csv")
print(a.columns.tolist())


#DROPPING UNWANTED COLUMNS
print()
a = a.drop('Date', axis = 1)


#INFORMATION ABOUT THE DATA
print()
print(a.info())


#SEARCHING THE MISSING VALUES
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#HANDLING MISSING VALUES

#HANDLING CATEGORICAL MISSING VALUES BY REPLACING WITH VALUE OF MODE
for col in ['RainToday', 'RainTomorrow', 'WindDir9am', 'WindDir3pm', 'Evaporation', 'Sunshine', 'WindGustDir']:
    a[col] = a[col].fillna(a[col].mode()[0])

#HANDLING NUMERIC(CONTINOUS) MISSING VALUES BY REPLACING MEDIAN VALUES
for col in [ 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Cloud9am', 'Cloud3pm', 'Temp3pm']:
    a[col] = a[col].fillna(a[col].median())


#LABEL ENCODING
print()
le = LabelEncoder()
for col in ['Location','RainToday', 'RainTomorrow', 'WindDir9am', 'WindDir3pm', 'Evaporation', 'Sunshine', 'WindGustDir']:
    a[col] = le.fit_transform(a[col])


#FEATURE SELECTION
print()
x = a.drop('RainTomorrow', axis = 1)
y = a['RainTomorrow']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled  =sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLING OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y ,test_size = 0.2, random_state= 1)

print()
smote = SMOTE(random_state = 1)
x_train_res , y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model = RandomForestClassifier(n_estimators=150, criterion='entropy',class_weight='balanced',max_depth=4)
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
scores = cross_val_score(model, x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALIZATION

# Confusion Matrix Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


#Feature Importance
plt.figure(figsize=(8,5))
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.title("Feature Importances (PCA Components)")
plt.xlabel("Principal Components")
plt.ylabel("Importance")
plt.show()


#Actual vs Predicted Bar Plot
comparison_sample = comparison.sample(50).reset_index(drop=True)

plt.figure(figsize=(12,6))
plt.plot(comparison_sample['Actual'], label='Actual', marker='o')
plt.plot(comparison_sample['Predict'], label='Predicted', marker='x')
plt.title("Actual vs Predicted - Sample 50")
plt.xlabel("Sample Index")
plt.ylabel("RainTomorrow")
plt.legend()
plt.grid(True)
plt.show()


#Cross-Validation Score Distribution
plt.figure(figsize=(8,5))
plt.plot(scores, marker='o', linestyle='--', color='green')
plt.title("Cross-Validation Scores (10-Fold)")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
