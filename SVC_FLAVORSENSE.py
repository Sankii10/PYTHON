#THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
# ALGORITH :- SUPPORT VECTOR CLASSIFIER
# SOURCE :- FlavorSense: Tastes Predicted by Life & Climate(https://www.kaggle.com/datasets/milapgohil/flavorsense-tastes-predicted-by-life-and-climate)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import missingno as msno


#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\FlavorSense.csv")
print(a.columns.tolist())


#DATASET INFORMATION
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
    print("N SUCH MISSING VALUE FOUND IN DATASET !!!")


#HANDLING OF MISSING VALUES
print()
a['age'] = a['age'].fillna(a['age'].median())

for col in ['sleep_cycle', 'exercise_habits', 'climate_zone', 'historical_cuisine_exposure']:
    a[col] = a[col].fillna(a[col].mode()[0])


#LABEL ENCODER
print()
le = LabelEncoder()
for col in ['sleep_cycle', 'exercise_habits', 'climate_zone', 'historical_cuisine_exposure', 'preferred_taste']:
    a[col] = le.fit_transform(a[col])


#FEATURE SELECTION
print()
x = a.drop('preferred_taste', axis = 1)
y = a['preferred_taste']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca = pca.fit_transform(x_scaled)


#SPLITTING OF DATA IN TRAIN AND TEST
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state= 1)

print()
smote = SMOTE(random_state= 1)
x_train_res,y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model = SVC(kernel='rbf', class_weight='balanced')
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
print("CLASSIFICATION REPORT:-\n", clr)


#COMPARISON OF ACTUAL AND PREDICTED
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model, x_pca,y, cv = 10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALIZATION

#Missing Value Heatmap (Before Imputation)
msno.matrix(pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\FlavorSense.csv"))
plt.title("Missing Value Heatmap")
plt.show()


# PCA Explained Variance Ratio
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.title("PCA - Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Explained Variance Ratio")
plt.grid(True)
plt.show()

#Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVC")
plt.show()


#Actual vs Predicted Count Plot
plt.figure(figsize=(10,5))
comparison['Actual'] = comparison['Actual'].astype(str)
comparison['Predict'] = comparison['Predict'].astype(str)

comparison_melted = pd.melt(comparison.reset_index(), id_vars='index', value_vars=['Actual', 'Predict'], 
                            var_name='Type', value_name='Taste')

sns.countplot(x='Taste', hue='Type', data=comparison_melted, palette='Set2')
plt.title("Actual vs Predicted Taste Distribution")
plt.show()
