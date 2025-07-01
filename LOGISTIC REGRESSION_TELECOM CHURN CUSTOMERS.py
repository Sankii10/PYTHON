#THE FOLLOWING CODE IS EXECUTED BY
# MR.SANKET GAIKWAD
#SOURCE :- KAGGEL(TELECOM CHURN CUSTOMERS)

__________________________________________________________________________________________________________________________________________

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,  ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


a  =pd.read_csv("C:\\Users\\anujg\\OneDrive\\Desktop\\MY DATA\\DATASET\\TeleCustChurn.csv")
print(a.columns.tolist())


print()
a = a.drop('customerID', axis = 1)

# Convert 'TotalCharges' to numeric, setting invalid (non-numeric) entries to NaN
print()
a['TotalCharges'] = pd.to_numeric(a['TotalCharges'], errors='coerce')


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


#HANDLING MISSING VALUE
print()
a['TotalCharges'] = a['TotalCharges'].fillna(a['TotalCharges'].mode()[0])


#LABEL ENCODER
print()
le = LabelEncoder()
for col in ['gender','Partner', 'Dependents','PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']:
    a[col] = le.fit_transform(a[col])


#FEATURE SELECTION
print()
x = a.drop('Churn', axis = 1)
y = a['Churn']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca =PCA(n_components = 0.95)
x_pca = pca.fit_transform(x_scaled)

#SPLITTING OF DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state = 1)


#SMOTE
print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model = LogisticRegression()
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print("THE PREDICTIONS ARE :-\n", y_pred)


#EVALUATIONS AND RESULT
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
print("THE CROSS SCORE VALIDATION IS :-\n", scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


# 1. CONFUSION MATRIX HEATMAP
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 2. CROSS VALIDATION SCORES PLOT
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), scores, marker='o', linestyle='--', color='b')
plt.title("Cross-Validation Scores (Logistic Regression)")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0.6, 1.0)
plt.grid(True)
plt.show()

# 3. PCA EXPLAINED VARIANCE RATIO PLOT
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='green')
plt.title("PCA - Cumulative Explained Variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.show()