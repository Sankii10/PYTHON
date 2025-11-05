#THE CODE IS EXECUTED BY :- SANKET GAIKWAD
#ALGORITHM USED :- SVM
#SOURCE :- https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


#IMPORT DATA 
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\add.csv")
print(a.columns.tolist())


#DROPPING UNREQUIRED FEATURES
print()
a = a.drop(['PatientID','DoctorInCharge'], axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION OF THE DATA
print()
print(a.info())

print()
print(a.describe())


#FINDING THE TOTAL SUM OF DUPLICATED VALUES IN DATA
print()
print("TOTAL SUM OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())


#FINDING THE DUPLICATED VALUES/ROWS
print()
c = a[a.duplicated()]
print(c)


#SEARCHING FOR MISSING OR NULL VALUES IN DATA
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND !!!")


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


# SHOWS THE PROPORTION (%) OF EACH CLASS IN THE TARGET VARIABLE
print()
print(a['Diagnosis'].value_counts(normalize = True))


# DISPLAY FEATURES MOST POSITIVELY OR NEGATIVELY CORRELATED WITH DIAGNOSIS
print()
print(a.corr(numeric_only=True)['Diagnosis'].sort_values(ascending=False))


#FEATURES SELECTION
print()
x = a.drop('Diagnosis', axis = 1)
y = a['Diagnosis']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components = 0.95)
x_pca=pca.fit_transform(x_scaled)


#SPLIT OF DATA INTO TRAIN AND TEST
print()
train_size = int(0.8 * len(x_pca))
x_train, x_test = x_pca[:train_size],x_pca[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING AND IMPLEMENTATION
print()
model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
model.fit(x_train_res, y_train_res)

y_pred_prob = model.predict_proba(x_test)[:,1]
y_pred = (y_pred_prob > 0.5).astype(int)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
print("ACCURACY SCORE:-\n", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX :-\n", confusion_matrix(y_test, y_pred))
print("CLASSIFICATION REPORT:-\n", classification_report(y_test, y_pred))


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
kf = KFold(n_splits = 10, shuffle = True, random_state= 1)
scores  =cross_val_score(model,x_pca, y,cv = kf, scoring = 'r2')
print(scores)
print("MEAN SCORES :-\n", np.mean(scores))


#DATA VISUALISATION

# 1️⃣ CONFUSION MATRIX HEATMAP
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("CONFUSION MATRIX HEATMAP")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 2️⃣ CLASS DISTRIBUTION BEFORE & AFTER SMOTE
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.countplot(x=y, palette='Set2')
plt.title("CLASS DISTRIBUTION BEFORE SMOTE")
plt.xlabel("Diagnosis")
plt.ylabel("Count")

plt.subplot(1,2,2)
sns.countplot(x=y_train_res, palette='Set1')
plt.title("CLASS DISTRIBUTION AFTER SMOTE")
plt.xlabel("Diagnosis")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# 3️⃣ PCA EXPLAINED VARIANCE RATIO (FEATURE IMPORTANCE IN PCA SPACE)
plt.figure(figsize=(7,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title("CUMULATIVE EXPLAINED VARIANCE BY PCA COMPONENTS")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Variance Explained")
plt.grid(True)
plt.show()

# 4️⃣ MODEL PERFORMANCE COMPARISON USING CROSS VALIDATION SCORES
plt.figure(figsize=(6,4))
plt.plot(scores, marker='o')
plt.title("CROSS VALIDATION SCORES (R²)")
plt.xlabel("Fold")
plt.ylabel("Score")
plt.grid(True)
plt.show()


