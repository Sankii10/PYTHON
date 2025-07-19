#THE FOLLOWING CODE IS EXECUTED BY :- MR. SANKET GAIKWAD
#ALGORITHM :- SUPPORT VECTOR CLASSIFIER
# SOURCE :- AI Assistant Usage in Student Life(https://www.kaggle.com/datasets/ayeshasal89/ai-assistant-usage-in-student-life-synthetic) 


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


#IMPORTING DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\ai.csv")
print(a.columns.tolist())


#DROPPING UNWANTED COLUMNS
print()
a = a.drop(['SessionID','SessionDate'], axis = 1)


#INFORMATION ABOUT DATASET
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
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#LABEL ENCODING
print()
le = LabelEncoder()
for col in ['StudentLevel', 'Discipline','TaskType','FinalOutcome', 'UsedAgain']:
    a[col] = le.fit_transform(a[col])


#FEATURE SELECTION
print()
x = a.drop('UsedAgain', axis = 1)
y = a['UsedAgain']


#STANDARD SCALER
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components=0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLITING OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state = 1)


print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model = SVC(kernel='rbf',class_weight='balanced')
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print(y_pred)


#MODEL EVALUATION 
print()
print("EVALUATION")
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

print()
print("ACCURACY SCORE:-\n", acc)
print("CONFUSION MATRIX:-\n", cm)
print("CLASSIFICATION REPORT:-\n", clr)


#COMPARISON
print()
print("COMPARISOn")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)


#CROSS VALIDATION SCORES
print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model, x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALIZATION

#Correlation Heatmap (before PCA)
plt.figure(figsize=(10,6))
sns.heatmap(a.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

#Target Variable Distribution (UsedAgain)
sns.countplot(x='UsedAgain', data=a)
plt.title("Distribution of Target Variable (UsedAgain)")
plt.xlabel("Used Again (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()


#PCA Explained Variance Plot
pca_full = PCA().fit(x_scaled)
plt.figure(figsize=(10,6))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance by PCA Components")
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle='--')
plt.show()


#Confusion Matrix Heatmap
from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - SVC")
plt.show()


#Comparison Plot of Actual vs Predicted
comparison_sample = comparison.sample(100, random_state=1).reset_index(drop=True)
plt.figure(figsize=(12,6))
plt.plot(comparison_sample['Actual'].values, label='Actual', marker='o')
plt.plot(comparison_sample['Predict'].values, label='Predicted', marker='x')
plt.title("Actual vs Predicted Labels (Sample of 100)")
plt.xlabel("Sample Index")
plt.ylabel("Class")
plt.legend()
plt.grid(True)
plt.show()

