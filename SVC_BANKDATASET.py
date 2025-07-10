#THE FOLLOWING CODE IS EXECUTED BY :-
# MR. SANKET GAIKWAD
# SOURCE :- BANK DATASET (https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\PYTHON\\DATASET\\bank.csv")
print(a.columns.tolist())

print()
a = a.drop('day', axis = 1)
print(a)

print()
print(a.info())

print()
b = a.isnull().sum()
print(b)

#HANDLING MISSING VALUES
print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VAUE FOUND IN DATASET !!!")


#LABEL ENCODER
print()
le = LabelEncoder()
for col in ['job', 'marital', 'education', 'default','housing', 'loan', 'contact','month','poutcome', 'deposit']:
    a[col] = le.fit_transform(a[col])


#FEATURE SELECTION
print()
x  =a.drop('deposit', axis = 1)
y = a['deposit']

#STANDARD SCALER
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca = pca.fit_transform(x_scaled)


#SPLITTING OF DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state= 1)


print()
smote = SMOTE(random_state = 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

#MODEL BUILDING AND EXECUTION
print()
model = SVC(kernel='rbf', class_weight='balanced')
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print("TEHE PREDICTIONS :-\n", y_pred)

print()
print("EVALUATION")
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

print()
print("ACCURACY SCORE:-\n", accuracy_score)
print("CONFUSION MATRIX:-\n", cm)
print("CLASSIFICATION REPORT :-\n", clr)

print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model, x_pca, y, cv  =10)
print("THE SCORES ARE :-\n", scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALIZATION
#Confusion Matrix plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Deposit', 'Deposit'], yticklabels=['No Deposit', 'Deposit'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - SVC')
plt.show()


#Classification Report as Heatmap
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

plt.figure(figsize=(8, 5))
sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap='YlGnBu')
plt.title('Classification Report - SVC')
plt.show()

#PCA Explained variance plot
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.grid(True)
plt.show()


#Actual VS Prdict
comparison['Match'] = comparison['Actual'] == comparison['Predict']
match_counts = comparison['Match'].value_counts()

plt.figure(figsize=(6, 4))
sns.barplot(x=match_counts.index, y=match_counts.values)
plt.xticks([0, 1], ['Mismatch', 'Match'])
plt.ylabel('Count')
plt.title('Actual vs Predicted Match Count')
plt.show()


#Cross Validation score distribution
plt.figure(figsize=(8, 4))
sns.boxplot(data=scores)
plt.title('Cross-Validation Score Distribution')
plt.xlabel('Accuracy Score')
plt.grid(True)
plt.show()
