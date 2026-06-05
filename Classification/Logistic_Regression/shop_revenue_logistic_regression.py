#THE CODE IS EXECUTED BY :- SANKET GAIKWAD
# DATASET :-Shopper's Behavior and Revenue
# SOURCE :- KAGGLE(https://www.kaggle.com/datasets/subhajournal/shoppers-behavior-and-revenue)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


#Import the dat
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\ShopRevenue.csv")
print(a.columns.tolist())

#Information Data
print()
print(a.info())

#Search for missing value 
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")

#Label Encoder
print()
le = LabelEncoder()
for col in ['Month', 'VisitorType', 'Weekend', 'Revenue']:
   a[col] = le.fit_transform(a[col])

#Feature selection
print()
x = a.drop( 'Revenue', axis = 1)
y = a[ 'Revenue']

#Standard Scaler
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

#Principal Component Analysis
print()
pca = PCA(n_components = 0.95)
x_pca =pca.fit_transform(x_scaled)

#Splitting of Data 
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state = 1)


print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

#Model Building
print()
model = LogisticRegression()
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print(y_pred)

#Evaluation
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
scores = cross_val_score(model, x_pca, y, cv  =10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#Data Visualization

#Confusion Matrix (Heatmap)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Revenue', 'Revenue'], yticklabels=['No Revenue', 'Revenue'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


#ROC Curve with AUC Score
from sklearn.metrics import roc_curve, roc_auc_score

y_prob = model.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

#Cross-Validation Scores (Bar Plot)
plt.figure(figsize=(8, 4))
plt.bar(range(1, 11), scores, color='skyblue')
plt.axhline(np.mean(scores), color='red', linestyle='--', label=f"Mean Score: {np.mean(scores):.4f}")
plt.xticks(range(1, 11))
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores')
plt.legend()
plt.grid(True)
plt.show()


#PCA Variance Explained (Optional)
pca_check = PCA().fit(x_scaled)
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca_check.explained_variance_ratio_), marker='o', linestyle='--', color='purple')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.show()
