#THE CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- RANDOM FOREST CLASSIFIER
#SOURCE :- Gallstone Dataset (UCI) ( https://www.kaggle.com/datasets/xixama/gallstone-dataset-uci )



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
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\gs.csv")
print(a.columns.tolist())


#INFORMATION OF DATA
print()
print(a.info())

print()
print(a.describe())


#FINDING THE MISSING VALUES
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATSET !!!!")


#FEATURE SELECTION
print()
x = a.drop('Gallstone Status', axis = 1)
y = a['Gallstone Status']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLITING OF DATA AS TRAIN & TEST
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=  0.2, random_state = 1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model = RandomForestClassifier(n_estimators=150, class_weight='balanced', criterion='entropy', max_depth=4)
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
acc=  accuracy_score(y_test, y_pred)
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
print("CROSS VALIDATION SCORES")
scores = cross_val_score(model,x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALISATION

#Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, 
            yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()


#PCA Explained Variance
plt.figure(figsize=(6, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.show()


#Cross-Validation Score Distribution
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(scores)+1), scores, marker='o')
plt.axhline(np.mean(scores), color='red', linestyle='--', label=f"Mean: {np.mean(scores):.2f}")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Cross-Validation Accuracy per Fold")
plt.legend()
plt.show()


#Actual vs Predicted Count Plot
plt.figure(figsize=(6, 4))
comparison_counts = comparison.value_counts().reset_index(name='count')
sns.barplot(data=comparison_counts, x='Actual', y='count', hue='Predict')
plt.title("Actual vs Predicted Counts")
plt.show()



