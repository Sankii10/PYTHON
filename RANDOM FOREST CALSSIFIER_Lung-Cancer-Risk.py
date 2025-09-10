#THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- RANDOM FOREST CLASSIFIER
#SOURCE :- https://www.kaggle.com/datasets/mikeytracegod/lung-cancer-risk-dataset





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
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\lung.csv")
print(a.columns.tolist())


#DROPPING UNREQUIRED FEATURE
print()
a = a.drop('patient_id', axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION OF THE DATA
print()
print(a.info())

print()
print(a.describe())


#SEARCHING FOR TOTAL DUPLICATED VALUES
print()
print("TOTAL SUM OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())


#DISPLAYING ROW WISE DUPLICATED VALUES
print()
c = a[a.duplicated()]
print(c)


#SEARCHING FOR MISSING VALUES IN DATA
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print('NO SUCH MISSING VALUE FOUND IN DATASET !!!')


#HANDING OF MISSING VALUES 
print()
a['alcohol_consumption'] = a['alcohol_consumption'].fillna(a['alcohol_consumption'].mode()[0])


#ONE HOT ENCODING
print()
a = pd.get_dummies(a, columns = ['gender', 'radon_exposure', 'asbestos_exposure', 'secondhand_smoke_exposure', 'copd_diagnosis', 'alcohol_consumption', 'family_history'], drop_first=True)

#LABEL ENCODING OF TARGET VARIABLE
le = LabelEncoder()
a['lung_cancer'] = le.fit_transform(a['lung_cancer'])


# SHOWS CLASS DISTRIBUTION (NORMALIZED TO PERCENTAGE) FOR LUNG_CANCER (0 = NO, 1 = YES)
print()
print(a['lung_cancer'].value_counts(normalize= True))


# SHOWS CORRELATION OF EACH NUMERIC FEATURE WITH LUNG_CANCER, SORTED FROM HIGHEST TO LOWEST
print()
print(a.corr(numeric_only= True)['lung_cancer'].sort_values(ascending = False))


#FEATURES SELECTION
print()
x = a.drop('lung_cancer', axis = 1)
y = a['lung_cancer']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca = pca.fit_transform(x_scaled)


#SPLIT OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state = 1)



# APPLY SMOTE TO BALANCE THE CLASSES IN TRAINING DATA
print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res  = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model = RandomForestClassifier(n_estimators=150, max_depth=None, criterion='entropy')
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
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

print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model,x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES :-\n", np.mean(scores))


#DATA VISUALISATION

# 1. CLASS DISTRIBUTION (TARGET VARIABLE)
plt.figure(figsize=(6,4))
sns.countplot(x=y, palette="coolwarm")
plt.title("Class Distribution of Lung Cancer")
plt.xlabel("Lung Cancer (0=No, 1=Yes)")
plt.ylabel("Count")
plt.show()

# 2. CONFUSION MATRIX HEATMAP
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 3. FEATURE IMPORTANCES (TOP 15 FEATURES)
importances = model.feature_importances_
indices = np.argsort(importances)[-15:]   # TOP 15 FEATURES
plt.figure(figsize=(10,6))
plt.barh(range(len(indices)), importances[indices], align="center", color="teal")
plt.yticks(range(len(indices)), [x.columns[i] for i in indices])
plt.title("Top 15 Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# 4. CROSS-VALIDATION SCORES DISTRIBUTION
plt.figure(figsize=(6,4))
sns.lineplot(x=range(1, len(scores)+1), y=scores, marker="o", color="red")
plt.title("Cross-Validation Scores")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.show()

# 5. COMPARISON OF ACTUAL VS PREDICTED
comparison_sample = comparison.sample(30, random_state=1)   # TAKE SAMPLE TO AVOID OVERCROWDING
plt.figure(figsize=(12,6))
plt.plot(comparison_sample.index, comparison_sample['Actual'], marker='o', label="Actual", color="blue")
plt.plot(comparison_sample.index, comparison_sample['Predict'], marker='x', label="Predicted", color="orange")
plt.title("Actual vs Predicted (Sample of 30)")
plt.xlabel("Index")
plt.ylabel("Class (0=No, 1=Yes)")
plt.legend()
plt.show()


