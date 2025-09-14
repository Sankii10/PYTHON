#THE CODE HAS BEEN EXECUTED BY MR.SANKET GAIKWAD
#ALGORITHM USED :- RANDOM FOREST CLASSIFIER
#SOURCE :- https://www.kaggle.com/datasets/reihanenamdari/breast-cancer


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
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\BC.csv")
print(a.columns.tolist())


#DROPPING UNREQUIRED COLUMNS
print()
a = a.drop(['T Stage ', 'N Stage', 'Grade', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18'], axis =1)


#INFORMATION AND STATISTICAL DESCRIPTION OF THE DATA
print()
print(a.info())

print()
print(a.describe())


#FINDING THE TOTAL NUMBER OF DUPLICATED VALUES
print()
print("TOTAL SUM OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())


#DISPLAY THE ROWS OF DUPLICATED VALUES
print()
c = a[a.duplicated()]
print(c)


#DROPPING THE DUPLICATED VALUES
print()
a.drop_duplicates()


#SEARCHING FOR THE MISSING VALUES 
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#SEGREGATING THE CATEGORICAL AND CONTINOUS VARIABLES
print()
cat = []
con = []

for i in a.columns:
    if a[i].dtypes == 'object':
        cat.append(i)
    else:
        con.append(i)

print()
print("CATEGORICAL VALUES :-\n", cat)
print("CONTINOUS VALUES :-\n", con)


#ONE HOT ENCODING
print()
a = pd.get_dummies(a, columns = ['Race', 'Marital Status', '6th Stage', 'differentiate', 'A Stage', 'Estrogen Status', 'Progesterone Status'], drop_first= True)


#LABEL ENCODING ON TARGET VARIABLE
le = LabelEncoder()
a['Status'] = le.fit_transform(a['Status'])


# CHECK THE DISTRIBUTION OF THE TARGET VARIABLE 'STATUS' IN PERCENTAGE TERMS
print()
print(a['Status'].value_counts(normalize= True))

# DISPLAY CORRELATION OF ALL NUMERIC FEATURES WITH THE TARGET VARIABLE 'STATUS', SORTED DESCENDING
print()
print(a.corr(numeric_only= True)['Status'].sort_values(ascending= False))


#FEATURE SELECTION
print()
x = a.drop('Status', axis = 1)
y = a['Status']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca= pca.fit_transform(x_scaled)


#SPLIT OF TRAIN AND TEST DATA 
print()
x_train , x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state= 1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model = RandomForestClassifier(n_estimators=300, criterion='entropy', max_depth=None, class_weight='balanced', random_state=1, verbose=False)  
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION OF DATASET
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
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALISATION

# 1. Target variable distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Status', data=a, palette='Set2')
plt.title("Distribution of Target Variable (Status)")
plt.xlabel("Status (0 = Alive, 1 = Dead)")
plt.ylabel("Count")
plt.show()


# 2. Missing values heatmap
plt.figure(figsize=(10,5))
sns.heatmap(a.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()


# 3. Correlation heatmap (with Status)
plt.figure(figsize=(12,8))
corr = a.corr(numeric_only=True)
sns.heatmap(corr, cmap='coolwarm', annot=False, linewidths=0.5)
plt.title("Correlation Heatmap of Features")
plt.show()

# Correlation of each feature with Status (barplot)
plt.figure(figsize=(10,5))
corr_status = corr['Status'].drop('Status').sort_values(ascending=False)
sns.barplot(x=corr_status.values, y=corr_status.index, palette='coolwarm')
plt.title("Correlation of Features with Target (Status)")
plt.xlabel("Correlation Value")
plt.ylabel("Features")
plt.show()


# 4. PCA explained variance ratio
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance Ratio")
plt.grid(True)
plt.show()


# 5. Confusion Matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Alive (0)', 'Dead (1)'], yticklabels=['Alive (0)', 'Dead (1)'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 6. Feature Importance from Random Forest
plt.figure(figsize=(12,6))
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = x.columns

sns.barplot(x=importances[indices][:20], y=np.array(features)[indices][:20], palette="viridis")
plt.title("Top 20 Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()


# 7. Cross-validation scores visualization
plt.figure(figsize=(8,5))
sns.boxplot(scores, color="lightblue")
sns.stripplot(scores, color="red", jitter=True, size=8)
plt.title("Cross-Validation Accuracy Distribution")
plt.ylabel("Accuracy")
plt.show()

plt.figure(figsize=(8,5))
plt.plot(range(1, len(scores)+1), scores, marker='o', linestyle='-', color='green')
plt.title("Cross-Validation Scores Across Folds")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()