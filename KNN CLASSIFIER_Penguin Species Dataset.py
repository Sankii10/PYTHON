#THE CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM :- KNN CLASSIFIER
#SOURCE :- https://www.kaggle.com/datasets/mubeenshehzadi/penguin-species-dataset



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


#IMPORTING THE DATA FROM CSV FILE
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\penguins.csv")
print(a.columns.tolist())


#DROPPING UNNECESSARY FEATURES
print()
a = a.drop(['Sr.No','year'], axis = 1)


#TOTAL SUM OF DUPLICATED VALUES
print()
print("TOTAL DUPLICATED VALUES IN DATA")
print(a.duplicated().sum())


#DISPLAYING DUPLICATED VALUES
print()
c = a[a.duplicated()]
print(c)


#TO FIND THE MISSING VALUES
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#HANDLING MISSING VALUES 
print()
a['sex'] = a['sex'].fillna(a['sex'].mode()[0])

for col in ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']:
    a[col] = a[col].fillna(a[col].mean())



#SEGREGATION THE CATEGORICAL AND CONTINOUS VARIABLES
print()
cat = []
con = []

for i in a:
    if a[i].dtypes == 'object':
        cat.append(i)
    else:
        con.append(i)

print()
print("CATEGORICAL VALUES :-\n", cat)
print("CONTINOUS VALUES :-\n", con)


#ONE HOT ENCODING AND LABEL ENCODING ON TARGET VARIABLE
print()
a = pd.get_dummies(a, columns = ['island', 'sex'] , drop_first=True)

le = LabelEncoder()
a['species'] = le.fit_transform(a['species'])


# PRINT THE NORMALIZED VALUE COUNTS (PROPORTION) OF EACH SPECIES IN THE DATASET
print()
print(a['species'].value_counts(normalize = True))


# PRINT THE CORRELATION OF ALL NUMERIC FEATURES WITH THE TARGET VARIABLE 'SPECIES' IN DESCENDING ORDER
print()
print(a.corr(numeric_only= True)['species'].sort_values(ascending = False))


#FEATURE SELECTION
print()
x = a.drop('species', axis = 1)
y = a['species']


#STANDARD SCALER
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLIT OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state= 1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING AND IMPLEMENTATION
print()
model = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='auto', metric='minkowski', p=2, leaf_size=20, n_jobs=-1)
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
print("ACCURACY SCORE :-\n",accuracy_score(y_test, y_pred))
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


# 1️⃣ CORRELATION HEATMAP
plt.figure(figsize=(10,6))
sns.heatmap(a.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("CORRELATION HEATMAP OF FEATURES", fontsize=14)
plt.show()


# 2️⃣ PAIRPLOT TO VISUALIZE RELATIONSHIPS BETWEEN FEATURES
sns.pairplot(a, hue='species', diag_kind='kde')
plt.suptitle("PAIRPLOT OF FEATURES BY SPECIES", y=1.02)
plt.show()


# 3️⃣ DISTRIBUTION OF NUMERIC FEATURES
a[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].hist(
    bins=15, figsize=(10,6), color='lightblue', edgecolor='black'
)
plt.suptitle("DISTRIBUTION OF NUMERIC FEATURES", fontsize=14)
plt.show()


# 4️⃣ COUNT PLOT OF SPECIES (TARGET VARIABLE)
plt.figure(figsize=(6,4))
sns.countplot(x='species', data=a, palette='viridis')
plt.title("DISTRIBUTION OF PENGUIN SPECIES", fontsize=14)
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()


# 5️⃣ CONFUSION MATRIX HEATMAP
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("CONFUSION MATRIX HEATMAP", fontsize=14)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 6️⃣ CROSS VALIDATION SCORES VISUALIZATION
plt.figure(figsize=(8,4))
plt.plot(range(1, 11), scores, marker='o', linestyle='--', color='green')
plt.title("CROSS VALIDATION SCORES (10-FOLD)", fontsize=14)
plt.xlabel("Fold Number")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()


# 7️⃣ COMPARISON BAR CHART (ACTUAL VS PREDICTED)
comparison_sample = comparison.head(30)  # limit to 30 for clarity
plt.figure(figsize=(12,5))
plt.plot(comparison_sample.index, comparison_sample['Actual'], marker='o', label='Actual', color='blue')
plt.plot(comparison_sample.index, comparison_sample['Predict'], marker='s', label='Predicted', color='red')
plt.title("ACTUAL VS PREDICTED SPECIES (SAMPLE)", fontsize=14)
plt.xlabel("Sample Index")
plt.ylabel("Encoded Species")
plt.legend()
plt.grid(True)
plt.show()