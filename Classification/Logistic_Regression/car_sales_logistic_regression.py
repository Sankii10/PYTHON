#THE FOLLOWING CODE WAS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- LOGISTIC REGRESSION
#SOURCE :- https://www.kaggle.com/datasets/msnbehdani/mock-dataset-of-second-hand-car-sales





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


#IMPORTING THE DATA 
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\carsales.csv")
print(a.columns.tolist())


#CREATED A NEW COLUMN 'Pricesales' BY DIVIDING 'Price' INTO 3 QUANTILE-BASED CATEGORIES: LOW, MEDIUM, HIGH
print()
a['Pricesales'] = pd.qcut(a['Price'], q = 3, labels = ['Low', 'Medium', 'High'])


#SEARCHED FOR NUMBER OF UNIQUES VALUES IN MODEL COLUMN
print(a['Model'].nunique())


#DROPING NOT REQUIRED FEATURE
print()
a = a.drop(['Price'], axis = 1)


#INFORMATION AND DESCRIPTION OF DATA
print()
print(a.info())

print()
print(a.describe())


#SEARCHING FOR TOTAL NUMBER OF DUPLICATES
print()
print("TOTAL SUM OF DUPLICATES :-")
print(a.duplicated().sum())


#DISPLAY OF DUPLICATED ROWS
print()
c = a[a.duplicated()]
print(c)


#DROPING THE DUPLICATED 
print()
a.drop_duplicates()


#SEARCHING FOR THE MISSING VALUES
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUES FOUND !!!")


#ONE HOT ENCODING 
print()
a = pd.get_dummies(a, columns = ['Manufacturer', 'Model','Fuel type'], drop_first=True)


#APPLYING LABEL ENCODING ON TARGET VARIABLE- pricesales
le = LabelEncoder()
a['Pricesales'] = le.fit_transform(a['Pricesales'])


#FEATURES SELECTION
print()
x = a.drop(['Pricesales'], axis = 1)
y = a['Pricesales']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca=pca.fit_transform(x_scaled)


#SPLITING OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state= 1)


print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING 
print()
model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial', random_state=42)
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATIONS
print()
print("ACCURACY SCORE:-\n", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX :-\n", confusion_matrix(y_test, y_pred))
print("CLASSIFICATION RECORDS :_\n", classification_report(y_test, y_pred))


#COMPARISONS
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

# 1. DISTRIBUTION OF TARGET VARIABLE (CLASS BALANCE)
plt.figure(figsize=(6,4))
sns.countplot(x=y, palette="Set2")
plt.title("Distribution of Car Price Categories (Low / Medium / High)")
plt.xlabel("Price Category")
plt.ylabel("Count")
plt.show()


# 2. CONFUSION MATRIX HEATMAP
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 3. EXPLAINED VARIANCE RATIO (PCA)
plt.figure(figsize=(8,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='b')
plt.title("Explained Variance by PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.show()


# 4. PCA SCATTER PLOT (FIRST 2 COMPONENTS)
plt.figure(figsize=(8,6))
sns.scatterplot(x=x_pca[:,0], y=x_pca[:,1], hue=y, palette="Set1", alpha=0.6)
plt.title("PCA Projection (First 2 Components)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Price Category")
plt.show()


# 5. FEATURE DISTRIBUTIONS BEFORE SCALING (ENGINE SIZE, YEAR, MILEAGE)
fig, axes = plt.subplots(1, 3, figsize=(15,5))
sns.histplot(a['Engine size'], kde=True, bins=30, ax=axes[0], color='teal')
axes[0].set_title("Engine Size Distribution")
sns.histplot(a['Year of manufacture'], kde=True, bins=30, ax=axes[1], color='orange')
axes[1].set_title("Year of Manufacture Distribution")
sns.histplot(a['Mileage'], kde=True, bins=30, ax=axes[2], color='purple')
axes[2].set_title("Mileage Distribution")
plt.tight_layout()
plt.show()


# 6. CROSS VALIDATION SCORES DISTRIBUTION
plt.figure(figsize=(6,4))
sns.boxplot(x=scores, color="skyblue")
plt.title("Cross-Validation Accuracy Scores")
plt.xlabel("Accuracy")
plt.show()


