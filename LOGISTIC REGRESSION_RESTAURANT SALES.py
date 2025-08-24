#THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- LOGISTIC REGRESSOR 
#SOURCE :- KAGGLE ( https://www.kaggle.com/datasets/rohitgrewal/restaurant-sales-data )



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


#IMPORTING THE DATASET 
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\Sales.csv")
print(a.columns.tolist())


#CONVERTED THE 'DATE' COLUMN TO DATETIME FORMAT (HANDLING ERRORS), THEN EXTRACTED THE YEAR AND MONTH INTO SEPARATE COLUMNS.
print()
a['Date'] = pd.to_datetime(a['Date'], dayfirst = True, errors = 'coerce')

a['Year'] = a['Date'].dt.year
a['Month'] = a['Date'].dt.month


#DROPING THE UNREQUIRED FEATURE
print()
a = a.drop(['Date', 'Order ID'], axis = 1)


#INFORMATION ABOUT THE DATA
print()
print(a.info())


#DESCRIBING THE DATA
print()
print(a.describe())

print()
print(a.head(10))


#TO FIND TOTAL NUMBER OF DUPLICATE VALUES
print()
print("TOTAL DUPLICATE VALUES ")
print(a.duplicated().sum())


#DISPLATED DUPLICATED ROWS
print()
c = a[a.duplicated()]
print(c)


#DROP THE DUPLICATE
a = a.drop_duplicates()


#TO SEARCH FOR MISSING VALUES
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#HANDLING THE MISSING VALUES
print()
for col in ['Year', 'Month']:
    a[col] = a[col].fillna(a[col].mode()[0])


# APPLIED ONE-HOT ENCODING TO SPECIFIED CATEGORICAL COLUMNS AND DROPPED THE FIRST LEVEL TO AVOID DUMMY VARIABLE TRAP
print()
a = pd.get_dummies(a, columns = ['Product','Payment Method', 'Manager', 'City'], drop_first=True)


# APPLIED LABEL ENCODING TO THE TARGET VARIABLE 'PURCHASE TYPE' TO CONVERT IT INTO NUMERIC FORM
le = LabelEncoder()
a['Purchase Type'] = le.fit_transform(a['Purchase Type'])


#FEATURE SELECTION
print()
x = a.drop('Purchase Type', axis = 1)
y = a['Purchase Type']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components = 0.95)
x_pca = pca.fit_transform(x_scaled)


#SPLITING OF DATA IN TRAIN AND TEST
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state = 1,  stratify=y)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000, C=0.5)
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
acc  =accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

print()
print("ACCURACY SCORE:-\n", acc)
print("CONFUSION MATRIX :-\n", cm)
print("CLASSIFICATION REPORT :-\n", clr)


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

#Distribution of Target Variable (Purchase Type)
plt.figure(figsize=(6,4))
sns.countplot(x='Purchase Type', data=a)
plt.title('Distribution of Purchase Type')
plt.xlabel('Purchase Type')
plt.ylabel('Count')
plt.show()


#Correlation Heatmap (Before PCA)
plt.figure(figsize=(12,8))
sns.heatmap(pd.DataFrame(x_scaled, columns=x.columns).corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap of Features')
plt.show()


#Explained Variance Ratio of PCA Components
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance Ratio')
plt.grid()
plt.show()


#Confusion Matrix Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


#Cross-Validation Scores Boxplot
plt.figure(figsize=(6,4))
sns.boxplot(scores)
plt.title('Cross-Validation Accuracy Scores')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()


