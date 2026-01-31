#THE CODE IS EXECUTED BY MR.SANKET GAIKWAD
#ALGORITHM :- LOGISTIC REGRESSION
#SOURCE :- https://www.kaggle.com/datasets/thedrzee/customer-churn-in-telecom-sample-dataset-by-ibm


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline


#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\Desktop\\My Data\\Tele.csv")
print(a.columns.tolist())


#DROPING THE UNREQUIRED FEATURE
print()
a = a.drop('customerID', axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION
print()
print(a.info())

print()
print(a.describe())

#SEARCHING THE DUPLICATED VALUES
print()
print("DUPLICATED VALUES")
print(a.duplicated().sum())

print()
c = a[a.duplicated()]
print(c)

print()
a = a.drop_duplicates()


#SEARCHING FOR MISSING VALUES
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!")


#SEGREGATION OF CATEGORICAL AND CONTINOUS VALUES
print()
cat = []
con = []

for i in a:
    if a[i].dtypes == 'object':
        cat.append(i)
    else:
        con.append(i)

print()
print("CATEGORICAL VALUES:-\n",cat)
print("CONTINOUS VALUES :-\n", con)


#ONE HOT ENCODING
print()
a = pd.get_dummies(a, columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'TotalCharges'], drop_first=True)


#TARGET VARIABLE LABEL ENCODING
le = LabelEncoder()
a['Churn']=le.fit_transform(a['Churn'])


#FEATURE SELECTION
print()
x = a.drop('Churn', axis = 1)
y = a['Churn']


# CALCULATES AND DISPLAYS THE PROPORTION (PERCENTAGE) OF EACH CLASS IN THE CHURN COLUMN
print()
print(a['Churn'].value_counts(normalize = True))


# COMPUTES AND SORTS THE CORRELATION OF ALL NUMERIC FEATURES WITH THE CHURN TARGET VARIABLE
print()
print(a.corr(numeric_only=True)['Churn'].sort_values(ascending = True))


#SPLIT OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)


#STANDARD SCALING
print()
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components=0.95)
x_train_pca = pca.fit_transform(x_train_sc)
x_test_pca =pca.transform(x_test_sc)


#MODEL BUILDING AND IMPLEMENTATION
print()
model = LogisticRegression(penalty='l2', C=0.5, solver='liblinear', class_weight='balanced', max_iter=1000, random_state=1)
model.fit(x_train_pca, y_train)

y_pred_prob = model.predict_proba(x_test_pca)[:,1]
y_pred = (y_pred_prob > 0.5).astype(int)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
print("ACCURACY SCORE :-\n", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIC :-\n", confusion_matrix(y_test, y_pred))
print("CLASSIFICATION SCORE :-\n", classification_report(y_test, y_pred))

print()

#COMPARISON
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, "predict":y_pred})
print(comparison)


#CROSS SCORE VALIDATION
print()
print("CROSS SCORE VALIDATION")
pipe = Pipeline([
    ('Scaler', StandardScaler()),
    ('PCA', PCA(n_components=0.95)),
    ('model',LogisticRegression(penalty='l2', C=0.5, solver='liblinear', class_weight='balanced', max_iter=1000, random_state=1))
])

cv = StratifiedKFold(n_splits=10, shuffle = True, random_state=1)

scores  =cross_val_score(pipe, x,y, cv= cv, scoring = 'accuracy')
print("SCORES :-\n", scores)
print("MEAN SCORES :-\n", np.mean(scores))


#DATA VISUALISATION
print()
print("DATA VISUALISATION")

plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=a)
plt.title("CHURN CLASS DISTRIBUTION")
plt.show()


plt.figure(figsize=(8,5))
sns.histplot(a['tenure'], bins=30, kde=True)
plt.title("TENURE DISTRIBUTION OF CUSTOMERS")
plt.show()


plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=a)
plt.title("MONTHLY CHARGES VS CHURN")
plt.show()


plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='tenure', data=a)
plt.title("TENURE VS CHURN")
plt.show()


plt.figure(figsize=(12,6))
corr_matrix = a.corr(numeric_only=True)
sns.heatmap(corr_matrix, cmap='coolwarm', linewidths=0.5)
plt.title("CORRELATION HEATMAP")
plt.show()


plt.figure(figsize=(6,4))
sns.countplot(x='Contract_Month-to-month', hue='Churn', data=a)
plt.title("MONTH TO MONTH CONTRACT VS CHURN")
plt.show()


plt.figure(figsize=(6,4))
sns.countplot(x='PaymentMethod_Electronic check', hue='Churn', data=a)
plt.title("ELECTRONIC CHECK PAYMENT VS CHURN")
plt.show()


plt.figure(figsize=(6,4))
sns.countplot(x='TechSupport_Yes', hue='Churn', data=a)
plt.title("TECH SUPPORT VS CHURN")
plt.show()
