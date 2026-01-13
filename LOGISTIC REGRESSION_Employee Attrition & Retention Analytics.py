#THIS CODE IS EXECUTED BY MR.SANKET GAIKWAD
#ALGORITHM USED :- LOGISTIC REGRESSION
#SOURCE :- https://www.kaggle.com/datasets/ajinkyachintawar/employee-attrition-and-retention-analytics-dataset


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

print()
a = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\Desktop\\My Data\\HR.csv")
print(a.columns.tolist())

a = a[~a['Age'].astype(str).str.startswith('#')]
a['Age'] = pd.to_numeric(a['Age'], errors='coerce')
a['Attrition'] = a['Attrition'].astype(str)
a = a.dropna()


#DROPING UNREQUIRED FEATURES
print()
a = a.drop(['EmployeeCount', 'EmployeeNumber','MonthlyRate','Over18','StandardHours'], axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION OF DATA
print()
print(a.info())

print()
print(a.describe())


#SEARCHING OF DUPLICATED VALUES
print()
print("DUPLICATED VALUES :-\n")
print(a.duplicated().sum())

print()
c = a[a.duplicated()]
print(c)


#SEARCHING FOR MISSING VALUES
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#SEGREGATION OF CATEGORICAL AND CONTINOUS VALUES
print()
cat = []
con = []

for i in a:
    if a[i].dtype == 'object':
        cat.append(i)
    else:
        con.append(i)

print()
print("CATEGORICAL VALUES :-\n", cat)
print("CONTINOUS VALUES :-\n", con)


#REPLACING THE MISSING VALUES WITH MODE
print()
for col in ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']:
    a[col] = a[col].fillna(a[col].mode()[0])


#REPLACING THE MISSING VALUES WITH MEDIAN
for col in ['DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate',
            'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
            'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
            'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
            'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']:
    a[col] = a[col].fillna(a[col].median())


#ONE HOT ENCODING
print()
a = pd.get_dummies(a,columns=['Age','BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime'],drop_first=True)


#LABEL ENCODING OF TARGET VARIABLE
print()
le = LabelEncoder()
a['Attrition'] = le.fit_transform(a['Attrition'])


#CALCULATES THE PROPORTION (PERCENTAGE DISTRIBUTION) OF EACH CLASS IN THE ATTRITION TARGET VARIABLE TO CHECK CLASS IMBALANCE
print(a['Attrition'].value_counts(normalize=True))
print()
print(a['Attrition'].value_counts(normalize=True))


#COMPUTES THE CORRELATION OF ALL NUMERIC FEATURES WITH ATTRITION AND SORTS THEM TO IDENTIFY THE MOST INFLUENTIAL VARIABLES
print()
print(a.corr(numeric_only=True)['Attrition'].sort_values(ascending=False))

#FEATURE SELECTION
x = a.drop('Attrition', axis=1)
y = a['Attrition']


#SPLIT OF TRAIN AND TEST DATA
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)


#STANDARD SCALING
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
pca = PCA(n_components=0.95)
x_pca = pca.fit_transform(x_scaled)


#MODEL BUILDING AND IMPLEMENTATION
print()
model = LogisticRegression(solver='liblinear',penalty='l2',C=0.5,max_iter=2000,tol=1e-4,random_state=1)
model.fit(x_train, y_train)

y_pred_prob = model.predict_proba(x_test)[:,1]
y_pred = (y_pred_prob >= 0.5).astype(int)
print(y_pred)


#EVALUATION 
print()
print("EVALUATION")
print("ACCURACY SCORE :-\n", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX :-\n", confusion_matrix(y_test, y_pred))
print("CLASSIFICATION REPORT :-\n", classification_report(y_test, y_pred))


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)


#CROSS SCORE VALIDATION
print()
print("CROSS SCORE VALIDATION")
kf = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(model, x_pca, y, cv=kf, scoring='accuracy')
print(scores)
print("MEAN SCORE :-\n", np.mean(scores))


# CLASS DISTRIBUTION OF ATTRITION TARGET VARIABLE
plt.figure(figsize=(6,4))
sns.countplot(x=a['Attrition'])
plt.title("ATTRITION CLASS DISTRIBUTION")
plt.xlabel("ATTRITION")
plt.ylabel("COUNT")
plt.show()


# AGE DISTRIBUTION OF EMPLOYEES
plt.figure(figsize=(6,4))
sns.histplot(pd.to_numeric(a.filter(like='Age_').idxmax(axis=1).str.replace('Age_','')), bins=20, kde=True)
plt.title("AGE DISTRIBUTION")
plt.xlabel("AGE")
plt.ylabel("FREQUENCY")
plt.show()


# ATTRITION VS OVERTIME
plt.figure(figsize=(6,4))
sns.countplot(x=a['Attrition'], hue=a.filter(like='OverTime_').idxmax(axis=1))
plt.title("ATTRITION VS OVERTIME")
plt.xlabel("ATTRITION")
plt.ylabel("COUNT")
plt.show()


# MONTHLY INCOME DISTRIBUTION BY ATTRITION
plt.figure(figsize=(6,4))
sns.boxplot(x=a['Attrition'], y=a['MonthlyIncome'])
plt.title("MONTHLY INCOME BY ATTRITION")
plt.xlabel("ATTRITION")
plt.ylabel("MONTHLY INCOME")
plt.show()


# YEARS AT COMPANY VS ATTRITION
plt.figure(figsize=(6,4))
sns.boxplot(x=a['Attrition'], y=a['YearsAtCompany'])
plt.title("YEARS AT COMPANY BY ATTRITION")
plt.xlabel("ATTRITION")
plt.ylabel("YEARS AT COMPANY")
plt.show()


# JOB SATISFACTION VS ATTRITION
plt.figure(figsize=(6,4))
sns.countplot(x=a['JobSatisfaction'], hue=a['Attrition'])
plt.title("JOB SATISFACTION VS ATTRITION")
plt.xlabel("JOB SATISFACTION")
plt.ylabel("COUNT")
plt.show()


# CORRELATION HEATMAP OF NUMERIC FEATURES
plt.figure(figsize=(12,8))
sns.heatmap(a.corr(numeric_only=True), cmap='coolwarm', annot=False)
plt.title("CORRELATION HEATMAP")
plt.show()


# CONFUSION MATRIX VISUALIZATION
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("CONFUSION MATRIX")
plt.xlabel("PREDICTED")
plt.ylabel("ACTUAL")
plt.show()


# TOP FEATURES CORRELATED WITH ATTRITION
top_corr = a.corr(numeric_only=True)['Attrition'].sort_values(ascending=False)[1:11]
plt.figure(figsize=(6,4))
top_corr.plot(kind='bar')
plt.title("TOP FEATURES CORRELATED WITH ATTRITION")
plt.xlabel("FEATURES")
plt.ylabel("CORRELATION VALUE")
plt.show()