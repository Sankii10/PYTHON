#THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- LOGISTIC REGRESSION
#SOURCE :- https://www.kaggle.com/datasets/mubeenshehzadi/customer-churn-dataset




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
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\Telco.csv")
print(a.columns.tolist())


#DROPING UNREQUIRED FEATURES
print()
a = a.drop(['customerID', 'TotalCharges'], axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION OF THE DATA 
print()
print(a.info())

print()
print(a.describe())


#SEARCHING FOR DUPLICATED VALUES TOTAL
print()
print("TOTAL SUM OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())


#DISPLAY OF DUPLICATED VALUES ROWS
print()
c = a[a.duplicated()]
print(c)

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
    print("NO SUCH MISSING VALUE FOUND IN DATSET !!!")


#SEGREGATTION OF CATEGORICAL VALUES AND CONTINOUS VALUES
print()
cat = []
con = []

for i in a.columns:
    if a[i].dtypes == 'object':
        cat.append(i)
    else:
        con.append(i)

print()
print("CATEGORICAL COLUMNS :-\n", cat)
print("CONTINOUS VALUES :-\n", con)


#ONE HOT ENCODING 
print()
a = pd.get_dummies(a, columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'], drop_first=True)


#LABEL ENCODING ON TAGET VARIABLE
le = LabelEncoder()
a['Churn'] = le.fit_transform(a['Churn'])


# SHOW THE PROPORTION OF CUSTOMERS WHO CHURNED VS. RETAINED
print()
print(a['Churn'].value_counts(normalize=True))


# SHOW CORRELATION OF ALL NUMERIC FEATURES WITH THE TARGET 'CHURN', SORTED DESCENDING
print()
print(a.corr(numeric_only=True)['Churn'].sort_values(ascending = False))


#FEATURE SELECTION
print()
x = a.drop('Churn', axis = 1)
y = a['Churn']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled  =sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLIT OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state = 1)

print()
smote = SMOTE(random_state=1)
x_train_res , y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING AND EXECUTION
print()
model = LogisticRegression(solver='liblinear', penalty='l2', C=10.0, max_iter=2000, random_state=1)
model.fit(x_train_res, y_train_res)

# PREDICTION PROBABILITIES INSTEAD OF DEFAULT THRESHOLD 0.5
y_pred_proba = model.predict_proba(x_test)[:,1]

# ADJUST THRESHOLD TO IMPROVE RECALL OR PRECISION (EXAMPLE THRESHOLD = 0.4)
y_pred = (y_pred_proba >= 0.4).astype(int)
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
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALISATION

# 1. CHURN DISTRIBUTION
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=a)
plt.title('DISTRIBUTION OF CHURN')
plt.xlabel('Churn')
plt.ylabel('COUNT')
plt.show()

# 2. MONTHLY CHARGES DISTRIBUTION BY CHURN
plt.figure(figsize=(8,5))
sns.histplot(data=a, x='MonthlyCharges', hue='Churn', kde=True, bins=30)
plt.title('MONTHLY CHARGES DISTRIBUTION BY CHURN')
plt.xlabel('Monthly Charges')
plt.ylabel('Frequency')
plt.show()

# 3. TENURE DISTRIBUTION BY CHURN
plt.figure(figsize=(8,5))
sns.histplot(data=a, x='tenure', hue='Churn', kde=True, bins=30)
plt.title('TENURE DISTRIBUTION BY CHURN')
plt.xlabel('Tenure (Months)')
plt.ylabel('Frequency')
plt.show()

# 4. CORRELATION HEATMAP
plt.figure(figsize=(12,10))
sns.heatmap(a.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('CORRELATION HEATMAP OF FEATURES')
plt.show()

# 5. CONTRACT TYPE VS CHURN
plt.figure(figsize=(6,4))
sns.countplot(x='Contract', hue='Churn', data=pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\Telco.csv"))
plt.title('CONTRACT TYPE VS CHURN')
plt.xlabel('Contract Type')
plt.ylabel('COUNT')
plt.show()

# 6. INTERNET SERVICE VS CHURN
plt.figure(figsize=(6,4))
sns.countplot(x='InternetService', hue='Churn', data=pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\Telco.csv"))
plt.title('INTERNET SERVICE TYPE VS CHURN')
plt.xlabel('Internet Service')
plt.ylabel('COUNT')
plt.show()

# 7. HEATMAP OF PCA COMPONENTS (optional, for feature importance visualization)
plt.figure(figsize=(10,6))
sns.heatmap(pd.DataFrame(x_pca).corr(), cmap='coolwarm')
plt.title('CORRELATION HEATMAP OF PCA COMPONENTS')
plt.show()