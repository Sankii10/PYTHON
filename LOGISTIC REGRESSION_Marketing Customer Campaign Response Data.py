#THIS CODE IS EXECUTED BY :- SANKET GAIKWAD
#ALGORTIHM USED :- LOGISTIC REGRESSION
#SOURCE :- https://www.opendatabay.com/data/ai-ml/18348d1d-7b4f-4e6b-8da7-126b86475b13

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


# Load data
a = pd.read_csv("C:\\Users\\ASUS\\Downloads\\MAProjectData.csv")
print(a.columns.tolist())

# Convert date and create tenure
a['Dt_Customer'] = pd.to_datetime(a['Dt_Customer'], utc = True)
max_date = a['Dt_Customer'].max()
a['Customer_Tenure'] = (max_date - a['Dt_Customer']).dt.days

print()
a['Total_Spend'] = (a['MntWines'] + a['MntFruits'] + a['MntMeatProducts'] + a['MntFishProducts'] + a['MntSweetProducts'] + a['MntGoldProds'])

# Convert Total_Spend into Spend_Class
a['Spend_Class'] = pd.qcut(a['Total_Spend'], q=3, labels=['Low','Medium','High'])

#Dropping the unrequired features
print()
a = a.drop(['Unnamed: 0', 'ID', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue','AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'Complain','MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds','Total_Spend'], axis = 1)


#Information and Statistical description of data 
print()
print(a.info())

print()
print(a.describe())

#Searching for duplicated values in dataset
print()
print("TOTAL NUMBER OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())

#display of duplicated values
print()
c = a[a.duplicated()]
print(c)

#dropping of duplicated values
print()
a = a.drop_duplicates()

#Searching for missing values in data
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#Replacing the missing value with median
print()
a['Income'] = a['Income'].fillna(a['Income'].median())


#segregating the categorical and continous column
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


#One hot encoding
print()
a = pd.get_dummies(a,columns = ['Education', 'Marital_Status'], drop_first=True)

#Target variable feature selection
print()
x = a.drop('Spend_Class', axis = 1)
y = a['Spend_Class']

#Standard Scaling
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

#Principal Component Analysis
pca =PCA(n_components=0.95)
x_pca = pca.fit_transform(x_scaled)

#Split of train and test dataset 
print()
x_train, x_test, y_train, y_test =train_test_split(x_pca, y, test_size=0.2, random_state= 1)


print()
smote = SMOTE(random_state=1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

#Model Building and Implementation
print("**** LOGISTIC REGRESSION MODEL ****")
print()
model = LogisticRegression(max_iter=500)
model.fit(x_train_res, y_train_res)

y_pred_prob_lr = model.predict_proba(x_test)
y_pred_prob_lr = pd.DataFrame(y_pred_prob_lr, columns = model.classes_)
pd.options.display.float_format = '{:.3f}'.format
print(y_pred_prob_lr)


print()
y_pred_lr = model.predict(x_test)
print(y_pred_lr)

#Evaluation
print()
print("EVALUATION FOR LOGISTIC REGRESSION")
print("ACCURACY SCORE :-\n", accuracy_score(y_test, y_pred_lr))
print("CONFUSION MATRIX :-\n", confusion_matrix(y_test, y_pred_lr))
print("CLASSIFICATION REPORT :-\n", classification_report(y_test, y_pred_lr))


#Comparison
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred_lr})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
kf = KFold(n_splits = 10, shuffle = True, random_state= 1)
scores = cross_val_score(model,x_pca, y, cv = kf, scoring = 'accuracy')
print(scores)
print("MEAN SCORES:-\n", np.mean(scores))


#Data Visualisation

# Confusion Matrix Heatmap
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()


# Class Distribution Before and After SMOTE
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
sns.countplot(x=y, palette='viridis')
plt.title("Original Class Distribution")
plt.xlabel("Spend Class")
plt.ylabel("Count")

plt.subplot(1,2,2)
sns.countplot(x=y_train_res, palette='viridis')
plt.title("SMOTE Class Distribution")
plt.xlabel("Spend Class")
plt.ylabel("Count")

plt.tight_layout()
plt.show()


# PCA Explained Variance
pca_var = PCA().fit(x_scaled)
plt.figure(figsize=(6,4))
plt.plot(np.cumsum(pca_var.explained_variance_ratio_), marker='o')
plt.title("Explained Variance by PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.axhline(y=0.95, color='r', linestyle='--')
plt.grid(True)
plt.show()


# Probability Distribution of Predictions
plt.figure(figsize=(8,4))
plt.hist(y_pred_prob_lr.max(axis=1), bins=20)
plt.title("Prediction Confidence Distribution")
plt.xlabel("Max Probability of Predicted Class")
plt.ylabel("Frequency")
plt.show()


# Actual vs Predicted Count
plt.figure(figsize=(6,4))
comparison['Predict'].value_counts().plot(kind='bar', alpha=0.7, label='Predicted')
y_test.value_counts().plot(kind='bar', alpha=0.7, label='Actual')
plt.title("Actual vs Predicted Class Count")
plt.xlabel("Spend Class")
plt.ylabel("Count")
plt.legend()
plt.show()

