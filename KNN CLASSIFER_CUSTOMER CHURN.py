#EXECUTED BY :- SANKET GAIKWAD
#ALGORITHM USED :- KNN CLASSIFIER



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


#IMPORT THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\MARKETING ANALYTICS\\Dataset\\abc1.csv")
print(a.columns.tolist())


#CREATION OF NEW VARIABLE AS CHURN TO ACT AS TARGET VARIABLE
print()
a['churn'] = ((a['IsActiveMember']== 0) & (a['Tenure'] < 3)&(a['Balance'] > 0)).astype(int)


#DROPPING UNREQUIRED FEATURES
print()
a = a.drop(['id', 'CustomerId', 'Surname'], axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION OF DATA
print()
print(a.info())

print()
print(a.describe())


#SUM OF DUPLICATED VALUES IN DATA
print()
print("TOTAL SUM OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())


#DISPLAY OF DUPLICATED VALUES AS PER ROW
print()
c = a[a.duplicated()]
print(c)


#DROPPING OF DUPLICATED VALUES 
print()
a = a.drop_duplicates()


#SEARCHING FOR MISSING VALUES IN DATA
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
print("CATRGORICAL COLUMNS :-\n", cat)
print("CONTINOUS COLUMNS :-\n", con)


#ONE HOT ENCODING
print()
a = pd.get_dummies(a, columns = ['Geography', 'Gender'], drop_first=True)


# PRINTING THE PROPORTION OF EACH CLASS IN THE 'CHURN' COLUMN TO CHECK CLASS DISTRIBUTION
print()
print(a['churn'].value_counts(normalize=True))


# CALCULATING AND SORTING THE CORRELATION OF ALL NUMERIC FEATURES WITH 'CHURN' IN DESCENDING ORDER
print()
print(a.corr(numeric_only= True)['churn'].sort_values(ascending = False))


#FEATURES ELECTION
print()
x = a.drop('churn', axis = 1)
y =a['churn']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca =PCA(n_components= 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLIT OF DATA INTO TRAIN AND TEST
print()
train_size = int(0.8 * len(x_pca))
x_train, x_test = x_pca[:train_size], x_pca[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING AND IMPLEMENTATION
print()
model = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto', metric='minkowski', p=2)
model.fit(x_train_res, y_train_res)

y_pred_prob = model.predict_proba(x_test)[:,1]
y_pred = (y_pred_prob > 0.5).astype(int)
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
kf = KFold(n_splits = 10, shuffle = True, random_state= 1)
scores = cross_val_score(model,x_pca, y, cv = kf, scoring = 'r2')
print(scores)
print("THE MEAN SCORES :-\n", np.mean(scores))


#DATA VISUALISATION

# 1. Churn Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='churn', data=a)
plt.title('Distribution of Churn')
plt.xlabel('Churn (0 = Stay, 1 = Left)')
plt.ylabel('Count')
plt.show()

# 2. Churn Proportion Pie Chart
plt.figure(figsize=(6,6))
a['churn'].value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue','salmon'], explode=[0,0.1])
plt.title('Proportion of Churners vs Non-Churners')
plt.ylabel('')
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(a.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap of Features')
plt.show()

# 4. Top Features vs Churn (Numeric)
numeric_cols = a.select_dtypes(include=np.number).columns.tolist()
numeric_cols.remove('churn')
plt.figure(figsize=(12,6))
a.groupby('churn')[numeric_cols].mean().plot(kind='bar')
plt.title('Average Values of Numeric Features by Churn')
plt.xlabel('Churn (0=Stay,1=Left)')
plt.ylabel('Average Value')
plt.xticks(rotation=0)
plt.show()

# 5. Pairplot for Key Features
key_features = ['CreditScore','Age','Balance','EstimatedSalary','NumOfProducts','churn']
sns.pairplot(a[key_features], hue='churn', palette='Set1')
plt.suptitle('Pairplot of Key Features vs Churn', y=1.02)
plt.show()

# 6. PCA 2D Visualization
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0], x_pca[:,1], c=y, cmap='coolwarm', alpha=0.5)
plt.title('PCA Component 1 vs Component 2 Colored by Churn')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Churn')
plt.show()

# 7. Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()





