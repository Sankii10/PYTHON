#THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- RANDOM FOREST CLASSIFIER
#SOURCE :- https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from datetime import datetime


#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\marketcamp.csv")
print(a.columns.tolist())

# GET CURRENT YEAR AND CALCULATE AGE
print()
current_date = datetime.now().year

a['age'] = current_date - a['Year_Birth']


# CONVERT DT_CUSTOMER TO DATETIME AND EXTRACT YEAR AND MONTH
print()
a['Dt_Customer'] = pd.to_datetime(a['Dt_Customer'], dayfirst = True, errors='coerce')

a['year'] = a['Dt_Customer'].dt.year
a['month'] = a['Dt_Customer'].dt.month


#DROPING UNREQUIRED FEATURES
print()
a = a.drop(['ID', 'Z_CostContact', 'Z_Revenue', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Year_Birth','Dt_Customer'], axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION OF THE DATA
print()
print(a.info())

print()
print(a.describe())


#TOTAL SUM OF DUPLICATED VALUES IN DATA
print()
print("TOTAL DUPLICATED VALUES IN DATA")
print(a.duplicated().sum())


#DUPLICATED VALUES ROW-WISE
print()
c = a[a.duplicated()]
print(c)

#DROPING THE DUPLICATED VALUES
print()
a.drop_duplicates()


#TO FIND THE MISSING VALUES IN DATA
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#HANDLING OF THE MISSING VALUES 
print()
a['Income'] = a['Income'].fillna(a['Income'].median())

print()
for col in ['year','month']:
    a[col] = a[col].fillna(a[col].mode()[0])



#SEGREGATING THE DATA AS CATEGORICAL AND CONTINOUS DATA COLUMNS
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
a = pd.get_dummies(a, columns = ['Education', 'Marital_Status'], drop_first =True)


# SHOWS THE PROPORTION OF EACH CLASS IN RESPONSE
print()
print(a['Response'].value_counts(normalize=True))

# SHOWS CORRELATION OF NUMERIC FEATURES WITH RESPONSE
print()
print(a.corr(numeric_only=True)['Response'].sort_values(ascending= False))


#FEATURES SELECTION
print()
x = a.drop('Response', axis = 1)
y = a['Response']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENTS ANALYSIS
print()
pca = PCA(n_components = 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLITING OF DATA AS TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test  = train_test_split(x_pca, y, test_size = 0.2, random_state = 1)

print()
smote = SMOTE(random_state=1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING AND IMPLEMENTATION
print()
model = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=15, min_samples_split=5, min_samples_leaf=3, random_state=1, n_jobs=-1)
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
print("ACCURACY SCORES:-\n", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX :-\n", confusion_matrix(y_test, y_pred))
print("CLASSIFICATION REPORT :-\n", classification_report(y_test, y_pred))


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores  =cross_val_score(model,x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES :-\n", np.mean(scores))


#DATA VISUALISATION

# 1. RESPONSE CLASS DISTRIBUTION
plt.figure(figsize=(6,4))
sns.countplot(x='Response', data=a)
plt.title('Response Class Distribution')
plt.xlabel('Response')
plt.ylabel('Count')
plt.show()

# 2. CORRELATION HEATMAP
plt.figure(figsize=(14,10))
sns.heatmap(a.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap of Numeric Features')
plt.show()


# 3. AGE DISTRIBUTION
plt.figure(figsize=(8,4))
sns.histplot(a['age'], bins=20, kde=True, color='skyblue')
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 4. INCOME DISTRIBUTION
plt.figure(figsize=(8,4))
sns.histplot(a['Income'], bins=20, kde=True, color='orange')
plt.title('Income Distribution')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.show()

# 5. RELATIONSHIP BETWEEN AGE AND MNTWINES (Spending on Wine)
plt.figure(figsize=(8,4))
sns.scatterplot(x='age', y='MntWines', hue='Response', data=a, palette='Set1')
plt.title('Age vs Spending on Wines by Response')
plt.xlabel('Age')
plt.ylabel('MntWines')
plt.show()

# 6. BOXPLOT OF MNTWINES BY RESPONSE
plt.figure(figsize=(6,4))
sns.boxplot(x='Response', y='MntWines', data=a)
plt.title('Boxplot of Wine Spending by Response')
plt.show()

# 8. PAIRPLOT OF SELECTED FEATURES
selected_features = ['MntWines', 'MntMeatProducts', 'MntFruits', 'Income', 'Response']
sns.pairplot(a[selected_features], hue='Response', palette='Set2')
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.show()




