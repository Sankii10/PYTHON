#THE CODE IS EXECUTED BY:- MR.SANKET GAIKWAD
#ALGORITHM USED :- EDA & SVC
#SOURCE :- Shopper's Behavior and Revenue ( https://www.kaggle.com/datasets/subhajournal/shoppers-behavior-and-revenue )



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling  import SMOTE
import warnings as wr
wr.filterwarnings('ignore')

a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\Shop.csv")
print(a.columns.tolist())


print(a.shape)


print()
print(a.describe())

print()
print(a.info())

print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#EXPLORATORY DATA ANALYSIS
#UNIVARIATE
rev_count = a['Revenue'].value_counts()

plt.figure(figsize=(10,7))
plt.bar(rev_count.index, rev_count, color = 'Orange')
plt.title('COUNT PLOT OF REVENUE')
plt.xlabel('Revenue')
plt.ylabel('COUNT')
plt.show()


#KERNEL DENSITY PLOT
sns.set_style('darkgrid')

#This is a Pandas command that selects numerical columns from your DataFrame
numeric_columns = a.select_dtypes(include = ["int64","float64"]).columns

plt.figure(figsize=(40,len(numeric_columns)* 3))
for idx,features in enumerate(numeric_columns, 1):
    plt.subplot(len(numeric_columns),2, idx)
    sns.histplot(a[features],kde = True)
    plt.title(f"{features} | Skewness: {round(a[features].skew(), 2)}")

plt.tight_layout()
plt.show()


#SWARM PLOT 
plt.figure(figsize=(10,2))

sns.swarmplot(x='Revenue', y='PageValues', data=a, palette='viridis')

plt.title("REVENUE VS PAGEVALUES")
plt.xlabel('Revenue')
plt.ylabel('PageValues')

plt.show()


#BIVARIATE
sns.set_palette("Pastel1")

plt.figure(figsize=(10,6))
sns.pairplot(a)
plt.suptitle('PAIR PLOT FOR DATAFRAME')
plt.show()

#MULTIVARIATE
plt.figure(figsize=(15,))

sns.heatmap(a.corr(), annot=True, fmt='.2f',cmap='Pastel1', linewidths= 2)

plt.title('Correlation Heatmap')
plt.show()


#LABEL ENCODING
print()
le = LabelEncoder()
for col in ['Month', 'VisitorType', 'Weekend', 'Revenue']:
    a[col] = le.fit_transform(a[col])


#FEATURE SELECTION
print()
x = a.drop('Revenue', axis = 1)
y = a['Revenue']


#STANDAR SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components = 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLITING OF TRAINING AND TESTING DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size= 0.2, random_state= 1)


print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)



#MODEL BUILDING
print()
model = SVC(kernel='rbf', class_weight='balanced')
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

print()
print("ACCURACY SCORE:-\n", acc)
print("CONFUSION MATRIX:-\n", cm)
print("CLASSIFICATION RECORDS:-\n" , clr)


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model, x_pca, y, cv  =10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))