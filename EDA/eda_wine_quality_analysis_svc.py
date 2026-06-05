#THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM :- EXPLORATORY DATA ANALYSIS(EDA) AND SUPPORT VECTOR MACHINE(SVC)
#SOURCE :- https://www.geeksforgeeks.org/data-analysis/exploratory-data-analysis-in-python/


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import warnings as wr
wr.filterwarnings('ignore')


#IMPORT THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\WineQT.csv")
print(a.columns.tolist())


#DROP THE UNWANTED COLUMNS
print()
a = a.drop('Id', axis = 1)


#INFORMATION OF DATA
print()
print(a.info())


#EXPLORATORY DATA ANALYSIS

print(a.nunique())

#Univariate Analysis
qualitycounts = a['quality'].value_counts()

plt.figure(figsize = (8,6))
plt.bar(qualitycounts.index, qualitycounts, color = 'Orange')
plt.title("COUNT PLOT OF QUALITY")
plt.xlabel('ALCOHOL')
plt.ylabel("COUNT")
plt.show()


#Kernel density plot for understanding variance in the dataset
sns.set_style("darkgrid")

numcolms = a.select_dtypes(include={'int64', 'float64'}).columns

plt.figure(figsize=(14, len(numcolms) * 3))
for idx, feature in enumerate(numcolms, 1):
    plt.subplot(len(numcolms), 2, idx)
    sns.histplot(a[feature], kde=True)
    plt.title(f"{feature} | Skewness: {round(a[feature].skew(), 2)}")

plt.tight_layout()
plt.show()


#Swarm Plot for showing the outlier in the data
plt.figure(figsize=(10, 8))

sns.swarmplot(x="quality", y="alcohol", data=a, palette='viridis')

plt.title('Swarm Plot for Quality and Alcohol')
plt.xlabel('Quality')
plt.ylabel('Alcohol')
plt.show()


#Bivariate Analysis
sns.set_palette("Pastel1")

plt.figure(figsize=(10, 6))

sns.pairplot(a)

plt.suptitle('Pair Plot for DataFrame')
plt.show()


# Box Plot for examining the relationship between alcohol and Quality
sns.boxplot(x='quality', y='alcohol', data=a)


#Multivariate Analysis
plt.figure(figsize=(15, 10))

sns.heatmap(a.corr(), annot=True, fmt='.2f', cmap='Pastel2', linewidths=2)

plt.title('Correlation Heatmap')
plt.show()

# SEARCHING MISSING VALUES
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#DESCRIBE THE DATA
print()
print(a['quality'].describe())

#A new column qualitylabel by labeling wines as 'good' if their quality score is 6 or higher, otherwise 'bad'.
print()
a['qualitylabel'] = a['quality'].apply(lambda x:'good' if x>=6 else 'bad')

print()
print(a['qualitylabel'].value_counts())


#LABEL ENCODER
print()
le = LabelEncoder()
a['qualitylabel'] = le.fit_transform(a['qualitylabel'])


#FEATURE SELECTION
print()
x = a.drop(['quality','qualitylabel'], axis = 1)
y = a['qualitylabel']


#STANDARD SCALER
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components =  0.95)
x_pca = pca.fit_transform(x_scaled)


#SPLITTING OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=  0.2, random_state = 1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model = SVC(kernel = 'rbf', class_weight='balanced')
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION OF MODEL
print()
print("EVALUATION")
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

print()
print("ACCURACY SCORE:-\n", acc)
print("CONFUSION MATRISC:-\n", cm)
print("CLASSIFICATION REPORT :-\n", clr)


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)


print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model, x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALIZATION

#Count Plot – Distribution of 'good' vs 'bad' quality wine:
plt.figure(figsize=(6,4))
sns.countplot(x='qualitylabel', data=a)
plt.title("Distribution of Wine Quality Labels")
plt.xlabel("Quality Label (0 = bad, 1 = good)")
plt.ylabel("Count")
plt.show()


#Heatmap – Correlation Matrix of original features:
plt.figure(figsize=(10,8))
sns.heatmap(a.drop('qualitylabel', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()


#Explained Variance by PCA Components:
plt.figure(figsize=(8,4))
explained_variance = pca.explained_variance_ratio_
plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.7, align='center')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Component Index')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.show()


#PCA Scatter Plot – Visualizing classes in first 2 principal components:
plt.figure(figsize=(8,6))
sns.scatterplot(x=x_pca[:,0], y=x_pca[:,1], hue=a['qualitylabel'], palette=['red','green'], alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection (2D) of Wine Data')
plt.legend(title='Quality Label')
plt.grid(True)
plt.show()


#Confusion Matrix Heatmap:
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad','Good'], yticklabels=['Bad','Good'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

