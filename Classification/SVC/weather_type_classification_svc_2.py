#THE FOLLOWING CODE IS EXECUTED BY MR.SANKET GAIKWAD
#ALGORITHM USED :- SUPPORT VECTOR MACHINE ( SVC)
#SOURCE :- https://www.kaggle.com/datasets/nikhil7280/weather-type-classification




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


#IMPORTING DATA 
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\weather.csv")
print(a.columns.tolist())


#INFORMATION AND STATISTICAL DESCRIPTION OF DATA
print()
print(a.info())

print()
print(a.describe())


#SEARCHING TOTAL DUPLICATED VALUES IN DATA
print()
print("TOTAL SUM OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())


#DISPLAY DUPLICATED VALUES ROWS
print()
c = a[a.duplicated()]
print(c)


#TO SEARCH FOR MISSING VALUES IN DATA
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATSET !!!")



#SEGREGATING THE CATEGORICAL AND CONTINOUS VALUES 
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
a = pd.get_dummies(a, columns = ['Cloud Cover', 'Season', 'Location'], drop_first=True)


#LABEL ENCODING ON TARGET VARIABLE
le = LabelEncoder()
a['Weather Type'] = le.fit_transform(a['Weather Type'])


#CALCULATES THE FREQUENCY OF EACH UNIQUE ROW IN DATAFRAME 'A' AND NORMALIZES IT TO SHOW PROPORTIONS INSTEAD OF RAW COUNTS
print()
print(a.value_counts(normalize= True))

#CALCULATES THE CORRELATION OF ALL NUMERIC COLUMNS WITH 'WEATHER TYPE' AND SORTS THE RESULTS IN DESCENDING ORDER
print()
print(a.corr(numeric_only= True)['Weather Type'].sort_values(ascending = False))


#FEATURE SELECTION
print()
x = a.drop('Weather Type', axis = 1)
y = a['Weather Type']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca = pca.fit_transform(x_scaled)


#SPLIT OF DATA IN TRAIN AND TEST 
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size= 0.2, random_state= 1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING AND EXECUTION
print()
model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=1)
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
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

print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model,x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES :-\n", np.mean(scores))


#DATA VISUALISATION

# 1. DISTRIBUTION OF TARGET VARIABLE
plt.figure(figsize=(8,5))
sns.countplot(x='Weather Type', data=a)
plt.title('Distribution of Weather Types')
plt.xlabel('Weather Type')
plt.ylabel('Count')
plt.show()

# 2. HEATMAP OF CORRELATION MATRIX
plt.figure(figsize=(12,8))
sns.heatmap(a.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# 3. PAIRPLOT OF NUMERIC FEATURES COLORED BY TARGET
numeric_cols = ['Temperature','Humidity','Wind Speed','Precipitation (%)','Atmospheric Pressure','UV Index','Visibility (km)']
sns.pairplot(a[numeric_cols + ['Weather Type']], hue='Weather Type', diag_kind='kde')
plt.suptitle('Pairplot of Numeric Features by Weather Type', y=1.02)
plt.show()

# 4. PCA COMPONENTS VISUALIZATION (2D)
if x_pca.shape[1] >= 2:
    plt.figure(figsize=(8,6))
    plt.scatter(x_pca[:,0], x_pca[:,1], c=y, cmap='rainbow', alpha=0.7)
    plt.title('PCA 2D Projection of Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Weather Type')
    plt.show()

# 5. CONFUSION MATRIX HEATMAP
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 6. FEATURE IMPORTANCE FROM PCA COMPONENTS
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(10,5))
plt.bar(range(len(explained_variance)), explained_variance, alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance of PCA Components')
plt.show()

# 7. CROSS VALIDATION SCORES VISUALIZATION
plt.figure(figsize=(8,5))
plt.plot(range(1, len(scores)+1), scores, marker='o', linestyle='--')
plt.title('10-Fold Cross Validation Scores')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.ylim([0,1])
plt.grid(True)
plt.show()

