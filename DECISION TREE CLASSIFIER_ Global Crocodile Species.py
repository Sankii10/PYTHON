#THE PROGRAM IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- DECISION TREE CLASSIFIER
#SOURCE :- https://www.kaggle.com/datasets/zadafiyabhrami/global-crocodile-species-dataset



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.combine import SMOTETomek


#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\croc.csv")
print(a.columns.tolist())


print()
# RENAMED THE COLUMN 'Date ' TO 'Date' TO REMOVE EXTRA SPACE
a.rename(columns={'Date ':'Date'}, inplace = True)

# CONVERTED THE 'Date' COLUMN TO DATETIME FORMAT, CONSIDERING DAY FIRST, AND COERCING ERRORS TO NaT
a['Date'] = pd.to_datetime(a['Date'], dayfirst = True, errors='coerce')


# EXTRACTED 'YEAR' FROM THE 'Date' COLUMN AND STORED IT IN A NEW COLUMN 'year' AND SAME DONE WITH MONTH
a['year'] = a['Date'].dt.year
a['month'] = a['Date'].dt.month


# MAPPED THE TEXTUAL CONSERVATION STATUS VALUES TO NUMERIC CODES USING THE 'status' DICTIONARY
print()
status = {
    "Least Concern": 0,
    "Near Threatened": 1,
    "Vulnerable": 2,
    "Endangered": 3,
    "Critically Endangered": 4,
    "Data Deficient": 5
}

a['Conservation Status'] = a['Conservation Status'].map(status)


#DROPPING THE UNWANTED FEATURES
print()
a = a.drop(['Date','Observation ID', 'Common Name', 'Scientific Name', 'Family', 'Genus', 'Observer Name', 'Notes'], axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION ABOUT THE DATA
print()
print(a.info())

print()
print(a.describe())


#SEACHING FOR TOTAL DUPLICATED VALUES IN DATA
print()
print("TOTAL DUPLICATED VALUES :-\n")
print(a.duplicated().sum())

#DISPLAYING OFDUPLICATED ROWS
print()
c = a[a.duplicated()]
print(c)


#SEARCING FOR MISSING VALUES IN DATASET 
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#HANDLING THE MISSING VALUES 
print()
for col in ['year', 'month']:
    a[col] = a[col].fillna(a[col].median())


# IDENTIFIED CATEGORICAL AND CONTINUOUS COLUMNS: 'cat' HOLDS OBJECT TYPES, 'con' HOLDS OTHERS
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
a = pd.get_dummies(a, columns = ['Sex', 'Age Class', 'Country/Region', 'Habitat Type'], drop_first = True)


#LABEL ENCODING OF THE TARGET VARIABLE - CONSERVATION STATUS
le = LabelEncoder()
a['Conservation Status'] = le.fit_transform(a['Conservation Status'])


# PRINTED THE NORMALIZED VALUE COUNTS OF 'Conservation Status' TO SHOW PROPORTION OF EACH CLASS
print()
print(a['Conservation Status'].value_counts(normalize= True))

# CALCULATED THE CORRELATION OF ALL NUMERIC COLUMNS WITH 'Conservation Status' AND SORTED IN DESCENDING ORDER
print()
print(a.corr(numeric_only= True)['Conservation Status'].sort_values(ascending = False))


#DROPPING UNWANTED FEATURES
print()
x = a.drop('Conservation Status', axis = 1)
y = a['Conservation Status']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca =PCA(n_components= 0.90)
x_pca = pca.fit_transform(x_scaled)


#SPLIT OF DATA AS TRAIN AND TEST
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state= 1, stratify = y)

print()
smote = SMOTETomek(random_state = 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=5, min_samples_leaf=2, random_state=1)
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION OF DATA
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


#CROSS SCORE VALIDATION 
print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model,x_pca, y, cv  =10)
print(scores)
print("THE MEAN SCORES :-\n", np.mean(scores))


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# DISTRIBUTION OF TARGET VARIABLE
plt.figure(figsize=(8,5))
sns.countplot(x='Conservation Status', data=a, palette='viridis')
plt.title('Distribution of Conservation Status')
plt.xlabel('Conservation Status')
plt.ylabel('Count')
plt.show()

# CORRELATION HEATMAP
plt.figure(figsize=(12,10))
sns.heatmap(a.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# PAIRPLOT OF NUMERIC FEATURES (LIMITED TO FIRST 4 CONTINUOUS FEATURES FOR CLARITY)
sns.pairplot(a[['Observed Length (m)', 'Observed Weight (kg)', 'year', 'month', 'Conservation Status']], 
             hue='Conservation Status', palette='viridis')
plt.show()

# BOXPLOTS OF NUMERIC FEATURES VS TARGET
numeric_cols = ['Observed Length (m)', 'Observed Weight (kg)', 'year', 'month']
for col in numeric_cols:
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Conservation Status', y=col, data=a, palette='coolwarm')
    plt.title(f'{col} vs Conservation Status')
    plt.show()

# DECISION TREE VISUALIZATION
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=[f'PC{i+1}' for i in range(x_pca.shape[1])], 
          class_names=[str(i) for i in sorted(y.unique())], filled=True, fontsize=10)
plt.title('Decision Tree Visualization')
plt.show()

# CONFUSION MATRIX HEATMAP
plt.figure(figsize=(8,6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

