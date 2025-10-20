#THE FOLLOWING CODE IS EXECUTED BY : MR.SANKET GAIKWAD
#ALGORITHM USED :- DECISION TREE CLASSIFIER
#SOURCE :- https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score
from imblearn.over_sampling import SMOTE


#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\diabetis.csv")
print(a.columns.tolist())

#INFORMATION AND STATISTICAL DESCRIPTION OF DATA
print()
print(a.info())

print()
print(a.describe())


#FINDING THE TOTAL SUM OF DUPLICATED VALUES IN DATA
print()
print("TOTAL SUM OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())

#DISPLAYING THE DUPLICATED ROWS VALUES
print()
c = a[a.duplicated()]
print(c)


#DROPING THE DUPLICATED VALUES 
a = a.drop_duplicates()


#SEARCHING FOR THE MISSING VALUES IN DATASET 
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#SEGREGATING THE CATEGORICAL AND CONTINOUS VALUES 
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


#ONE HOT ENCODING
print()
a = pd.get_dummies(a, columns = ['gender', 'smoking_history'], drop_first=True)


# SHOW THE PROPORTION (PERCENTAGE) OF EACH CLASS IN THE 'DIABETES' COLUMN
print()
print(a['diabetes'].value_counts(normalize=True))

# DISPLAY CORRELATION OF ALL NUMERIC FEATURES WITH THE 'DIABETES' COLUMN IN DESCENDING ORDER
print()
print(a.corr(numeric_only=True)['diabetes'].sort_values(ascending=False))

#FEATURES SELECTION
print()
x = a.drop('diabetes', axis = 1)
y =a['diabetes']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca =PCA(n_components=0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLITING OF DATA AS TRAIN AND TEST DATA
print()
train_size = int(0.8 * len(x_pca))
x_train, x_test = x_pca[:train_size], x_pca[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING AND IMPLEMENTATION
print()
model = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=5, min_samples_leaf=2, splitter='best', max_features=None, random_state=42)
model.fit(x_train_res, y_train_res)

y_pred_prob  = model.predict_proba(x_test)[:,1]
y_pred = (y_pred_prob > 0.5).astype(int)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
print("ACCURACY SCORE:-\n", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX :-\n", confusion_matrix(y_test, y_pred))
print("CLASSIFICATION REPORT :-\n", classification_report(y_test, y_pred))
print("RECALL SCORE :-\n", recall_score(y_test, y_pred))
print("PRECISION SCORE:-\n", precision_score(y_test, y_pred))


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

# CORRELATION HEATMAP TO VISUALIZE RELATIONSHIP BETWEEN FEATURES
plt.figure(figsize=(12,8))
sns.heatmap(a.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("CORRELATION HEATMAP")
plt.show()

# COUNT PLOT FOR DIABETES DISTRIBUTION
plt.figure(figsize=(6,4))
sns.countplot(x='diabetes', data=a)
plt.title("DISTRIBUTION OF DIABETES CLASSES")
plt.show()

# COUNT PLOT FOR GENDER DISTRIBUTION (AFTER ONE-HOT ENCODING)
plt.figure(figsize=(6,4))
sns.countplot(x='gender_Male', data=a)
plt.title("GENDER DISTRIBUTION")
plt.show()

# AGE VS DIABETES BOX PLOT
plt.figure(figsize=(6,4))
sns.boxplot(x='diabetes', y='age', data=a)
plt.title("AGE DISTRIBUTION BY DIABETES STATUS")
plt.show()

# HISTOGRAMS FOR CONTINUOUS FEATURES
a[con].hist(bins=20, figsize=(15,10), color='skyblue', edgecolor='black')
plt.suptitle("DISTRIBUTION OF CONTINUOUS FEATURES")
plt.show()

# FEATURE IMPORTANCE BAR GRAPH FROM DECISION TREE MODEL
plt.figure(figsize=(10,6))
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(x_pca.shape[1]), importances[indices])
plt.title("FEATURE IMPORTANCE FROM DECISION TREE (PCA COMPONENTS)")
plt.xlabel("PCA COMPONENT INDEX")
plt.ylabel("IMPORTANCE")
plt.show()

# CONFUSION MATRIX HEATMAP
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("CONFUSION MATRIX HEATMAP")
plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.show()

# PCA SCATTER PLOT TO VISUALIZE DATA SEPARATION
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0], x_pca[:,1], c=y, cmap='coolwarm', alpha=0.7)
plt.title("PCA SCATTER PLOT (FIRST TWO COMPONENTS)")
plt.xlabel("PCA COMPONENT 1")
plt.ylabel("PCA COMPONENT 2")
plt.show()

# DECISION TREE VISUALIZATION
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, fontsize=6)
plt.title("DECISION TREE STRUCTURE")
plt.show()


