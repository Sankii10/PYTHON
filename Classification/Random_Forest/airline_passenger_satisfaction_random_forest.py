#THE PROGRAM IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- RANDOM FOREST CLASSIFICATION
#SOURCE :- https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE


#IMPORTING THE TRAIN AND TEST DATASET 
x = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\train.csv")
y = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\test.csv")

#CONCATENATE THE TRAIN AND TEST DATASET 
print()
a = pd.concat([x,y], axis = 0)
print(a.columns.tolist())


#INFORMATION AND STATISTICAL DESCRIPTION OF DATA
print()
print(a.info())

print()
print(a.describe())


#FINDING THE SUM OF DUPLICATED VALUES IN DATA
print()
print("TOTAL SUM OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())


#DISPLAY THE DUPLICATED VALUE IN ROWS
print()
c = a[a.duplicated()]
print(c)


#SEARCHING FOR MISSING / NULL VALUES IN DATA
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#HANDLING MISSING/NULL VALUES IN DATA
print()
a['Arrival Delay in Minutes'] = a['Arrival Delay in Minutes'].fillna(a['Arrival Delay in Minutes'].mean())


#SEGREGATE THE CATEGORICAL AND CONTINOUS VALUE
print()
cat = []
con = []

for i in a:
    if a[i].dtypes == 'object':
        cat.append(i)
    else:
        con.append(i)

print()
print("CATEGORICAL VALUES :-\n", cat)
print("CONTINOUS VALUES :-\n", con)


#ONE HOT ENCODING 
print()
a = pd.get_dummies(a, columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class'], drop_first = True)


#LABEL ENCODING FOR TARGET VARIABLE
le = LabelEncoder()
a['satisfaction'] = le.fit_transform(a['satisfaction'])


#PRINTS THE PROPORTION OF EACH CLASS IN THE 'SATISFACTION' COLUMN AS FRACTIONS OF THE TOTAL
print()
print(a['satisfaction'].value_counts(normalize = True))


#PRINTS THE CORRELATION OF ALL NUMERIC COLUMNS WITH 'SATISFACTION' IN DESCENDING ORDER
print()
print(a.corr(numeric_only= True)['satisfaction'].sort_values(ascending = False))


#FEATURES SELECTION
print()
x = a.drop('satisfaction', axis = 1)
y = a['satisfaction']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLITING OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state= 1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING AND IMPLEMENTATION
print()
model = RandomForestClassifier(n_estimators=300, criterion='gini', max_depth=20, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', bootstrap=True, random_state=1)
model.fit(x_train_res,y_train_res)

y_pred_prob = model.predict_proba(x_test)[:,1]
y_pred = (y_pred_prob > 0.5).astype(int)


#EVALUATION
print()
print("EVALUATION")
print("ACCURACY SCORE:-\n", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX :-\n", confusion_matrix(y_test, y_pred))
print("CLASSIFICATION REPORT :-\n", classification_report(y_test, y_pred))
print("RECALL SCORE :-\n", recall_score(y_test, y_pred))
print("ROC SCORE :-\n", roc_auc_score(y_test, y_pred))


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)


#CROSS SCORE VALIDATION
print("CROSS SCORE VALIDATION")
kf = KFold(n_splits = 10, shuffle = True, random_state= 1)
scores  =cross_val_score(model,x_pca, y, cv = kf, scoring = 'r2')
print(scores)
print("THE MEAN SCORES :-\n", np.mean(scores))


print()
#DATA VISUALISATION

# 1️⃣ Plot the distribution of the target variable 'satisfaction'
plt.figure(figsize=(6,4))
sns.countplot(x='satisfaction', data=a, palette='viridis')
plt.title('Distribution of Satisfaction Classes')
plt.xlabel('Satisfaction (0=Neutral/Dissatisfied, 1=Satisfied)')
plt.ylabel('Count')
plt.show()

# 2️⃣ Pie chart of class proportion
plt.figure(figsize=(6,6))
a['satisfaction'].value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue','orange'], explode=[0,0.1])
plt.title('Proportion of Satisfaction Classes')
plt.ylabel('')
plt.show()

# 3️⃣ Correlation heatmap of numeric features
plt.figure(figsize=(12,8))
sns.heatmap(a.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap of Numeric Features')
plt.show()

# 4️⃣ Distribution plots of key continuous features
continuous_features = ['Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes']  # adjust based on your dataset
for col in continuous_features:
    plt.figure(figsize=(6,4))
    sns.histplot(a[col], kde=True, color='teal')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

# 5️⃣ Boxplots of continuous features vs target to check relationship
for col in continuous_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='satisfaction', y=col, data=a, palette='Set2')
    plt.title(f'{col} vs Satisfaction')
    plt.show()

# 6️⃣ Countplots for categorical features
categorical_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class']  # original categorical columns
for col in categorical_features:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, hue='satisfaction', data=pd.concat([x,y], axis=0), palette='pastel')
    plt.title(f'{col} vs Satisfaction')
    plt.show()

# 7️⃣ Pairplot for a subset of important numeric features colored by satisfaction
subset_features = ['Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes','satisfaction']
sns.pairplot(a[subset_features], hue='satisfaction', palette='coolwarm', diag_kind='kde')
plt.show()

# 8️⃣ Feature importance plot from Random Forest
importances = model.feature_importances_
feature_names = x.columns
feat_importances = pd.Series(importances, index=feature_names)
feat_importances = feat_importances.sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_importances[:15], y=feat_importances.index[:15], palette='magma')
plt.title('Top 15 Feature Importances from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()



