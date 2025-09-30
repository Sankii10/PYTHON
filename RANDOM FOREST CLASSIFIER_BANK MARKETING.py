#THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM :- RANDOM FOREST CLASSIFIER
#SOURCE :- https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


#IMPORTING DATA 
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\bank.csv")
print(a.columns.tolist())


#DROPPING UNREQUIRED FEATURES
print()
a = a.drop(['duration', 'contact', 'poutcome', 'day'], axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION OF DATA
print()
print(a.info())

print()
print(a.describe())

#TOTAL NUMBER OF DUPLICATED VALUES
print()
print("TOTAL SUM OF DUPLICATED VALUES :-")
print(a.duplicated().sum())


#DISPLAYING ROW WISE DUPLICATED VALUES
print()
c = a[a.duplicated()]
print(c)


#DROPPING THE DUPLICATED VALUES
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
print("CATEGORICAL VALUES :-\n", cat)
print("CONTINOUS VALUES :-\n", con)


#ONE HOT ENCODING
print()
a = pd.get_dummies(a, columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'month'], drop_first=True)


#LABEL ENCODING OF TARGET VARIABLE
print()
le = LabelEncoder()
a['deposit'] = le.fit_transform(a['deposit'])

# SHOWS THE RELATIVE FREQUENCY OF EACH VALUE IN 'DEPOSIT'
print()
print(a['deposit'].value_counts(normalize= True))

# SHOWS CORRELATION OF ALL NUMERIC COLUMNS WITH 'DEPOSIT' IN DESCENDING ORDER
print()
print(a.corr(numeric_only=True)['deposit'].sort_values(ascending= False))

#FEATURE SELECTION
print()
x = a.drop('deposit', axis =1)
y = a['deposit']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

print()
#Removed PCA for RandomForest (better performance)
x_pca = x_scaled


#SPLIT OF DATA IN TRAIN AND TEST
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size= 0.2, random_state= 1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#mODEL BUILDING AND EXECUTION
print()
model = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', bootstrap=True, class_weight='balanced', random_state=1, n_jobs=-1)
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

# SETTING STYLE
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10,6)

# 1. DISTRIBUTION OF TARGET VARIABLE
plt.figure()
sns.countplot(x='deposit', data=a)
plt.title('DISTRIBUTION OF TARGET VARIABLE (DEPOSIT)')
plt.show()

# 2. RELATIVE FREQUENCY OF TARGET VARIABLE
plt.figure()
a['deposit'].value_counts(normalize=True).plot(kind='bar', color=['skyblue','salmon'])
plt.title('RELATIVE FREQUENCY OF DEPOSIT')
plt.ylabel('Proportion')
plt.show()

# 3. CORRELATION HEATMAP OF NUMERIC FEATURES
plt.figure()
sns.heatmap(a.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('CORRELATION HEATMAP')
plt.show()

# 4. HISTOGRAMS OF NUMERIC FEATURES
a[con].hist(bins=20, figsize=(15,10), color='lightgreen')
plt.suptitle('HISTOGRAMS OF NUMERIC FEATURES')
plt.show()

# 5. BOX PLOTS OF NUMERIC FEATURES VS TARGET
for col in con:
    plt.figure()
    sns.boxplot(x='deposit', y=col, data=a)
    plt.title(f'{col} VS DEPOSIT')
    plt.show()


# 6. FEATURE IMPORTANCE FROM RANDOM FOREST
importances = model.feature_importances_
features = x.columns
feature_importance_df = pd.DataFrame({'Feature':features, 'Importance':importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('FEATURE IMPORTANCE FROM RANDOM FOREST')
plt.show()

# 7. CONFUSION MATRIX HEATMAP
plt.figure()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('CONFUSION MATRIX')
plt.xlabel('PREDICTED')
plt.ylabel('ACTUAL')
plt.show()

# 8. DISTRIBUTION OF PREDICTED PROBABILITIES
y_prob = model.predict_proba(x_test)[:,1]
plt.figure()
sns.histplot(y_prob, bins=20, kde=True, color='orange')
plt.title('DISTRIBUTION OF PREDICTED PROBABILITIES FOR DEPOSIT=1')
plt.xlabel('Predicted Probability')
plt.show()