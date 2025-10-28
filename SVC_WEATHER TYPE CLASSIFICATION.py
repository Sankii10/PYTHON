#NAME : SANKET GAIKWAD
#ALGORITHM :- SUPPORT VECTOR CLASSIFIER
#SOURCE :- https://www.kaggle.com/datasets/nikhil7280/weather-type-classification


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


#IMPORT DATA 
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\wcd.csv")
print(a.columns.tolist())


#INFORMATION AND STATISTICAL DESCRIPTION OF THE DATA
print()
print(a.info())

print()
print(a.describe())


#SEARCHING FOR THE SUM OF DUPLICATED VALUES 
print()
print("TOTAL SUM OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())

#DISPLAY OF DUPLICATED VALUES 
print()
c = a[a.duplicated()]
print(c)


#SEARCHING FOR MISSING VALUES IN DATA
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#SEGREGATE THE CATEGORICAL VALUES AND CONTINOUS VALUES 
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
a = pd.get_dummies(a, columns = ['Cloud Cover', 'Season', 'Location'], drop_first=True)

#LABEL ENCODING ON TARGET VARIABLE
le = LabelEncoder()
a['Weather Type'] = le.fit_transform(a['Weather Type'])


# DISPLAY THE PROPORTION (PERCENTAGE) OF EACH CATEGORY IN THE 'WEATHER TYPE' COLUMN
print()
print(a['Weather Type'].value_counts(normalize=True))


# SHOW THE CORRELATION OF ALL NUMERIC FEATURES WITH 'WEATHER TYPE' IN DESCENDING ORDER
print()
print(a.corr(numeric_only= True)['Weather Type'].sort_values(ascending= False))

#FEATURE SELECTION
print()
x = a.drop('Weather Type',axis = 1)
y = a['Weather Type']

#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
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
model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, tol=1e-4, shrinking=True, random_state=42)
model.fit(x_train_res, y_train_res)

y_pred_prob = model.predict_proba(x_test)
y_pred_prob = pd.DataFrame(y_pred_prob, columns = ['Probability(0)','Probability(1)','Probability(2)','Probability(3)'])
pd.options.display.float_format = '{:.3f}'.format
print(y_pred_prob)

print()
y_pred = model.predict(x_test)
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
scores  =cross_val_score(model,x_pca, y, cv = kf, scoring= 'r2')
print(scores)
print("MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALISATION
# DISTRIBUTION OF WEATHER TYPES
plt.figure(figsize=(8,5))
sns.countplot(x='Weather Type', data=a)
plt.title('Distribution of Weather Types')
plt.xlabel('Weather Type')
plt.ylabel('Count')
plt.show()

# CORRELATION HEATMAP
plt.figure(figsize=(12,8))
sns.heatmap(a.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap of Features')
plt.show()

# PCA EXPLAINED VARIANCE
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.show()

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# CLASSIFICATION REPORT VISUALIZATION (PRECISION, RECALL, F1)
from sklearn.metrics import precision_recall_fscore_support

precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
classes = le.classes_

report_df = pd.DataFrame({
    'Class': classes,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Support': support
})

report_df.set_index('Class', inplace=True)
report_df.plot(kind='bar', figsize=(10,6))
plt.title('Classification Metrics by Class')
plt.ylabel('Score')
plt.ylim(0,1)
plt.grid(axis='y')
plt.show()

#  ACTUAL VS PREDICTED COMPARISON (FOR TEST SET)
plt.figure(figsize=(12,5))
plt.plot(range(len(y_test)), y_test, marker='o', linestyle='', label='Actual')
plt.plot(range(len(y_pred)), y_pred, marker='x', linestyle='', label='Predicted')
plt.title('Actual vs Predicted Weather Type')
plt.xlabel('Sample Index')
plt.ylabel('Weather Type')
plt.legend()
plt.show()


