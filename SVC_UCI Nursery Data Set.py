#THE CODE IS EXECUTED BY MR.SANKET GAIKWAD
#ALGORITHM USED :- SVC
#SOURCE :- https://www.kaggle.com/datasets/heitornunes/nursery

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\Desktop\\My Data\\nursery.csv")
print(a.columns.tolist())


#INFORMATION AND STATISTICAL INFORMATION OF DATA
print()
print(a.info())

print()
print(a.describe())


#DUPLICATED VALUES IN DATA
print()
print("DUPLICATED VALUES :-\n")
print(a.duplicated().sum())

print()
c = a[a.duplicated()]
print(c)


#MISSING VALUES IN DATA
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#SEGREGATION OF CATEGORICAL AND CONTINOUS DATA
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
a = pd.get_dummies(a, columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health'], drop_first=True)


#TARGET VARIABLE LABEL ENCODING
le = LabelEncoder()
a['final evaluation'] = le.fit_transform(a['final evaluation'])


#FEATURE SELECTION
print()
x = a.drop('final evaluation', axis = 1)
y = a['final evaluation']


#SPLIT OF DATA AS TRAIN AND TEST
print()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 1)


# THIS LINE SHOWS THE PERCENTAGE DISTRIBUTION OF EACH CLASS IN THE TARGET VARIABLE
print()
print(a['final evaluation'].value_counts(normalize=True))


# THIS LINE DISPLAYS THE CORRELATION OF NUMERICAL FEATURES WITH THE TARGET VARIABLE IN ASCENDING ORDER
print()
print(a.corr(numeric_only=True)['final evaluation'].sort_values(ascending=True))


#STANDARD SCALER
print()
sc= StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca =PCA(n_components= 0.95)
x_pca =pca.fit_transform(x_scaled)


#MODEL BUILDING AND IMPLEMENTING
print()
model = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', probability=True, random_state=1)
model.fit(x_train, y_train)

y_pred_prob = model.predict_proba(x_test)
print(y_pred_prob)

print()
y_pred_prob = pd.DataFrame(y_pred_prob, columns = ['not_recom','priority','recommend','spec_prior','very_recom'])
pd.options.display.float_format = '{:.3f}'.format
print(y_pred_prob)

print()
y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
print("ACCURACY SCORE :-\n", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX :-\n", accuracy_score(y_test, y_pred))
print("CLASSIFICATION REPORT :-\n", classification_report(y_test, y_pred))


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)


#CROSS SCORE VALIDATION
print()
print("CROSS SCORE VALIDATION")
kf = KFold(n_splits = 10, shuffle = True, random_state= 1)
scores = cross_val_score(model,x_pca, y, cv = kf, scoring='accuracy')
print(scores)
print("MEAN SCORES :-\n", np.mean(scores))


#DATA VISUALISATION

plt.figure(figsize=(16, 12))

# TARGET CLASS DISTRIBUTION
plt.subplot(2, 2, 1)
a['final evaluation'].value_counts().plot(kind='bar')
plt.title("TARGET CLASS DISTRIBUTION")
plt.xlabel("FINAL EVALUATION CLASS")
plt.ylabel("COUNT")

# CORRELATION HEATMAP
plt.subplot(2, 2, 2)
sns.heatmap(
    a.corr(numeric_only=True),
    cmap='coolwarm',
    linewidths=0.5
)
plt.title("CORRELATION HEATMAP")

# CONFUSION MATRIX HEATMAP
plt.subplot(2, 2, 3)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)
plt.title("CONFUSION MATRIX")
plt.xlabel("PREDICTED")
plt.ylabel("ACTUAL")

# CROSS VALIDATION ACCURACY DISTRIBUTION
plt.subplot(2, 2, 4)
plt.plot(scores, marker='o')
plt.title("CROSS VALIDATION ACCURACY")
plt.xlabel("FOLD")
plt.ylabel("ACCURACY")
plt.ylim(0.9, 1.01)

plt.tight_layout()
plt.show()