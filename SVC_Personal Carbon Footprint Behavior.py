#THE CODE IS IMPLEMENTED BY MR.SANKET GAIKWAD
#ALGORITHM USED :- SVC
#SOURCE :- https://www.kaggle.com/datasets/sonalshinde123/personal-carbon-footprint-behavior-dataset


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline


#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\Desktop\\My Data\\PCF.csv")
print(a.columns.tolist())


#REMOVING THE UNREQUIRED FEATURES
print()
a = a.drop(['user_id','carbon_footprint_kg'], axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION OF DATA
print()
print(a.info())

print()
print(a.describe())


#DUPLICATED VALUES SEARCHING
print()
print("DUPLICATED VALUES")
print(a.duplicated().sum())

print()
c = a[a.duplicated()]
print(c)


#MISSING VALUES SEARCHING
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!")


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
a = pd.get_dummies(a, columns = ['day_type', 'transport_mode', 'food_type'], drop_first=True)


#LABEL ENCODING FOR TARGET VARIABLE
le = LabelEncoder()
a['carbon_impact_level'] = le.fit_transform(a['carbon_impact_level'])


#FEATURE SELECTION
print()
x = a.drop('carbon_impact_level', axis = 1)
y = a['carbon_impact_level']


#SPLIT OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state= 1)


#STANDARD SCALING
print()
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components=0.95)
x_train_pca =pca.fit_transform(x_train_sc)
x_test_pca = pca.transform(x_test_sc)


#MODEL BUILDING AND IMPLEMENTATION
print()
model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=1)
model.fit(x_train_pca, y_train)

y_pred_prob = model.predict_proba(x_test_pca)
print(y_pred_prob)

print()
y_pred_prob = pd.DataFrame(y_pred_prob, columns = ['Low', 'Medium','High'])
pd.options.display.float_format = '{:.3f}'.format
print(y_pred_prob)

print()
y_pred = model.predict(x_test_pca)
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


#CROSS SCORE VALIDATION
print()
print("CROSS SCORE VALIDATION")

pipe = Pipeline([
    ('Scaler', StandardScaler()),
    ('PCA', PCA(n_components=0.95)),
    ('model',SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=1))
])

cv = StratifiedKFold(n_splits=10, shuffle = True, random_state= 1)

scores = cross_val_score(pipe, x,y, cv= cv, scoring = 'f1_macro')

print("CV F1 Macro Scores:", scores)
print("Mean CV F1 Macro Score:", scores.mean())
print("Std CV F1 Macro Score:", scores.std())


# DATA VISUALISATION

# CLASS DISTRIBUTION OF TARGET VARIABLE
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("CLASS DISTRIBUTION OF CARBON IMPACT LEVEL")
plt.xlabel("CARBON IMPACT LEVEL")
plt.ylabel("COUNT")
plt.show()


# FEATURE CORRELATION HEATMAP
plt.figure(figsize=(10,6))
sns.heatmap(pd.DataFrame(x).corr(), cmap='coolwarm', annot=False)
plt.title("FEATURE CORRELATION HEATMAP")
plt.show()


# PCA EXPLAINED VARIANCE RATIO
plt.figure(figsize=(7,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel("NUMBER OF PRINCIPAL COMPONENTS")
plt.ylabel("CUMULATIVE EXPLAINED VARIANCE")
plt.title("PCA EXPLAINED VARIANCE")
plt.show()


# CONFUSION MATRIX HEATMAP
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['LOW','MEDIUM','HIGH'],
            yticklabels=['LOW','MEDIUM','HIGH'])
plt.xlabel("PREDICTED LABEL")
plt.ylabel("ACTUAL LABEL")
plt.title("CONFUSION MATRIX FOR SVC MODEL")
plt.show()


# CLASS WISE PRECISION RECALL AND F1 SCORE
report = classification_report(y_test, y_pred, output_dict=True)
metrics_df = pd.DataFrame(report).transpose().iloc[:3, :3]

metrics_df.plot(kind='bar', figsize=(8,5))
plt.title("PRECISION RECALL AND F1 SCORE BY CLASS")
plt.ylabel("SCORE")
plt.xticks(rotation=0)
plt.ylim(0,1)
plt.grid(axis='y')
plt.show()


# PROBABILITY DISTRIBUTION FOR PREDICTED CLASSES
plt.figure(figsize=(8,5))
sns.histplot(y_pred_prob['Low'], color='blue', label='LOW', kde=True)
sns.histplot(y_pred_prob['Medium'], color='green', label='MEDIUM', kde=True)
sns.histplot(y_pred_prob['High'], color='red', label='HIGH', kde=True)
plt.title("PREDICTED PROBABILITY DISTRIBUTION")
plt.xlabel("PROBABILITY")
plt.ylabel("FREQUENCY")
plt.legend()
plt.show()


# CROSS VALIDATION F1 SCORE DISTRIBUTION
plt.figure(figsize=(7,4))
plt.plot(scores, marker='o')
plt.title("CROSS VALIDATION F1 MACRO SCORES")
plt.xlabel("FOLD")
plt.ylabel("F1 MACRO SCORE")
plt.ylim(0,1)
plt.grid()
plt.show()



