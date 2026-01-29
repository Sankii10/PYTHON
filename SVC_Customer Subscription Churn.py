#THE CODE IS EXECUTED BY MR.SANKET GAIKWAD
#ALGORITHM USED :- SVC( SUPPORT VECTORE MACHINE)
#SOURCE :- https://www.kaggle.com/datasets/jayjoshi37/customer-subscription-churn-and-usage-patterns


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# DATA LOADING
a = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\Desktop\\My Data\\Custchurn.csv")
print(a.columns.tolist())


#DROPING UNWANTED FEATURES
a = a.drop(['user_id', 'signup_date'], axis=1)

#INFORMATION AND STATISTICAL DSCRIPTION OF DATA
print()
print(a.info())

print()
print(a.describe())


#SEARCHING FOR DUPLICATED VALUES
print("DUPLICATED VALUES")
print(a.duplicated().sum())
print(a[a.duplicated()])


#SEARCHING FOR MISSING VALUES
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis=1)]
if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!")


# CATEGORICAL AND CONTINUOUS VARIABLES
cat = []
con = []

for i in a:
    if a[i].dtype == 'object':
        cat.append(i)
    else:
        con.append(i)

print("CATEGORICAL VALUES")
print(cat)
print("CONTINUOUS VALUES")
print(con)


#ONE HOT ENCODING
a = pd.get_dummies(a, columns=['plan_type'], drop_first=True)


#LABEL ENCODING ON TARGET VARIABLE
le = LabelEncoder()
a['churn'] = le.fit_transform(a['churn'])


#FEATURE SELECTION
x = a.drop('churn', axis=1)
y = a['churn']


# TRAIN TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1, stratify=y
)


# SCALING 
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)


#PRINCIPAL COMPONENT ANALYSIS
pca = PCA(n_components=0.95)
x_train_pca = pca.fit_transform(x_train_sc)
x_test_pca = pca.transform(x_test_sc)


# APPLY SMOTE ON TRAINING DATA ONLY
smote = SMOTE(random_state=1)
x_train_res, y_train_res = smote.fit_resample(x_train_pca, y_train)


# MODEL TRAINING AND IMPLEMENTATION
model = SVC(kernel='rbf', C=3, gamma=0.08, probability=True, random_state=1)
model.fit(x_train_res, y_train_res)

y_pred_prob = model.predict_proba(x_test_pca)[:, 1]
y_pred = (y_pred_prob > 0.45).astype(int)
print(y_pred)


# EVALUATION
print("ACCURACY SCORE")
print(accuracy_score(y_test, y_pred))

print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))


# COMPARISON
comparison = pd.DataFrame({'Actual': y_test, 'Predict': y_pred})
print(comparison)


# CROSS VALIDATION WITH SMOTE
pipe = Pipeline([
    ('Scaler', StandardScaler()),
    ('PCA', PCA(n_components=0.95)),
    ('SMOTE', SMOTE(random_state=1)),
    ('model', SVC(kernel='rbf', C=3, gamma=0.08, probability=True, random_state=1))
])

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

scores = cross_val_score(pipe, x, y, cv=cv, scoring='accuracy')

print("CROSS VALIDATION SCORES")
print(scores)

print("MEAN CROSS VALIDATION SCORE")
print(np.mean(scores))


# CLASS DISTRIBUTION OF TARGET VARIABLE
plt.figure()
sns.countplot(x=y)
plt.title("CLASS DISTRIBUTION OF CHURN")
plt.xlabel("CHURN CLASS")
plt.ylabel("COUNT")
plt.show()


# CONFUSION MATRIX HEATMAP
plt.figure()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("CONFUSION MATRIX HEATMAP")
plt.xlabel("PREDICTED LABEL")
plt.ylabel("ACTUAL LABEL")
plt.show()


# ROC CURVE
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="ROC CURVE (AUC = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("FALSE POSITIVE RATE")
plt.ylabel("TRUE POSITIVE RATE")
plt.title("ROC CURVE")
plt.legend(loc="lower right")
plt.show()


# PCA EXPLAINED VARIANCE
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("NUMBER OF COMPONENTS")
plt.ylabel("CUMULATIVE EXPLAINED VARIANCE")
plt.title("PCA EXPLAINED VARIANCE")
plt.show()


# CROSS VALIDATION SCORE DISTRIBUTION
plt.figure()
plt.bar(range(1, len(scores) + 1), scores)
plt.xlabel("FOLD NUMBER")
plt.ylabel("ACCURACY SCORE")
plt.title("CROSS VALIDATION ACCURACY SCORES")
plt.show()
