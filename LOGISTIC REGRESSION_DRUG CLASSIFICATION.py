#EXECUTED BY :- SANKET GAIKWAD
#ALGORITHM USED :- LOGISTIC REGRESSION
#SOURCE :- https://www.kaggle.com/datasets/prathamtripathi/drug-classification



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


#IMPORTING OF DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\drug.csv")
print(a.columns.tolist())


#INFORMATION AND STATISTICAL DESCRIPTION OF DATA
print()
print(a.info())

print()
print(a.describe())


#SUM OF DUPLICATED VALUES IN DATA
print()
print("TOTAL SUM OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())


#DISPLAY OF DUPLICATED VALUES ROWS
print()
c = a[a.duplicated()]
print(c)


#SEARCHING OF MISSING OR NULL VALUES IN DATA
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis=  1)]

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
print("CATEGORICAL COLUMN :-\n", cat)
print("CONTINOUS COLUMNS:-\n", con)


#ONE HOT ENCODING
print()
a = pd.get_dummies(a, columns = ['Sex', 'BP', 'Cholesterol'], drop_first=True)


#LABEL ENCODING OF TARGET DATA
le = LabelEncoder()
a['Drug'] = le.fit_transform(a['Drug'])

print()
x = a.drop('Drug', axis = 1)
y = a['Drug']


# SHOW THE PROPORTION OF EACH CLASS IN THE TARGET VARIABLE
print()
print(a['Drug'].value_counts(normalize = True))


# DISPLAY FEATURES' CORRELATION WITH TARGET VARIABLE IN DESCENDING ORDER
print()
print(a.corr(numeric_only=True)['Drug'].sort_values(ascending = False))


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components = 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLITING OF DATA AS TRAIN AND TEST
print()
train_size = int(0.8 * len(x_pca))
x_train, x_test = x_pca[:train_size], x_pca[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING AND IMPLEMENTATION
print()
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000, penalty='l2', C=1.5, n_jobs=-1, random_state=42)
model.fit(x_train_res, y_train_res)

y_pred_prob = model.predict_proba(x_test)
y_pred_prob = pd.DataFrame(y_pred_prob, columns = ['Probability(0)','Probability(1)','Probability(2)','Probability(3)','Probability(4)'])
pd.options.display.float_format = '{:.3f}'.format
print(y_pred_prob)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
print("ACCURACY SCORE:-\n", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX :-\n", confusion_matrix(y_test, y_pred))
print("CLASSIFICATION REPORT :-\n", classification_report(y_test, y_pred))


#COMPARISON OF ACTUAL AND PREDICTED 
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'predict':y_pred})
print(comparison)


#CROSS SCORE VALIDATION
print()
print("CROSS SCORE VALIDATION")
kf = KFold(n_splits = 10, shuffle = True, random_state= 1)
scores = cross_val_score(model,x_pca, y, cv = kf, scoring = 'r2')
print(scores)
print("MEAN SCORES :-\n", np.mean(scores))



#DATA VISUALISATION

# TARGET VARIABLE CLASS DISTRIBUTION
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Drug Class Distribution")
plt.xlabel("Drug Type (Encoded)")
plt.ylabel("Count")
plt.show()

# CORRELATION HEATMAP FOR NUMERICAL FEATURES
plt.figure(figsize=(10,6))
sns.heatmap(a.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# PCA EXPLAINED VARIANCE RATIO PLOT
plt.figure(figsize=(6,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title("PCA Cumulative Explained Variance")
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Variance Explained")
plt.grid(True)
plt.show()

# CONFUSION MATRIX HEATMAP
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# COMPARISON: ACTUAL vs PREDICTED (Bar Plot)
comparison_reset = comparison.reset_index(drop=True)
plt.figure(figsize=(10,4))
plt.plot(comparison_reset['Actual'], label='Actual', marker='o')
plt.plot(comparison_reset['predict'], label='Predicted', linestyle='--', marker='x')
plt.title("Actual vs Predicted Drug Classification")
plt.xlabel("Index")
plt.ylabel("Drug Class (Encoded)")
plt.legend()
plt.grid(True)
plt.show()

# FEATURE IMPORTANCE USING MODEL COEFFICIENTS (MULTICLASS)
plt.figure(figsize=(10,6))
coeff_importance = np.mean(np.abs(model.coef_), axis=0)
plt.bar(range(len(coeff_importance)), coeff_importance)
plt.title("Feature Importance from Logistic Regression")
plt.xlabel("Feature Index")
plt.ylabel("Importance (|Coefficients|)")
plt.grid(axis='y')
plt.show()








