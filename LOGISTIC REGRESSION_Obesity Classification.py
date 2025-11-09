#THE PROJECT IS EXECUTED BY :- SANKET GAIKWAD
#ALGORITHM USED :- LOGISTIC REGRESSION
#SOURCE:- https://www.kaggle.com/datasets/sujithmandala/obesity-classification-dataset?utm_source=chatgpt.com




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


#IMPORT THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\obesity.csv")
print(a.columns.tolist())


#DROPPING THE UNREQUIRED FEATURE
print()
a = a.drop('ID', axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION OF DATA
print()
print(a.info())

print()
print(a.describe())


#FINDING THE DUPLICATED VALUE SUM IN DATA
print()
print("TOTAL SUM OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())


#DISPLAY THE ROWS WITH DUPLICATED VALUES
print()
c = a[a.duplicated()]
print(c)


#FINDING THE MISSING OR NULL VALUES IN DATA
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#SEGREGATION OF CATEGORICAL AND CONTINOUS VALUES 
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
a = pd.get_dummies(a, columns = ['Gender'], drop_first= True)


#LABEL ENCODING OF TARGET VARIABLE
le = LabelEncoder()
a['Label'] = le.fit_transform(a['Label'])


# PRINT THE NORMALIZED VALUE COUNTS OF THE 'LABEL' COLUMN
print()
print(a['Label'].value_counts(normalize = True))


# PRINT THE CORRELATION OF ALL NUMERIC FEATURES WITH 'LABEL', SORTED IN DESCENDING ORDER
print()
print(a.corr(numeric_only=True)['Label'].sort_values(ascending=False))


#FEATURE SELECTION
print()
x = a.drop('Label', axis = 1)
y = a['Label']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components=0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLIT OF DATA INTO TRAIN AND TEST DATA
print()
train_size = int(0.8 * len(x_pca))
x_train, x_test = x_pca[:train_size], x_pca[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL IMPLEMENTATION AND EXECUTION
print()
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=10, max_iter=1000, random_state=42)
model.fit(x_train_res, y_train_res)

y_pred_prob = model.predict_proba(x_test)
y_pred_prob = pd.DataFrame(y_pred_prob, columns = ['Probability(0)', 'Probability(1)','Probability(2)','Probability(3)'])
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
kf = KFold(n_splits=10, shuffle = True, random_state=1)
scores  =cross_val_score(model, x_pca, y, cv = kf, scoring = 'r2')
print(scores)
print("MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALISATION

# CORRELATION HEATMAP
plt.figure(figsize=(10,8))
sns.heatmap(a.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# TARGET VARIABLE DISTRIBUTION
plt.figure(figsize=(6,4))
sns.countplot(x='Label', data=a)
plt.title("Distribution of Target Variable")
plt.show()

# PCA COMPONENTS EXPLAINED VARIANCE
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.show()

# CONFUSION MATRIX HEATMAP
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.title("Confusion Matrix")
plt.show()

# CLASSIFICATION REPORT HEATMAP
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).iloc[:-1, :].T
plt.figure(figsize=(10,5))
sns.heatmap(df_report, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title("Classification Report Heatmap")
plt.show()

# DISTRIBUTION OF FIRST TWO PCA COMPONENTS
plt.figure(figsize=(8,6))
sns.scatterplot(x=x_pca[:,0], y=x_pca[:,1], hue=y, palette='Set1')
plt.title("PCA Component 1 vs Component 2")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
