#THE PROGRAM IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- LOGISTIC REGRESSION


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score
from imblearn.over_sampling import SMOTE


#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\MARKETING ANALYTICS\\Dataset\\hbat.csv")
print(a.columns.tolist())

#DROPPING THE UNREQUIRED FEATURES
print()
a = a.drop(['id','x2', 'x3', 'x4', 'x5'], axis =1)

#INFORMATION AND STATISTICAL DESCRIPTION OF THE DATA
print()
print(a.info())

print()
print(a.describe())


#TO FIND THE TOTAL SUM OF DUPLICATED VALUES
print()
print("TOTAL SUM OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())

#DISPLAYING OF DUPLICATED VALUES ROWS 
print()
c = a[a.duplicated()]
print(c)


#TO SEARCH FOR NULL/MISSING VALUES
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


# PRINTS THE NORMALIZED VALUE COUNTS OF THE TARGET COLUMN 'x1'
print()
print(a['x1'].value_counts(normalize = True))


# PRINTS THE CORRELATION OF ALL NUMERICAL COLUMNS WITH 'x1' SORTED IN DESCENDING ORDER
print()
print(a.corr(numeric_only=True)['x1'].sort_values(ascending = False))


#FEATURE SELECTION
print()
x = a.drop('x1', axis = 1)
y = a['x1']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca = pca.fit_transform(x_scaled)


#SPLIT OF DATA AS TEST AND TRAIN DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state= 1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING AND IMPLEMENTATION
print()
model = LogisticRegression(solver='lbfgs', penalty='l2', C=10, max_iter=2000, multi_class='multinomial', random_state=1)
model.fit(x_train_res, y_train_res)

y_predict_prob = model.predict_proba(x_test)
print(y_predict_prob)

y_predict_prob = pd.DataFrame(y_predict_prob, columns = ['Probability(1)', 'Probability(2)', 'probability(3)'])
pd.options.display.float_format = '{:.3f}'.format
print(y_predict_prob)

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
kf = KFold(n_splits = 10, shuffle=True, random_state= 1)
scores = cross_val_score(model, x_pca, y, cv = kf, scoring = 'r2')
print(scores)
print("THE MEAN SCORES :-\n", np.mean(scores))


#DATA VISUALISATION

# 1. TARGET VARIABLE DISTRIBUTION
plt.figure(figsize=(6,4))
sns.countplot(x='x1', data=a, palette='viridis')
plt.title('Distribution of Target Variable x1')
plt.xlabel('x1')
plt.ylabel('Count')
plt.show()

# 2. NORMALIZED VALUE COUNTS PIE CHART
plt.figure(figsize=(6,6))
a['x1'].value_counts(normalize=True).plot.pie(autopct='%1.1f%%', colors=['#66b3ff','#ff9999','#99ff99'])
plt.title('Proportion of Each Class in x1')
plt.ylabel('')
plt.show()

# 3. CORRELATION HEATMAP
plt.figure(figsize=(12,10))
sns.heatmap(a.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# 4. PCA COMPONENT VARIANCE EXPLAINED
plt.figure(figsize=(8,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='b')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Cumulative Explained Variance')
plt.grid(True)
plt.show()

# 5. CONFUSION MATRIX HEATMAP
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 6. CLASSIFICATION REPORT HEATMAP (OPTIONAL)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).iloc[:-1, :].T
plt.figure(figsize=(8,6))
sns.heatmap(report_df, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Classification Report Heatmap')
plt.show()

# 7. COMPARISON OF ACTUAL VS PREDICTED
comparison_sample = comparison.head(50)  # visualize first 50 predictions
plt.figure(figsize=(15,4))
plt.plot(comparison_sample['Actual'].values, marker='o', label='Actual', color='b')
plt.plot(comparison_sample['Predict'].values, marker='x', label='Predicted', color='r')
plt.title('Comparison of Actual vs Predicted Values (First 50 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('x1')
plt.legend()
plt.show()


