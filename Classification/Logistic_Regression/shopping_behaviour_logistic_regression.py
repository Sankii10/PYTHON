#THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- LOGISTIC REGRESSION
#SOURCE :- https://www.kaggle.com/datasets/zubairamuti/shopping-behaviours-dataset





import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, class_likelihood_ratios


#IMPORTING THE DATA
a =  pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\shop.csv")
print(a.columns.tolist())


#DROPPING UNREQUIRED FEATURES
print()
a = a.drop(['Customer ID', 'Item Purchased', 'Purchase Amount (USD)', 'Promo Code Used'], axis = 1)


#INFORMATION ABOUT THE DATA AND STATISTICAL DESCRIPTION OF THE DATA
print()
print(a.info())

print()
print(a.describe())


#FINDING TOTAL NUMBER OF DUPLICATES
print()
print("TOTAL SUM OF DUPLICATED VALUES ")
print(a.duplicated().sum())


#SEARCHING FOR DUPLICATED ROWS
print()
c = a[a.duplicated()]
print(c)


#SEARCHING OF MISSING VALUES IN DATA
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATSET !!!")


#ONE HOT ENCODNG 
print()
a = pd.get_dummies(a, columns = ['Size', 'Discount Applied', 'Shipping Type', 'Season', 'Color', 'Location', 'Category', 'Gender', 'Payment Method', 'Frequency of Purchases'], drop_first= True)

#TARGET VARIBLE LABEL ENCODING
le = LabelEncoder()
a['Subscription Status'] = le.fit_transform(a['Subscription Status'])


# DISPLAY THE DISTRIBUTION OF THE TARGET VARIABLE AS PROPORTIONS (CLASS BALANCE CHECK)
print()
print(a['Subscription Status'].value_counts(normalize = True))


# SHOW CORRELATION OF ALL NUMERIC FEATURES WITH THE TARGET 'SUBSCRIPTION STATUS', SORTED IN DESCENDING ORDER
print()
print(a.corr(numeric_only=True)['Subscription Status'].sort_values(ascending=False))


#FEATURES SELECTION
print()
x = a.drop('Subscription Status', axis = 1)
y = a['Subscription Status']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca =PCA(n_components= 0.95)
x_pca=pca.fit_transform(x_scaled)


#SPLITING OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state= 1)


#MODEL BUILDING
print()
model = LogisticRegression(class_weight='balanced', solver='liblinear', penalty='l2', C=1.0, max_iter=1000, random_state=1)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
print('ACCURACY SCORE:-\n', accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX :-\n",confusion_matrix(y_test, y_pred))
print("CLASSIFICATION REPORT :-\n", class_likelihood_ratios(y_test, y_pred))


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model,x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))



#DATA VISUALISATION

# 1. TARGET VARIABLE DISTRIBUTION (CLASS BALANCE)
plt.figure(figsize=(6,4))
sns.countplot(x='Subscription Status', data=a, palette='Set2')
plt.title("TARGET VARIABLE DISTRIBUTION (SUBSCRIPTION STATUS)")
plt.show()

# 2. CORRELATION HEATMAP
plt.figure(figsize=(12,8))
sns.heatmap(a.corr(numeric_only=True), cmap="coolwarm", annot=False, cbar=True)
plt.title("CORRELATION HEATMAP OF FEATURES")
plt.show()

# 3. PCA EXPLAINED VARIANCE RATIO
plt.figure(figsize=(6,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("NUMBER OF COMPONENTS")
plt.ylabel("CUMULATIVE EXPLAINED VARIANCE")
plt.title("PCA EXPLAINED VARIANCE RATIO")
plt.grid(True)
plt.show()

# 4. CONFUSION MATRIX HEATMAP
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NO', 'YES'], yticklabels=['NO', 'YES'])
plt.xlabel("PREDICTED")
plt.ylabel("ACTUAL")
plt.title("CONFUSION MATRIX")
plt.show()

# 5. COMPARISON OF ACTUAL VS PREDICTED
comparison['Result'] = np.where(comparison['Actual']==comparison['Predict'], 'Correct', 'Incorrect')
plt.figure(figsize=(6,4))
sns.countplot(x='Result', data=comparison, palette='Set1')
plt.title("ACTUAL VS PREDICTED OUTCOMES")
plt.show()

# 6. CROSS VALIDATION SCORE DISTRIBUTION
plt.figure(figsize=(6,4))
plt.plot(range(1, 11), scores, marker='o')
plt.xlabel("FOLD")
plt.ylabel("ACCURACY")
plt.title("CROSS VALIDATION SCORES")
plt.grid(True)
plt.show()






