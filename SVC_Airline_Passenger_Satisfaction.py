# THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
# ALGORITHM USED :- SUPPORT VECTOR MACHINE ( SVC )
# SOURCE :- Airline Passenger Satisfaction ( https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction )




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


#IMPORTING x:- train.csv and y:- test.csv files 
x = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\train.csv")
y = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\test.csv")


#CONCATINATE(MERGE) BOTH THE FILES COLUMNWISE 
a = pd.concat([x,y], ignore_index=True)
print(a.columns.tolist())


#DROPPING NOT REQUIRED FEATURE
print()
a = a.drop(['srno', 'id'], axis = 1)


#INFORMATION ABOUT THE DATA 
print()
print(a.info())

print()
print(a.describe())

#SEARCHING DUPLICATE VALUES
# TO SEARCH THE DUPLICATE VALUES IN TOTAL
print()
print("THE SUM OF DUPLICATES IS:-\n")
print(a.duplicated().sum())


#TO SEARCH FOR DUPLICATE VALUES ROWS
print()
c = a[a.duplicated()]
print(c)



#MISSING VALUES 
#TO SERACH MISSING VALUES 
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATSET !!!")


#HANDLING MISSING VALUES 
print()
a['Arrival Delay in Minutes' ] = a['Arrival Delay in Minutes'].fillna(a['Arrival Delay in Minutes'].median())


# USING OF DUMMY VARIABLES ON CATEGORICAL VARIABLES [ Apply One-Hot Encoding to Categorical Columns ]
print()
a = pd.get_dummies(a, columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class'], drop_first= True)

#Convert Target Variable (satisfaction) to Numeric
a['satisfaction'] = a['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})


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
pca = PCA(n_components = 0.95)
x_pca = pca.fit_transform(x_scaled)


#SPLITING OF DATA INTO TRAIN AND TEST DATASET 
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y , test_size=0.2, random_state= 1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model = SVC(kernel='rbf', class_weight='balanced', gamma='auto')
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

print()
print("ACCURACY SCORE:-\n", acc)
print("COMFUSION MATRIX :-\n", cm)
print("CLASSIFICATION REPORT :-\n", clr)


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)


print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model,x_pca, y, cv = 5)
print(scores)
print("THE MEAN SCORES :-\n", np.mean(scores))


#DATA VISUALISATION

#Class Distribution Before and After SMOTE
# Class distribution before SMOTE
sns.countplot(x=y)
plt.title("Class Distribution Before SMOTE")
plt.show()

# Class distribution after SMOTE
sns.countplot(x=y_train_res)
plt.title("Class Distribution After SMOTE")
plt.show()


#Correlation Heatmap (Before PCA)
plt.figure(figsize=(12, 6))
sns.heatmap(pd.DataFrame(x_scaled, columns=x.columns).corr(), cmap="coolwarm")
plt.title("Correlation Heatmap Before PCA")
plt.show()


#Explained Variance by PCA
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance by PCA Components")
plt.grid()
plt.show()


#Cross-Validation Scores (Bar Plot)
plt.figure(figsize=(6, 4))
plt.bar(range(1, 6), scores)
plt.axhline(np.mean(scores), color='red', linestyle='--', label=f"Mean Score: {np.mean(scores):.2f}")
plt.title("Cross-Validation Accuracy Scores")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


