#THE FOLLOWING CODE EXECUTED BY :- SANKET GAIKWAD
#ALGORITHM USED :- XBOOST [ XGBClassifier ]
#SOURCE :- https://www.kaggle.com/datasets/prachi13/customer-analytics


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\Train.csv")
print(a.columns.tolist())


#DROPING THE UNREQUIRED FEATURE
print()
a = a.drop('ID', axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION OF DATA
print()
print(a.info())

print()
print(a.describe())


#SEARCHING FOR SUM OF DUPLICATED VALUES
print()
print("TOTAL SUM OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())


#DISPLAY DUPLICATED VALUES ROW-WISE VALUES
print()
c = a[a.duplicated()]
print(c)


#SEARCHING FOR MISSING OR NULL VALUES 
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
a = pd.get_dummies(a, columns =['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender'], drop_first=True)


# PRINT THE PROPORTION (PERCENTAGE) OF EACH CLASS IN THE TARGET VARIABLE
print()
print(a['Reached.on.Time_Y.N'].value_counts(normalize= True))


# DISPLAY THE CORRELATION OF ALL NUMERIC FEATURES WITH THE TARGET VARIABLE, SORTED FROM HIGHEST TO LOWEST
print()
print(a.corr(numeric_only= True)['Reached.on.Time_Y.N'].sort_values(ascending = False))


#FEATURE SELECTION
print()
x = a.drop('Reached.on.Time_Y.N', axis = 1)
y = a['Reached.on.Time_Y.N']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#SPLIT OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=1)


#SMOTE (OVERSAMPLING)
print()
smote = SMOTE(random_state=1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL IMPLEMENTATION AND EXECUTION (XGBOOST MODEL TO REDUCE OVERFITTING AND IMPROVE PERFORMANCE)
print()
model = XGBClassifier(n_estimators=400,learning_rate=0.05,max_depth=6,subsample=0.85,colsample_bytree=0.8,min_child_weight=3,gamma=0.2,random_state=1,eval_metric='logloss')
model.fit(x_train_res, y_train_res)

y_pred_prob = model.predict_proba(x_test)[:,1]
y_pred = (y_pred_prob >= 0.5).astype(int)
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


#CROSS SCORE VALIDATION
print()
print("CROSS SCORE VALIDATION")
kf = KFold(n_splits = 10, shuffle = True, random_state=1)
scores = cross_val_score(model, x_scaled, y, cv=kf, scoring='accuracy')
print(scores)
print("MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALISATION


# CORRELATION HEATMAP
plt.figure(figsize=(12,8))
sns.heatmap(a.corr(numeric_only=True), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# DISTRIBUTION PLOTS OF CONTINUOUS FEATURES
for col in ['Customer_care_calls','Customer_rating','Cost_of_the_Product',
            'Prior_purchases','Discount_offered','Weight_in_gms']:
    plt.figure(figsize=(7,4))
    sns.histplot(a[col], kde=True, color='blue')
    plt.title(f"Distribution of {col}")
    plt.show()


# TARGET VARIABLE COUNT PLOT
plt.figure(figsize=(6,4))
sns.countplot(x=a['Reached.on.Time_Y.N'])
plt.title("Target Variable Count")
plt.show()

# CONFUSION MATRIX HEATMAP
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ROC CURVE
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()



