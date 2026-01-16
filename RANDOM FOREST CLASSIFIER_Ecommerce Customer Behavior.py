#THE CODE IS EXECUTED BY MR.SANKET GAIKWAD
#ALGORITHM USED :- RANDOM FOREST CLASSIFIER
#SOURCE :- https://www.kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#IMPORTING THE DATA
a = pd.read_csv('C:\\Users\\ASUS\\OneDrive\\Desktop\\My Data\\ecom.csv')
print(a.columns.tolist())


#CREATION OF NEW VARIABLE AS observed_ltv 
print()
a['observed_ltv'] = a['Total_Purchases'] * a['Average_Order_Value']


#DROPPING NOT REQUIRED FEATURES
print()
a = a.drop('Lifetime_Value', axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION OF DATA
print()
print(a.info())

print()
print(a.describe())


#DUPLICATED VALUES IN DATASET
print()
print("DUPLICATED VALUES ")
print(a.duplicated().sum())

print()
c = a[a.duplicated()]
print(c)


#MISSING VALUES IN DATASET
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


#REPLACING THE MISSING VALUES WITH MEDIAN
print()
for col in ['Age', 'Session_Duration_Avg', 'Pages_Per_Session', 'Wishlist_Items','Days_Since_Last_Purchase', 'Discount_Usage_Rate', 'Returns_Rate', 'Email_Open_Rate', 'Customer_Service_Calls', 'Product_Reviews_Written', 'Social_Media_Engagement_Score', 'Mobile_App_Usage', 'Payment_Method_Diversity','Credit_Balance']:
    a[col] = a[col].fillna(a[col].median())


#ONE HOT ENCODING
print()
a = pd.get_dummies(a, columns = ['Gender', 'Country', 'City', 'Signup_Quarter'], drop_first=True)


#FEATURE SELECTION
print()
x = a.drop('Churned', axis = 1)
y = a['Churned']


#SPLITING OF DATA AS TRAIN AND TEST
print()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 1)


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca =PCA(n_components= 0.95)
x_pca =pca.fit_transform(x_scaled)


#MODEL BUILDING AND IMPLEMENTATION
print()
model = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=10, min_samples_leaf=5, max_features='sqrt', class_weight='balanced', random_state=1, n_jobs=-1)
model.fit(x_train, y_train)

y_pred_prob = model.predict_proba(x_test)[:,1]
y_pred = (y_pred_prob >= 0.5).astype(int)
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
kf = KFold(n_splits = 10, shuffle = True, random_state= 1)
scores = cross_val_score(model,x_pca, y, cv = kf, scoring = 'accuracy')
print(scores)
print("MEAN SCORES:-\n", np.mean(scores))


# CLASS DISTRIBUTION OF TARGET VARIABLE
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Churn Distribution")
plt.xlabel("Churned")
plt.ylabel("Count")
plt.show()


# CONFUSION MATRIX HEATMAP
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# FEATURE IMPORTANCE PLOT FROM RANDOM FOREST
importances = model.feature_importances_
feature_names = x.columns

feat_imp = pd.Series(importances, index=feature_names)
feat_imp = feat_imp.sort_values(ascending=False).head(15)

plt.figure(figsize=(8,6))
feat_imp.plot(kind='barh')
plt.title("Top 15 Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.gca().invert_yaxis()
plt.show()


# ROC CURVE
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()


# PRECISION VS RECALL CURVE
from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

plt.figure(figsize=(6,5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()


# ACTUAL VS PREDICTED COUNTS
comparison['Actual'].value_counts().plot(kind='bar', alpha=0.7, label='Actual')
comparison['Predict'].value_counts().plot(kind='bar', alpha=0.7, label='Predicted')
plt.title("Actual vs Predicted Churn Count")
plt.legend()
plt.show()
