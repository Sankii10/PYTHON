#THE CODE IS EXECUTED BY ;- SANKET GAIKWAD
#ALGORITHM USED :- RANDOM FOREST CLASSIFIER
#SOURCE :- https://www.kaggle.com/datasets/kundanbedmutha/hotel-booking-reservation



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


#IMPORTING DATA FROM FILE
a = pd.read_csv('C:\\Users\\ASUS\\Desktop\\My Data\\hotel.csv')
print(a.columns.tolist())


# PRINT THE PROPORTION OF EACH UNIQUE VALUE IN THE 'BABIES' COLUMN
print()
print(a['babies'].value_counts(normalize= True))


#DROPING UNREQUIRED FEATURES
print()
a = a.drop(['hotel', 'city', 'reservation_status', 'reservation_status_date', 'assigned_room_type','booking_changes', 'company', 'agent','country', 'babies'],axis = 1)


#INFORMATION ABOUT DATA
print()
print(a.info())


#STATISTICAL DESCRIPTION OF DATA
print()
print(a.describe())


#TOATL SUM OF DUPLICATED VALUES
print()
print("TOTAL NUMBER OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())


#DISPLAY DUPLICATED VALUES 
print()
c = a[a.duplicated()]
print(c)


#DROPING THE DUPLICATED VALUES
print()
a = a.drop_duplicates()


#SEARCHING FOR MISSING AND NULL VALUES
print()
b = a.isnull().sum()
print(b)

missing_value =a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")

print()
a['children'] = a['children'].fillna(a['children'].mode()[0])


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
print("CONTINOUS VALUE :-\n", con)


#ONE HOT ENCODING
print()
a = pd.get_dummies(a,columns = ['arrival_date_month', 'meal', 'market_segment', 'distribution_channel', 'reserved_room_type', 'deposit_type', 'customer_type'], drop_first= True)


# PRINT THE PROPORTION OF EACH UNIQUE VALUE IN THE 'IS CANCELLED' COLUMN
print()
print(a['is_canceled'].value_counts(normalize = True))


# PRINT THE CORRELATION VALUES OF ALL NUMERIC FEATURES WITH THE TARGET VARIABLE 'IS_CANCELED' IN DESCENDING ORDER
print()
print(a.corr(numeric_only=True)['is_canceled'].sort_values(ascending = False))


#TARGET FEATURE SELECTION
print()
x = a.drop('is_canceled', axis = 1)
y = a['is_canceled']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components = 0.95)
x_pca =pca.fit_transform(x_scaled)

#SPLIT OF DATA AS TRAIN AND TEST
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size= 0.2, random_state= 1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING AND IMPLEMENTATION
print()
model = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=8, min_samples_leaf=4, max_features='sqrt', bootstrap=True, oob_score=True, random_state=1, n_jobs=-1)
model.fit(x_train_res, y_train_res)

y_pred_prob = model.predict_proba(x_test)[:,1]
y_pred = (y_pred_prob > 0.5).astype(int)
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
kf = KFold(n_splits= 10, shuffle = True, random_state= 1)
scores = cross_val_score(model,x_pca, y, cv = kf, scoring = 'accuracy')
print(scores)
print("MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALISATION


# 1. TARGET VARIABLE DISTRIBUTION
plt.figure(figsize=(6,4))
sns.countplot(data=a, x='is_canceled')
plt.title("Distribution of Target Variable (is_canceled)")
plt.show()

# 2. HEATMAP OF CORRELATIONS
plt.figure(figsize=(14,10))
sns.heatmap(a.corr(numeric_only=True), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 3. FEATURE IMPORTANCE USING RANDOM FOREST
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feat_names = x.columns

plt.figure(figsize=(12,8))
sns.barplot(x=importances[indices][:20], y=feat_names[indices][:20])
plt.title("Top 20 Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# 4. PCA EXPLAINED VARIANCE
plt.figure(figsize=(6,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid(True)
plt.show()

# 5. CONFUSION MATRIX (VISUAL)
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
