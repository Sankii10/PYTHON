#THE CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- RANDOM FOREST CLASSIFIER
#SOURCE :- https://www.kaggle.com/datasets/mayuringle8890/indian-credit-card-datasetrewards-fees-usage


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



#IMPORT DATA 
a = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\Desktop\\My Data\\CC.csv")
print(a.columns.tolist())


#DROPING THE UNREQUIRED VARIABLE
print()
a = a.drop('credit_card_name', axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION DATA
print()
print(a.info())

print()
print(a.describe())


#SEARCHING DUPLICATED VALUES
print()
print("DUPLICATED VALUES !!")
print(a.duplicated().sum())

print()
c = a[a.duplicated()]
print(c)


#SEARCHING FOR MISSING VALUES
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!")


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
print("CATEGORICAL COLUMNS :-\n", cat)
print("CONTINOUS COLUMNS :-\n", con)


#ONE HOT ENCODING
print()
a = pd.get_dummies(a, columns = ['speciality', 'primary_usage_category', 'bank_name'], drop_first=True)


#TARGET VARIABLE LABEL ENCODING
le = LabelEncoder()
a['reward_type'] = le.fit_transform(a['reward_type'])


#FEATURE SELECTION
print()
x = a.drop('reward_type', axis = 1)
y = a['reward_type']


#SPLIT OF DATA AS TRAIN AND TEST
print()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# THIS LINE CALCULATES AND DISPLAYS THE PROPORTIONAL DISTRIBUTION OF EACH CLASS IN THE TARGET VARIABLE (REWARD_TYPE) TO CHECK CLASS BALANCE BEFORE MODELING
print()
print(a['reward_type'].value_counts(normalize=True))


# THIS LINE COMPUTES THE CORRELATION OF ALL NUMERIC FEATURES WITH THE TARGET VARIABLE (REWARD_TYPE) AND SORTS THEM IN DESCENDING ORDER TO IDENTIFY THE MOST INFLUENTIAL FEATURES
print()
print(a.corr(numeric_only=True)['reward_type'].sort_values(ascending=False))


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca = pca.fit_transform(x_scaled)



#MODEL BUILDING AND IMPLEMENTATION
print()
model = RandomForestClassifier(n_estimators=800, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, class_weight='balanced_subsample', random_state=1, n_jobs=-1)
model.fit(x_train, y_train)

y_pred_prob = model.predict_proba(x_test)
print(y_pred_prob)

print()
y_pred_prob = pd.DataFrame(y_pred_prob, columns = ['cashback','shopping_rewards','fuel_rewards','dining_rewards','travel_points','premium_points'])
pd.options.display.float_format = '{:.3f}'.format
print(y_pred_prob)

print()
y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
print("ACCURACY SCORE :-\n", accuracy_score(y_test , y_pred))
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
scores = cross_val_score(model, x, y, cv=kf, scoring='accuracy')
print(scores)
print("MEAN SCORES :-\n", np.mean(scores))


# TARGET VARIABLE CLASS DISTRIBUTION
plt.figure(figsize=(8,5))
sns.countplot(x=y)
plt.title("TARGET VARIABLE (REWARD_TYPE) CLASS DISTRIBUTION")
plt.xlabel("REWARD TYPE (ENCODED)")
plt.ylabel("COUNT")
plt.show()

# DISTRIBUTION OF POINTS ACCUMULATION RATE
plt.figure(figsize=(8,5))
sns.histplot(a['points_accumulation_rate'], kde=True)
plt.title("DISTRIBUTION OF POINTS ACCUMULATION RATE")
plt.xlabel("POINTS ACCUMULATION RATE")
plt.ylabel("FREQUENCY")
plt.show()

# DISTRIBUTION OF ANNUAL CHARGES
plt.figure(figsize=(8,5))
sns.histplot(a['annual_charges_in_inr'], kde=True)
plt.title("DISTRIBUTION OF ANNUAL CHARGES IN INR")
plt.xlabel("ANNUAL CHARGES (INR)")
plt.ylabel("FREQUENCY")
plt.show()


# CORRELATION HEATMAP FOR NUMERIC FEATURES
plt.figure(figsize=(10,6))
sns.heatmap(a.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("CORRELATION HEATMAP OF NUMERIC FEATURES")
plt.show()

# CONFUSION MATRIX HEATMAP
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("CONFUSION MATRIX HEATMAP")
plt.xlabel("PREDICTED LABEL")
plt.ylabel("ACTUAL LABEL")
plt.show()

# FEATURE IMPORTANCE FROM RANDOM FOREST
importances = model.feature_importances_
feature_names = x.columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df)
plt.title("TOP 15 FEATURE IMPORTANCES FROM RANDOM FOREST")
plt.xlabel("IMPORTANCE SCORE")
plt.ylabel("FEATURE")
plt.show()

