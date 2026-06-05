#THE CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- RANDOM FOREST CLASSIFIER
#SOURCE :- https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

a = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\Desktop\\My Data\\heart.csv")
print(a.columns.tolist())

print()
print(a.info())

print()
print(a.describe())

print()
print("DUPLICATED VALUES ")
print(a.duplicated().sum())

print()
c = a[a.duplicated()]
print(c)

print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")

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

print()
a = pd.get_dummies(a, columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'],drop_first=True)

le = LabelEncoder()
a['HeartDisease'] = le.fit_transform(a['HeartDisease'])


#FEATURE SELECTION
print()
x = a.drop('HeartDisease', axis = 1)
y = a['HeartDisease']


#SPLIT OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=1)


# CALCULATES AND DISPLAYS THE PROPORTION OF EACH CLASS IN THE HEART DISEASE TARGET VARIABLE
print()
print(a['HeartDisease'].value_counts(normalize=True))


# COMPUTES AND DISPLAYS THE CORRELATION OF ALL NUMERICAL FEATURES WITH THE HEART DISEASE TARGET VARIABLE
print()
print(a.corr(numeric_only=True)['HeartDisease'].sort_values(ascending=False))


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca =pca.fit_transform(x_scaled)


#MODEL BUILDING AND AND EXECUTION
print()
model = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', class_weight='balanced', random_state=1)
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
print("MEAN SCORES :-\n", np.mean(scores))


# CONFUSION MATRIX HEATMAP
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("PREDICTED")
plt.ylabel("ACTUAL")
plt.title("CONFUSION MATRIX")
plt.show()

# ACTUAL VS PREDICTED BAR PLOT
comparison_plot = comparison.reset_index(drop=True).head(30)

plt.figure(figsize=(12,5))
plt.plot(comparison_plot['Actual'].values, label='ACTUAL', marker='o')
plt.plot(comparison_plot['Predict'].values, label='PREDICTED', marker='x')
plt.xlabel("OBSERVATIONS")
plt.ylabel("CLASS LABEL")
plt.title("ACTUAL VS PREDICTED VALUES")
plt.legend()
plt.show()


# CLASS DISTRIBUTION IN TARGET VARIABLE
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.xlabel("HEART DISEASE CLASS")
plt.ylabel("COUNT")
plt.title("TARGET VARIABLE DISTRIBUTION")
plt.show()


# FEATURE IMPORTANCE PLOT
importances = model.feature_importances_
features = x.columns

feature_imp = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=feature_imp)
plt.title("TOP FEATURE IMPORTANCE FROM RANDOM FOREST")
plt.show()


# CROSS VALIDATION SCORE DISTRIBUTION
plt.figure(figsize=(7,4))
plt.plot(scores, marker='o')
plt.axhline(np.mean(scores), color='red', linestyle='--', label='MEAN SCORE')
plt.xlabel("FOLDS")
plt.ylabel("ACCURACY SCORE")
plt.title("CROSS VALIDATION ACCURACY SCORES")
plt.legend()
plt.show()

# PREDICTED PROBABILITY DISTRIBUTION
plt.figure(figsize=(7,4))
sns.histplot(y_pred_prob, bins=20, kde=True)
plt.xlabel("PREDICTED PROBABILITY")
plt.ylabel("FREQUENCY")
plt.title("PREDICTED PROBABILITY DISTRIBUTION")
plt.show()


