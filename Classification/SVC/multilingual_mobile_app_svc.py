# THE FOLLOWING CODE IS EXECUTED BY:- MR.SANKET GAIKWAD
# ALGORITHM USED :- SVC 
# SOURCE :- KAGGLE ( https://www.kaggle.com/datasets/pratyushpuri/multilingual-mobile-app-reviews-dataset-2025/data )


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from scipy.sparse import hstack
from imblearn.over_sampling import SMOTE
from scipy.sparse import csr_matrix

# LOAD DATASET
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\app.csv")

# CREATE TARGET VARIABLE (POSITIVE IF RATING > 3, ELSE NEGATIVE)
a['newrating'] = a['rating'].apply(lambda x: 'Positive' if x > 3 else 'Negative')

# CONVERT DATE AND EXTRACT YEAR, MONTH
a['review_date'] = pd.to_datetime(a['review_date'], dayfirst=True, errors='coerce')
a['year'] = a['review_date'].dt.year
a['month'] = a['review_date'].dt.month

# DROP UNNECESSARY COLUMNS
a = a.drop(['rating', 'review_date', 'app_version', 'review_id', 'user_id', 'app_name'], axis=1)

# HANDLE MISSING VALUES
for col in ['review_text', 'year', 'month', 'user_country', 'user_gender']:
    a[col] = a[col].fillna(a[col].mode()[0])

# TF-IDF FOR REVIEW_TEXT
tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
text_features = tfidf.fit_transform(a['review_text'])

# DROP TEXT FROM STRUCTURED FEATURES
a = a.drop(['review_text'], axis=1)

# ONE-HOT ENCODE CATEGORICAL VARIABLES
a = pd.get_dummies(a, columns=['app_category', 'review_language', 'verified_purchase', 'device_type', 'user_country', 'user_gender'], drop_first=True)

# ENCODE TARGET
le = LabelEncoder()
a['newrating'] = le.fit_transform(a['newrating'])

# SEPARATE FEATURES AND TARGET
x = a.drop('newrating', axis=1)
y = a['newrating']

# SCALE NUMERIC FEATURES AND CONVERT TO SPARSE MATRIX
x = x.astype(float)
x_sparse = csr_matrix(x.values)

# COMBINE TEXT + STRUCTURED FEATURES
X_combined = hstack([text_features, x_sparse])

# TRAIN-TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=1)

# APPLY SMOTE FOR CLASS BALANCING
smote = SMOTE(random_state=1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

# DIMENSIONALITY REDUCTION FOR SVM EFFICIENCY
svd = TruncatedSVD(n_components=300)
x_train_res_svd = svd.fit_transform(x_train_res)
x_test_svd = svd.transform(x_test)

# HYPERPARAMETER TUNING FOR SVM USING GRID SEARCH
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(x_train_res_svd, y_train_res)

# BEST MODEL
model = grid.best_estimator_
print("BEST PARAMS:", grid.best_params_)

# PREDICTIONS
y_pred = model.predict(x_test_svd)

# EVALUATION METRICS
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

print("\nACCURACY SCORE:", acc)
print("\nCONFUSION MATRIX:\n", cm)
print("\nCLASSIFICATION REPORT:\n", clr)

# CROSS-VALIDATION SCORES
scores = cross_val_score(model, svd.transform(X_combined), y, cv=10)
print("\nCROSS-VALIDATION SCORE:", scores)
print("MEAN CV SCORE:", np.mean(scores))

# DATA VISUALIZATION

# CONFUSION MATRIX HEATMAP
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('CONFUSION MATRIX')
plt.xlabel('PREDICTED')
plt.ylabel('ACTUAL')
plt.show()

# CLASS DISTRIBUTION BEFORE SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette='Set2')
plt.title('CLASS DISTRIBUTION BEFORE SMOTE')
plt.xlabel('CLASS')
plt.ylabel('COUNT')
plt.show()

# CLASS DISTRIBUTION AFTER SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train_res, palette='Set1')
plt.title('CLASS DISTRIBUTION AFTER SMOTE')
plt.xlabel('CLASS')
plt.ylabel('COUNT')
plt.show()

# CROSS-VALIDATION SCORES (BAR PLOT)
plt.figure(figsize=(8, 4))
plt.bar(range(1, len(scores)+1), scores)
plt.title('CROSS-VALIDATION ACCURACY SCORES')
plt.xlabel('FOLD')
plt.ylabel('ACCURACY')
plt.ylim(0, 1)
plt.show()

# EXPLAINED VARIANCE BY SVD COMPONENTS
explained_variance = svd.explained_variance_ratio_
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(explained_variance)+1), explained_variance.cumsum(), marker='o')
plt.title('EXPLAINED VARIANCE BY SVD COMPONENTS')
plt.xlabel('NUMBER OF COMPONENTS')
plt.ylabel('CUMULATIVE EXPLAINED VARIANCE')
plt.grid()
plt.show()


