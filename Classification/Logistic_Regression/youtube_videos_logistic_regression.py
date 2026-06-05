# THE FOLLOWING CODE IS EXECUTED BY:- SANKET GAIKWAD
# ALGORITHM :- LOGISTIC REGRESSIOn
#SOURCE :- KAGGLE ( https://www.kaggle.com/datasets/cyberevil545/youtube-videos-data-for-ml-and-trend-analysis )


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\utube25.csv")
print(a.columns.tolist())


# CREATED A NEW COLUMN 'engagement' BASED ON VIEW RANGES CLASSIFIED INTO 'Low', 'Medium', AND 'High'
print()
conditions = [
    a['views'] < 100000,
    (a['views'] >= 100000) & (a['views'] < 1000000),
    a['views'] >= 1000000
]

choices = ['Low', 'Medium', 'High']

a['engagement'] = np.select(conditions, choices, default='Low')


# CREATED A NEW COLUMN 'resolution' BY MULTIPLYING VIDEO HEIGHT AND WIDTH
print()
a['resolution'] = a['height'] * a['width']


# CONVERTED THE 'duration' COLUMN FROM TIME FORMAT TO TOTAL SECONDS AND STORED IT IN 'duration_seconds'
print()
a['duration_seconds'] = pd.to_timedelta(a['duration']).dt.total_seconds()


#DROPPING UNWANTED FEATURES
print()
a = a.drop(['video_id', 'url', 'title', 'likes', 'hashtags', 'description', 'comments', 'bitrate(video)', 'width', 'views', 'height','duration'], axis = 1)


#INFORMATION ABOUT DATASET 
print()
print(a.info())


print()
print(a.describe())


#SEARCHING FOR TOTAL SUM OF DUPLICATE VALUES
print()
print("TOTAL DUPLICATE VALUES")
print(a.duplicated().sum())


#DUPLICATED VALUES ROW DISPLAY
print()
c = a[a.duplicated()]
print(c)


#DROP DUPLICATE VALUES
print()
a.drop_duplicates()


#SEARCHING FOR NULL VALUES 
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")



# PERFORMED ONE-HOT ENCODING ON 'codec' AND 'category' COLUMNS AND DROPPED THE FIRST DUMMY TO AVOID MULTICOLLINEARITY
print()
a = pd.get_dummies(a, columns = ['codec', 'category'], drop_first = True)


#LABEL ENCODING OF TARGET VARIABLE
le = LabelEncoder()
a['engagement'] = le.fit_transform(a['engagement'])


#FEATURE SELECTION
print()
x = a.drop('engagement', axis = 1)
y = a['engagement']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components = 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLIT OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state= 1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
acc = accuracy_score(y_test, y_pred)
cm =confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

print()
print("ACCURACY SCORE:-\n", acc)
print("CONFUSION MATRIXX :-\n", cm)
print("CLASSIFICATION REPORT :-\n", clr)


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print("CROSS SCORE VALIDATION")
scores = cross_val_score(model,x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALISATION

# 1. CLASS DISTRIBUTION BEFORE SMOTE
plt.figure(figsize=(6,4))
sns.countplot(x=y, palette='viridis')
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Engagement Level')
plt.ylabel('Count')
plt.show()


# 2. CLASS DISTRIBUTION AFTER SMOTE
plt.figure(figsize=(6,4))
sns.countplot(x=y_train_res, palette='magma')
plt.title('Class Distribution After SMOTE')
plt.xlabel('Engagement Level')
plt.ylabel('Count')
plt.show()


# 3. CONFUSION MATRIX HEATMAP
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# 4. PCA EXPLAINED VARIANCE RATIO
plt.figure(figsize=(6,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title('PCA Explained Variance Ratio')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()


# 5. FEATURE DISTRIBUTION (DURATION, RESOLUTION)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(a['duration_seconds'], bins=30, kde=True, color='teal')
plt.title('Distribution of Duration (Seconds)')

plt.subplot(1,2,2)
sns.histplot(a['resolution'], bins=30, kde=True, color='orange')
plt.title('Distribution of Resolution')
plt.tight_layout()
plt.show()


# 6. COMPARISON OF ACTUAL VS PREDICTED (BAR PLOT)
comparison_df = comparison.head(50).reset_index(drop=True)  # first 50 samples for visualization
plt.figure(figsize=(12,5))
plt.plot(comparison_df.index, comparison_df['Actual'], marker='o', label='Actual')
plt.plot(comparison_df.index, comparison_df['Predict'], marker='x', label='Predicted')
plt.title('Actual vs Predicted Engagement Levels (Sample)')
plt.xlabel('Sample Index')
plt.ylabel('Engagement Level')
plt.legend()
plt.show()


