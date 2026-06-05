#THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- SUPPORT VECTOR MACHINE [ SVC ]
#SOURCE :- https://www.kaggle.com/datasets/albertobircoci/support-ticket-priority-dataset-50k



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


#IMPORTING DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\tickets.csv")
print(a.columns.tolist())


# CONVERTED PRIORITY COLUMN FROM CATEGORICAL (LOW, MEDIUM, HIGH) TO NUMERICAL (0, 1, 2)
print()
a['newpriority'] = a['priority'].map({'low':0, 'medium':1, 'high':2})


#DROPPING THE UNREQUIRED FEATURES
print()
a = a.drop(['priority','ticket_id', 'company_id'], axis = 1)


#INFORMATION AND STATISTICAL DESCRIPTION OF DATA
print()
print(a.info())

print()
print(a.describe())


#FINDING TOTAL SUM OF DUPLICATED VALUES IN DATA
print()
print("TOTAL SUM OF DUPLICATED VALUES :-")
print(a.duplicated().sum())


#DISPLAYING ROW WISE DUPLICATED DATA
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
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#HANDLING THE MISSING VALUES
print()
for col in ['newpriority','customer_sentiment']:
    a[col] = a[col].fillna(a[col].mode()[0])


#ONE HOT ENCODING
print()
a = pd.get_dummies(a, columns = ['day_of_week', 'company_size', 'industry', 'customer_tier', 'product_area', 'booking_channel', 'reported_by_role', 'customer_sentiment', 'region'], drop_first = True)


#FEATURES SELECTION 
print()
x = a.drop('newpriority', axis = 1)
y = a['newpriority']


#STANDARD SCALING
print()
sc=  StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca = pca.fit_transform(x_scaled)


#SPLITING OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size= 0.2, random_state=1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model = SVC(kernel='rbf', gamma='auto', probability=True)
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
print()
print("ACCURACY SCORE:-\n", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX :-\n", confusion_matrix(y_test, y_pred))
print("CLASSIFICATION REPORT :-\n", classification_report(y_test, y_pred))


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDAATION")
scores = cross_val_score(model,x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALISATION

# 1. Priority Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='newpriority', data=a)
plt.title('Distribution of Ticket Priority')
plt.xlabel('Priority (0=Low, 1=Medium, 2=High)')
plt.ylabel('Count')
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(a.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 3. Feature Importance from PCA (Variance Explained)
plt.figure(figsize=(8,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_)*100, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('PCA - Cumulative Explained Variance')
plt.grid(True)
plt.show()

# 4. Pairplot of selected numeric features colored by priority
numeric_cols = ['org_users', 'past_30d_tickets', 'past_90d_incidents', 'customers_affected', 'description_length']
sns.pairplot(a[numeric_cols + ['newpriority']], hue='newpriority', palette='Set1', corner=True)
plt.suptitle('Pairplot of Key Numeric Features by Priority', y=1.02)
plt.show()

# 5. Confusion Matrix Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

# 6. Actual vs Predicted comparison scatter (good for multi-class check)
plt.figure(figsize=(6,4))
plt.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, label='Predicted', alpha=0.6)
plt.title('Actual vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Priority')
plt.legend()
plt.show()

# 7. Distribution of numeric features after scaling
plt.figure(figsize=(12,6))
sns.histplot(pd.DataFrame(x_scaled, columns=x.columns), kde=True, bins=30)
plt.title('Distribution of Scaled Features')
plt.show()

