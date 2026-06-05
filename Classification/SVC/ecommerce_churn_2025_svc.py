#NAME :- SANKET GAIKWAD
#ALGORITHM :- SUPPORT VECTOR MACHINE ( SVC )
#SOURCE :- https://www.kaggle.com/datasets/nabihazahid/e-commerce-customer-insights-and-churn-dataset





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC

# Load dataset
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\Ecommerce.csv")
print(a.columns.tolist())

# Convert date columns to datetime
a['signup_date'] = pd.to_datetime(a['signup_date'], dayfirst=True, errors='coerce')
a['last_purchase_date'] = pd.to_datetime(a['last_purchase_date'], dayfirst=True, errors='coerce')
a['order_date'] = pd.to_datetime(a['order_date'], dayfirst=True, errors='coerce')

# Extract year and month from date columns
a['signyear'] = a['signup_date'].dt.year
a['signmonth'] = a['signup_date'].dt.month
a['lastyear'] = a['last_purchase_date'].dt.year
a['lastmonth'] = a['last_purchase_date'].dt.month
a['orderyear'] = a['order_date'].dt.year
a['ordermonth'] = a['order_date'].dt.month

# Map subscription status to numerical values
a['subscription_status'] = a['subscription_status'].map({'active':0, 'paused':1, 'cancelled':2})

# Drop unnecessary columns
a = a.drop(['order_id', 'customer_id', 'product_id', 'product_name', 'signup_date', 'last_purchase_date', 'order_date',
            'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20'], axis=1)

# Check for duplicates
print("TOTAL SUM OF DUPLICATED VALUES :-\n", a.duplicated().sum())

# Check for missing values
print("MISSING VALUES :-\n", a.isnull().sum())

# Fill missing values in 'signyear' and 'signmonth' with mode
for col in ['signyear', 'signmonth']:
    a[col] = a[col].fillna(a[col].mode()[0])

# Identify categorical and continuous columns
cat = [col for col in a.columns if a[col].dtype == 'object']
con = [col for col in a.columns if a[col].dtype != 'object']

# One-hot encode categorical variables
a = pd.get_dummies(a, columns=cat, drop_first=True)

# Display class distribution
print("Class Distribution:\n", a['subscription_status'].value_counts(normalize=True))

# Display correlation matrix
print("Correlation Matrix:\n", a.corr(numeric_only=True)['subscription_status'].sort_values(ascending=False))

# Separate features and target variable
x = a.drop('subscription_status', axis=1)
y = a['subscription_status']

# Standardize features
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

# Apply PCA to retain 95% variance
pca = PCA(n_components=0.95)
x_pca = pca.fit_transform(x_scaled)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=1)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

# Optimized SVC model for better accuracy, precision, recall
model = SVC(C=10.0, kernel='rbf', gamma=0.01, degree=3, random_state=1)
model.fit(x_train_res, y_train_res)

# Make predictions
y_pred = model.predict(x_test)
print(y_pred)

# Evaluate model performance
print("\nEVALUATION")
print("ACCURACY SCORE:-\n", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX :-\n", confusion_matrix(y_test, y_pred))
print("CLASSIFICATION REPORT :-\n", classification_report(y_test, y_pred))

# Display comparison between actual and predicted values
print("\nCOMPARISON")
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

# Perform cross-validation
print("\nCROSS SCORE VALIDATION")
scores = cross_val_score(model, x_pca, y, cv=10)
print(scores)
print("THE MEAN SCORES :-\n", np.mean(scores))

# Optional Visualizations
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Active','Paused','Cancelled'], yticklabels=['Active','Paused','Cancelled'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

sns.countplot(x='subscription_status', data=a)
plt.title('Class Distribution')
plt.show()
