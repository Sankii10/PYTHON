#THE FOLLOWING CODE IS EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM :- SUPPORT VECTOR CLASSIFIER
#SOURCE :- BMW Car Sales Classification Dataset ( https://www.kaggle.com/datasets/sumedh1507/bmw-car-sales-dataset )



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from imblearn.over_sampling import SMOTE


#IMPORTING OF THE DATA
a=  pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\BMW.csv")
print(a.columns.tolist())


#INFORMATION ABOUT THE DATA
print()
print(a.info())

print()
print(a.describe())


#FINDING THE MISSING VALUES
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#LABEL ENCODING
print()
le = LabelEncoder()
for col in ['Model', 'Region', 'Color', 'Fuel_Type', 'Sales_Classification', 'Transmission']:
    a[col] = le.fit_transform(a[col])

#Check correlation with target for leakage
cor_matrix = a.corr()
target_corr = cor_matrix['Sales_Classification'].drop('Sales_Classification')
print("\nCorrelation with target:\n", target_corr)

# Define threshold for leakage detection
leakage_threshold = 0.85
leakage_features = target_corr[abs(target_corr) > leakage_threshold].index.tolist()

if leakage_features:
    print(f"\nRemoving highly correlated potential leakage features: {leakage_features}")
    a = a.drop(columns=leakage_features)
else:
    print("\nNo features above leakage threshold found.")


#FEATURE SELECTIONS
print()
x = a.drop('Sales_Classification', axis = 1)
y = a['Sales_Classification']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca =pca.fit_transform(x_scaled)


#SPLITING OF TRAIN AND TEST DATASET 
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state = 1, stratify=y)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#MODEL BUILDING
print()
model = SVC(kernel='poly', C=0.5, class_weight='balanced')
model.fit(x_train_res, y_train_res)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUDATION")
acc =accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

print()
print("ACCURACY SCORE:-\n", acc)
print("CONFUSION MATRIX :-\n", cm)
print("CLASSIFICATION REPORT :-\n", clr)


#COMPARISON
print()
print("COMPARISON")
comparison =pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model,x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALIZATION

# 1. Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(a.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.show()

# 2. Distribution of target variable
plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette="viridis")
plt.title("Sales Classification Distribution", fontsize=14)
plt.xlabel("Sales Classification")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 3. Boxplot for numerical features (spread & outliers)
num_cols = a.select_dtypes(include=np.number).columns.tolist()
plt.figure(figsize=(12, 6))
a[num_cols].boxplot()
plt.xticks(rotation=45)
plt.title("Numerical Feature Distributions", fontsize=14)
plt.tight_layout()
plt.show()

# 4. PCA Explained Variance Plot
pca_full = PCA().fit(x_scaled)
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Confusion Matrix Heatmap
cm_df = pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y))
plt.figure(figsize=(6, 5))
sns.heatmap(cm_df, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 6. Cross-validation score distribution
plt.figure(figsize=(6, 4))
sns.barplot(x=list(range(1, len(scores)+1)), y=scores, palette="magma")
plt.axhline(np.mean(scores), color='red', linestyle='--', label=f"Mean: {np.mean(scores):.3f}")
plt.title("Cross-validation Scores per Fold", fontsize=14)
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()



