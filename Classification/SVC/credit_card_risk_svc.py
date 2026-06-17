"""
THE FOLLOWING CODE IS WRITTEN BY :-
NAME :- SANKET GAIKWAD
DATASET :- CREDIT RISK DATASET(SOURCE :- KAGGLE) https://www.kaggle.com/datasets/laotse/credit-risk-dataset

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  
 
 
# LOAD DATA

a = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\Desktop\\credit_risk_dataset.csv")
print("Columns:", a.columns.tolist())
 
a = a.drop('loan_percent_income', axis=1)
 
print("\n--- DATA INFO ---")
print(a.info())
 
print("\n--- MISSING VALUES ---")
print(a.isnull().sum())
 
 

# HANDLE MISSING VALUES

for col in ['person_emp_length', 'loan_int_rate']:
    a[col] = a[col].fillna(a[col].median())
 

# DEFINE FEATURE TYPES


ordinal_cols = ['loan_grade']
nominal_cols = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']
numeric_cols = ['person_age', 'person_income', 'person_emp_length',
                'loan_amnt', 'loan_int_rate', 'cb_person_cred_hist_length']
 
# Encode ordinal column
le = LabelEncoder()
a['loan_grade'] = le.fit_transform(a['loan_grade'])
 
x = a.drop('loan_status', axis=1)
y = a['loan_status']
 
print("\n--- CLASS DISTRIBUTION ---")
print(y.value_counts())
 
 

# 4. TRAIN/TEST SPLIT

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1, stratify=y
)
 
 
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_cols + ordinal_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_cols)
])
 
x_train_prep = preprocessor.fit_transform(x_train)   # fit only on train
x_test_prep  = preprocessor.transform(x_test)         # transform only
 
 

# PCA: fit on train, transform both

pca = PCA(n_components=0.95)
x_train_pca = pca.fit_transform(x_train_prep)   # fit only on train
x_test_pca  = pca.transform(x_test_prep)         # transform only
 
print(f"\nPCA retained {pca.n_components_} components to explain 95% variance")
 
 

# SMOTE: apply only on training data

smote = SMOTE(random_state=1)
x_train_res, y_train_res = smote.fit_resample(x_train_pca, y_train)
 
print(f"\nAfter SMOTE. Train class distribution:")
print(pd.Series(y_train_res).value_counts())
 
 

# MODEL TRAINING

model = SVC(kernel='rbf', class_weight='balanced', random_state=1)
model.fit(x_train_res, y_train_res)
 
y_pred = model.predict(x_test_pca)
 
 

# EVALUATION

print("\n" + "="*50)
print("EVALUATION ON HOLD-OUT TEST SET")
print("="*50)
 
acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)
 
print(f"\nACCURACY SCORE:\n {acc:.4f}")
print(f"\nCONFUSION MATRIX:\n {cm}")
print(f"\nCLASSIFICATION REPORT:\n{clr}")
 
comparison = pd.DataFrame({'Actual': y_test, 'Predict': y_pred})
print("\nCOMPARISON (first 20 rows):")
print(comparison.head(20))
 
 

# CROSS-VALIDATION using Pipeline 

 
cv_pipeline = ImbPipeline(steps=[
    ('preprocessor', ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_cols + ordinal_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_cols)
    ])),
    ('pca',   PCA(n_components=0.95)),
    ('smote', SMOTE(random_state=1)),
    ('svm',   SVC(kernel='rbf', class_weight='balanced', random_state=1))
])
 
print("\n" + "="*50)
print("CROSS-VALIDATION (10-fold, leak-free)")
print("="*50)
 
cv_scores = cross_val_score(cv_pipeline, x, y, cv=10, scoring='accuracy', n_jobs=-1)
print(f"\nFold Scores:\n {np.round(cv_scores, 4)}")
print(f"\nMean CV Accuracy: {np.mean(cv_scores):.4f}")
print(f"Std CV Accuracy:  {np.std(cv_scores):.4f}")
 
 

#VISUALIZATIONS

 
# Plot 1: Class distribution before SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x='loan_status', data=a)
plt.title('Loan Status Distribution (Before SMOTE)')
plt.xlabel('Loan Status (0 = No Default, 1 = Default)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
 
# Plot 2: PCA explained variance
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', color='steelblue')
plt.axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
 
# Plot 3: Confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.title('Confusion Matrix (Hold-out Test Set)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
 
# Plot 4: Actual vs Predicted (sample of 50)
sample_comparison = comparison.sample(50, random_state=1).reset_index(drop=True)
plt.figure(figsize=(12, 4))
plt.plot(sample_comparison['Actual'],  label='Actual',    marker='o', linestyle='-')
plt.plot(sample_comparison['Predict'], label='Predicted', marker='x', linestyle='--')
plt.title('Sample Comparison: Actual vs Predicted (50 samples)')
plt.xlabel('Sample Index')
plt.ylabel('Loan Status')
plt.legend()
plt.tight_layout()
plt.show()
 
# Plot 5: CV score distribution
plt.figure(figsize=(8, 4))
plt.bar(range(1, 11), cv_scores, color='steelblue', edgecolor='black')
plt.axhline(y=np.mean(cv_scores), color='red', linestyle='--', label=f'Mean = {np.mean(cv_scores):.3f}')
plt.title('10-Fold Cross-Validation Accuracy Scores')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
 
