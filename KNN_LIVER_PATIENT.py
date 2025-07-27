#THE FOLLOWING CODE IS EXECUTED BY :- SANKET GAIKWAD
# ALGORIITHM :- KNN CKASSIFIER( HYPERTUNNING)
# SOURCE :- Indian Liver Patient Records (https://www.kaggle.com/datasets/uciml/indian-liver-patient-records)



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline # Use imblearn's pipeline for SMOTE in CV

# Load Data
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\liver.csv")
print("Original Columns:", a.columns.tolist())

print("\n--- Data Info ---")
print(a.info())


# Handle Missing Values 
print("\n--- Missing Values Before Imputation ---")
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis=1)]
if not missing_value.empty:
    print("\nRows with Missing Values:")
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")

# Impute missing 'Albumin_and_Globulin_Ratio' with median
a['Albumin_and_Globulin_Ratio'] = a['Albumin_and_Globulin_Ratio'].fillna(a['Albumin_and_Globulin_Ratio'].median())
print("\nMissing Values After Imputation:")
print(a.isnull().sum())

# Encode Categorical Features
print("\n--- Encoding 'Gender' ---")
le = LabelEncoder()
a['Gender'] = le.fit_transform(a['Gender'])
print("Gender unique values after encoding:", a['Gender'].unique())

# Define Features (x) and Target (y)
x = a.drop('Dataset', axis=1)
y = a['Dataset']

# Display class distribution before SMOTE
print("\nClass distribution before SMOTE (original dataset):")
print(y.value_counts())

# Split Data into Training and Testing Sets
# Stratified split to maintain class distribution in train/test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)


print(f"\nTraining set size: {len(x_train)}")
print(f"Test set size: {len(x_test)}")
print("Class distribution in training set:")
print(y_train.value_counts())
print("Class distribution in test set:")
print(y_test.value_counts())


# Build a Pipeline for Preprocessing and Model Training 
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)), # Keep 95% of variance
    ('smote', SMOTE(random_state=1)), # SMOTE will be applied only on training folds
    ('knn', KNeighborsClassifier())
])

#  Hyperparameter Tuning using GridSearchCV 
param_grid = {
    'knn__n_neighbors': range(1, 21, 2), # Odd numbers from 1 to 19
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan'] # Common distance metrics for KNN
}

# Use StratifiedKFold for cross-validation to preserve class balance
cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

print("\n--- Performing GridSearchCV for Hyperparameter Tuning ---")
grid_search = GridSearchCV(pipeline, param_grid, cv=cv_strategy, scoring='f1_weighted', n_jobs=-1, verbose=1)
grid_search.fit(x_train, y_train) # Fit on the original (un-SMOTEd) training data

print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best cross-validation F1-weighted score: {grid_search.best_score_:.4f}")

# Get the best model
best_model = grid_search.best_estimator_

# Make Predictions on the Test Set with the Best Model 
y_pred = best_model.predict(x_test)
print("\n--- Predictions on Test Set ---")
print(y_pred)

# Evaluate the Best Model
print("\n--- EVALUATION OF BEST MODEL ---")
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

print(f"\nACCURACY SCORE:\n {acc:.4f}")
print("CONFUSION MATRIX:\n", cm)
print("CLASSIFICATION REPORT:\n", clr)

print("\n--- COMPARISON (Actual vs. Predict) ---")
comparison = pd.DataFrame({'Actual': y_test, 'Predict': y_pred})
print(comparison.head()) # print only head for brevity

# 10. Cross-Validation Score of the Best Model (using the pipeline) 
print("\n--- CROSS-VALIDATION SCORES (using best pipeline) ---")
# Use the same cross-validation strategy for consistency
scores_best_model = cross_val_score(best_model, x, y, cv=cv_strategy, scoring='f1_weighted', n_jobs=-1)
print("Individual Cross-validation F1-weighted scores:\n", scores_best_model)
print(f"THE MEAN F1-weighted SCORE (Cross-Validation): {np.mean(scores_best_model):.4f}")

# Plotting Confusion Matrix for better visualization
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted Class 1", "Predicted Class 2"],
            yticklabels=["Actual Class 1", "Actual Class 2"])
plt.title('Confusion Matrix for Best KNN Model')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Visualize the F1-scores from the classification report
report_dict = classification_report(y_test, y_pred, output_dict=True)
f1_scores = [report_dict['1']['f1-score'], report_dict['2']['f1-score']]
classes = ['Class 1 (Liver Patient)', 'Class 2 (Non-Liver Patient)']

plt.figure(figsize=(8, 5))
sns.barplot(x=classes, y=f1_scores, palette='viridis')
plt.title('F1-Scores for Each Class (Best KNN Model)')
plt.ylabel('F1-Score')
plt.ylim(0, 1)
plt.show()