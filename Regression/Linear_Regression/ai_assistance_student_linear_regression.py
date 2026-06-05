#THE FOLLOWING CODE WAS ECXECUTED BY:- MR. SANKET GAIKWAD
#ALGORITHM :- LINEAR REGRESSOR(RIDGE AND GRIDSEARCH)
#SOURCE :- AI Assistant Usage in Student Life (https://www.kaggle.com/datasets/ayeshasal89/ai-assistant-usage-in-student-life-synthetic)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\ai.csv")
print(a.columns.tolist())


#DROPPING UNWANTED COLUMNS
print()
a = a.drop(['SessionID','SessionDate'],axis = 1)


#INFORMATION ABOUT THE DATASET
print()
print(a.info())


#SEARCHING MISSING VALUES
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
for col in ['StudentLevel', 'Discipline','TaskType', 'FinalOutcome', 'UsedAgain']:
    a[col] = le.fit_transform(a[col])


#FEATURE SELECTION
print()
x = a.drop('SatisfactionRating', axis = 1)
y = a['SatisfactionRating']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCIPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca = pca.fit_transform(x_scaled)


#SPLIT OF TRAIN AND TEST DATA
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state= 1)


#MODEL BUILDING
print()
ridge_params = {'alpha':np.logspace(3,-3,10)}
ridge_cv = GridSearchCV(Ridge(), ridge_params, cv = 10, scoring = 'r2')
ridge_cv.fit(x_pca, y)

print()
print("BEST PARAMETERS:-\n", ridge_cv.best_params_)
print("BEST SCORE:-\n", ridge_cv.best_score_)
print("BEST INDEX:-\n", ridge_cv.best_index_)

print()
model = ridge_cv.best_estimator_
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)


#EVALUATING MODEL
print()
print("EVALUATION")
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2_val = r2_score(y_test, y_pred)

print()
print("MEAN SQUARRED ERROR:-\n", mse)
print("MEAN ABSOLUTE ERROR:-\n", mae)
print("R2_SCORE:-\n", r2_val)


#COMPARISON OF ACTUAL AND PREDICTED
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores = cross_val_score(model, x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALIZATION

#Comparison: Actual vs Predicted (Scatter Plot)
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5, color='green')
plt.xlabel("Actual Satisfaction Rating")
plt.ylabel("Predicted Satisfaction Rating")
plt.title("Actual vs Predicted Satisfaction Ratings")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Perfect prediction line
plt.grid(True)
plt.tight_layout()
plt.show()


#Residual Plot (Prediction Errors)
residuals = y_test - y_pred

plt.figure(figsize=(8,6))
sns.histplot(residuals, kde=True, bins=30, color='purple')
plt.title("Distribution of Residuals (Errors)")
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()


#Feature Importance using PCA Components
# Create DataFrame for PCA loadings
pca_components = pd.DataFrame(
    pca.components_,
    columns=x.columns,
    index=[f'PC{i+1}' for i in range(pca.n_components_)]
)

plt.figure(figsize=(12,6))
sns.heatmap(pca_components.T, cmap='coolwarm', annot=True)
plt.title("PCA Component Loadings")
plt.xlabel("Principal Components")
plt.ylabel("Original Features")
plt.tight_layout()
plt.show()


#Cross-Validation Scores (Bar Chart)
plt.figure(figsize=(8,5))
plt.bar(range(1, 11), scores, color='skyblue')
plt.xlabel("Fold Number")
plt.ylabel("R² Score")
plt.title("Cross-Validation Scores Across Folds")
plt.grid(True)
plt.tight_layout()
plt.show()


