#THE FOLLOWING CODE S EXECUTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- EXPLORATORY DATA ANALYSIS(EDA) & KNN REGRESSOR
#SOURCE :- Medical Cost Personal Datasets ( https://www.kaggle.com/datasets/mirichoi0218/insurance )



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



#IMPORTING THE DATA
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\insurance.csv")
print(a.columns.tolist())


#INFORMATION ABOUT DATA
print()
print(a.info())


#FINDING MISSING VALUES
print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#EXPLORATORY DATA ANALYSIS
#Univariate Analysis

charges_count = a['charges'].value_counts()

plt.figure(figsize = (8,6))
plt.bar(charges_count.index, charges_count, color = 'deeppink')
plt.title("COUNT PLOT OF CHARGES")
plt.xlabel('CHARGES')
plt.ylabel("COUNT")
plt.show()


#KERNEL DENSITY PLOT
sns.set_style('darkgrid')
numerical_columns = a.select_dtypes(include={'int64','float64'}).columns
plt.figure(figsize = (10, len(numerical_columns)*3))
for idx, feature in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns),2,idx)
    sns.histplot(a[feature], kde = True)
    plt.title(f"{feature} | Skewness :{round(a[feature].skew(), 2)}")
plt.tight_layout()
plt.show()


#SWARM PLOT
plt.figure(figsize = (10,12))

sns.swarmplot(x = 'smoker', y = 'charges', data = a, palette='viridis')
plt.xlabel('SMOKER')
plt.ylabel('CHARGES')
plt.show()


#BIVARIATE
sns.set_palette("Pastel1")
plt.figure(figsize=(10,12))
sns.pairplot(a)
plt.suptitle("PAIR PLOT OF DATASET ")
plt.show()


#LABEL ENCODING
print()
le = LabelEncoder()
for col in ['sex','smoker', 'region']:
    a[col] = le.fit_transform(a[col])

#Multivariate Analysis
plt.figure(figsize=(15, 10))
sns.heatmap(a.corr(), annot=True, fmt='.2f', cmap='Pastel2', linewidths=2)
plt.title('Correlation Heatmap')
plt.show()


#FEATURE SELECTION
print()
x = a.drop('charges', axis = 1)
y = a['charges']


#STANDARD SCALING
print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)


#PRINCPAL COMPONENT ANALYSIS
print()
pca = PCA(n_components= 0.95)
x_pca = pca.fit_transform(x_scaled)


#SPLITING OF DATA INTO TRAIN AND TEST
print()
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.2, random_state= 1)


#MODEL BUILDING
print()
model = KNeighborsRegressor(n_neighbors=5)
model.fit(x_train, y_train)

print()
y_pred = model.predict(x_test)
print(y_pred)


#EVALUATION
print()
print("EVALUATION")
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2_val = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2_score(y_test, y_pred)) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1)


print()
print("MEAN SQUARED ERROR:-\n", mse)
print("MEAN ABSOLUTE ERROR:-\n", mae)
print("R2_SCORE:-\n", r2_val)
print("ADJUSTED R2 :-\n", adj_r2)


#COMPARISON
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
scores  =cross_val_score(model,x_pca, y, cv = 10)
print(scores)
print("THE MEAN SCORES:-\n", np.mean(scores))


#DATA VISUALIZATION

# Actual vs Predicted Scatter Plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Charges")
plt.show()


# Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True, color='purple')
plt.xlabel("Residuals")
plt.title("Distribution of Residuals")
plt.show()


# Feature Importance (using PCA components)
plt.figure(figsize=(6,4))
plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, color='orange')
plt.xlabel("PCA Component")
plt.ylabel("Variance Explained")
plt.title("Variance Explained by PCA Components")
plt.show()


# Cross-Validation Scores Bar Plot
plt.figure(figsize=(6,4))
plt.bar(range(1, len(scores)+1), scores, color='green')
plt.xlabel("Fold")
plt.ylabel("R² Score")
plt.title("Cross-Validation Scores per Fold")
plt.show()










