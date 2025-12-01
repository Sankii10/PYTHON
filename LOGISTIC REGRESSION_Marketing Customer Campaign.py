#THE CODE IS IMPLEMENTED BY :- MR.SANKET GAIKWAD
#ALGORITHM USED :- LOGISTIC REGRESSION
#SOURCE :- https://www.opendatabay.com/data/ai-ml/18348d1d-7b4f-4e6b-8da7-126b86475b13


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE


# Load data
a = pd.read_csv("C:\\Users\\ASUS\\Downloads\\MAProjectData.csv")
print(a.columns.tolist())

# Convert date and create tenure
a['Dt_Customer'] = pd.to_datetime(a['Dt_Customer'], utc = True)
max_date = a['Dt_Customer'].max()
a['Customer_Tenure'] = (max_date - a['Dt_Customer']).dt.days


# Total Spend
print()
a['Total_Spend'] = (a['MntWines'] + a['MntFruits'] + a['MntMeatProducts'] + a['MntFishProducts'] + a['MntSweetProducts'] + a['MntGoldProds'])


# Convert Total_Spend into Spend_Class
a['Spend_Class'] = pd.qcut(a['Total_Spend'], q=3, labels=['Low','Medium','High'])


#Dropping the unrequired features
print()
a = a.drop(['Unnamed: 0', 'ID', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue','AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'Complain','MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds','Total_Spend'], axis = 1)


#Information and Statistical description of data 
print()
print(a.info())

print()
print(a.describe())


#Searching for duplicated values in dataset
print()
print("TOTAL NUMBER OF DUPLICATED VALUES :-\n")
print(a.duplicated().sum())


#display of duplicated values
print()
c = a[a.duplicated()]
print(c)

#dropping of duplicated values
print()
a = a.drop_duplicates()


#Searching for missing values in data
print()
b = a.isnull().sum()
print(b)

missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")


#Replacing the missing value with median
print()
a['Income'] = a['Income'].fillna(a['Income'].median())


#segregating the categorical and continous column
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


#One hot encoding
print()
a = pd.get_dummies(a,columns = ['Education', 'Marital_Status'], drop_first=True)


#Target variable feature selection
print()
x = a.drop('Spend_Class', axis = 1)
y = a['Spend_Class']


#Standard Scaling
print()
sc = StandardScaler()
x_scaled  =sc.fit_transform(x)


#Principal COmponent Analysis
print()
pca = PCA(n_components = 0.95)
x_pca =pca.fit_transform(x_scaled)


#Split of train and test dataset 
print()
x_train, x_test, y_train, y_test =train_test_split(x_pca, y, test_size=0.2, random_state= 1)

print()
smote = SMOTE(random_state= 1)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)


#Model Building and Implementation
print()
model = LogisticRegression(max_iter=500)
model.fit(x_train_res, y_train_res)

y_pred_prob = model.predict_proba(x_test)
y_pred_prob = pd.DataFrame(y_pred_prob, columns = ['Low', 'Medium', 'High'])
pd.options.display.float_format = '{:.3f}'.format
print(y_pred_prob)


print()
y_pred = model.predict(x_test)
print(y_pred)


#Evaluation
print()
print("EVALUATION")
print("ACCURACY SCORE :-\n", accuracy_score(y_test, y_pred))
print("CONFUSION MATRIX :-\n", confusion_matrix(y_test, y_pred))
print("CLASSIFICATION REPORT :-\n", classification_report(y_test, y_pred))


#Comparison
print()
print("COMPARISON")
comparison = pd.DataFrame({'Actual':y_test, 'Predict':y_pred})
print(comparison)

print()
print("CROSS SCORE VALIDATION")
kf = KFold(n_splits = 10, shuffle = True, random_state= 1)
scores = cross_val_score(model,x_pca, y, cv = kf, scoring = 'accuracy')
print(scores)
print("MEAN SCORES:-\n", np.mean(scores))

#Data Visualisation

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# 1) Spend Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=a, x="Spend_Class", order=["Low","Medium","High"])
plt.title("Spend Class Distribution")
plt.xlabel("Spend Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 2) Income Distribution
plt.figure(figsize=(6,4))
sns.histplot(a["Income"], kde=True, bins=30)
plt.title("Income Distribution")
plt.xlabel("Income")
plt.tight_layout()
plt.show()

# 3) Boxplot of Income by Spend Class
plt.figure(figsize=(6,4))
sns.boxplot(data=a, x="Spend_Class", y="Income", order=["Low","Medium","High"])
plt.title("Income vs Spend Class")
plt.tight_layout()
plt.show()

# 4) Customer Tenure Histogram
plt.figure(figsize=(6,4))
sns.histplot(a["Customer_Tenure"], kde=True, bins=30)
plt.title("Customer Tenure Distribution (Days)")
plt.xlabel("Tenure (days)")
plt.tight_layout()
plt.show()

# 5) Correlation Heatmap (numeric features)
numeric_df = a.select_dtypes(include=[np.number])
plt.figure(figsize=(12,10))
corr = numeric_df.corr()
sns.heatmap(corr, cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 6) PCA Explained Variance Plot
plt.figure(figsize=(7,4))
evr = pca.explained_variance_ratio_
plt.bar(range(1, len(evr)+1), evr, alpha=0.8)
plt.plot(range(1, len(evr)+1), np.cumsum(evr), marker="o", linestyle="--")
plt.title("PCA Explained Variance")
plt.xlabel("Principal Component")
plt.ylabel("Variance Ratio")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 7) PCA Scatter Plot (PC1 vs PC2)
plt.figure(figsize=(7,5))
# For full dataset
pca_df = pd.DataFrame(x_pca[:, :2], columns=["PC1","PC2"])
pca_df["Spend_Class"] = a["Spend_Class"].values

sns.scatterplot(
    data=pca_df,
    x="PC1", y="PC2",
    hue="Spend_Class",
    palette="Set1",
    s=40
)
plt.title("PCA Scatter Plot (PC1 vs PC2)")
plt.tight_layout()
plt.show()

# 8) Confusion Matrix Heatmap (Manual)
labels = ["Low","Medium","High"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 9) Classification Report Heatmap (Manual)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Keep only classes
class_report = report_df.loc[["Low","Medium","High"], ["precision","recall","f1-score"]]

plt.figure(figsize=(6,4))
sns.heatmap(class_report, annot=True, fmt=".2f", cmap="magma")
plt.title("Classification Report Heatmap")
plt.tight_layout()
plt.show()