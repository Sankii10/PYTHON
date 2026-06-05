# THE FOLLOWING CODE IS A PRACTICE CODE BY MR.SANKET GAIKWAD
# EXPLORATARY DATA ANALYSIS
# SOURCE :- https://www.geeksforgeeks.org/data-analysis/exploratory-data-analysis-in-python/ 



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')


#LOADING OF DATA 
a = pd.read_csv("C:\\Users\\ASUS\\Desktop\\My Data\\WineQt.csv")
print(a.columns.tolist())

#INFORMATION ABOUT DATA
print(a.info())


#DESCRIBE THE DATA
print(a.describe())


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


#FINDING DUPLICATES
print()
print(a.nunique())


# Univariate Analysis
# Analysis involving only one variable (feature/column) at a time.
print()
quality_counts  =a['quality'].value_counts()
plt.figure(figsize=(8,6))
plt.title("Count Plot of Quality")
plt.bar(quality_counts.index, quality_counts, color='Orange')
plt.xlabel("QUANTITY")
plt.ylabel("COUNTS")
plt.show()


# Kernel density plot for understanding variance in the dataset
sns.set_style("darkgrid")

numcol = a.select_dtypes(include={"int64","float64"}).columns

plt.figure(figsize=(14,len(numcol)* 3))
for idx,feature in enumerate(numcol,1):
    plt.subplot(len(numcol),2,idx)
    sns.histplot(a[feature], kde = 'True')
    plt.title(f"{feature} ! Skewness : {round(a[feature].skew(), 2)}")

plt.tight_layout()
plt.show()


# Swarm Plot for showing the outlier in the data
plt.figure(figsize = (10,8))

sns.swarmplot(x="quality", y="alcohol", data=a, palette='viridis')

plt.title("SWARM PLOT FOR WINE DATASET")
plt.xlabel("QUALITY")
plt.ylabel("ALCOHOL")
plt.show()



#  Bivariate Analysis
# Analysis involving two variables to understand the relationship between them.
sns.set_palette('Pastel1')

plt.figure(figsize=(10,6))

sns.pairplot(a)

plt.suptitle('PAIR PLOT FOR DATAFRAME')
plt.show()


# Box Plot for examining the relationship between alcohol and Quality
sns.boxplot(x="quality", y="alcohol", data = a)



# Multivariate Analysis
# Analysis involving three or more variables simultaneously.
plt.figure(figsize=(15,20))

sns.heatmap(a.corr(), annot=True,fmt='.2f', cmap = 'Pastel2', linewidths=2)

plt.title("CORRELATION HEATMAP")
plt.show()












