"""
THE BELOW CODE IS EXECUTED BY...
NAME :- SANKET GAIKWAD
DATASET :- MALL CUSTOMER DATASET (SOURCE : KAGGLE)
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, silhouette_score

a = pd.read_csv("C:\\Users\\anujg\\OneDrive\\Desktop\\MY DATA\\DATASET\\Mall_Customers.csv")
print(a.columns.tolist())

print()
a = a.drop('CustomerID',axis = 1)

print()
print(a.info())

print()
b = a.isnull().sum()
print(b)

print()
missing_value = a[a.isna().any(axis = 1)]

if not missing_value.empty:
    print(missing_value)
else:
    print("NO SUCH MISSING VALUE FOUND IN DATASET !!!")

print()
le = LabelEncoder()
a['Genre'] = le.fit_transform(a['Genre'])

print()
x = a.drop('Spending Score (1-100)', axis = 1)
y = a['Spending Score (1-100)']

print()
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

print()
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)


# Elbow Method (WCSS Curve)
print()
wcss = []
for i in range(1,11):
    trial = KMeans(n_clusters= i, init='k-means++')
    trial.fit(x_pca)
    wcss.append(trial.inertia_)

print()
plt.figure(figsize=(10,20))
plt.plot(range(1,11), wcss, marker = 'o')
plt.title("Elbow Method - Optimal Number of Clusters")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# Silhouette Scores for different k
print("\nSilhouette Scores for k = 2 to 9:")
for k in range(2, 10):
    model = KMeans(n_clusters=k, init='k-means++', random_state=42)
    labels = model.fit_predict(x_pca)
    score = silhouette_score(x_pca, labels)
    print(f'k = {k} → Silhouette Score = {score:.4f}')


#BUILDING ALGORITHM
print()
model = KMeans(n_clusters=5, init='k-means++')
model.fit_predict(x_pca) 


#VISUALIZE THE CLUSTERS 
# Predict cluster labels
labels = model.predict(x_pca)

# Convert PCA components to a DataFrame
pca_df = pd.DataFrame(x_pca, columns=['PC1', 'PC2'])

# Add cluster labels
pca_df['Cluster'] = labels

# Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='Set1', s=100)
plt.title("Customer Segments by KMeans Clustering")
plt.grid(True)
plt.show()


#Silhouette Score – Measures clustering quality (range: -1 to 1)
score = silhouette_score(x_pca, labels)
print(f"Silhouette Score: {score:.4f}")


# Interpret the Clusters
cluster_centers = model.cluster_centers_
original_features_centroids = pca.inverse_transform(cluster_centers)
centroid_df = pd.DataFrame(sc.inverse_transform(original_features_centroids), columns=x.columns)

print("\nCentroids of each cluster:")
print(centroid_df)


