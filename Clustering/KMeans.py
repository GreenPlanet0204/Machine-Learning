# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# load the dataset
dataset = pd.read_csv('dataset/segmented_customers.csv')
x = dataset.iloc[:, [3, 4]].values

# find optimal number of clusters using the elbow method
wcss_list = []  # initialize the list for the values of WCSS
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss_list)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters(k)')
plt.ylabel('wcss_list')
plt.show()

# train the k-means model on a dataset
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_pred = kmeans.fit_predict(x)

# visualize the clusters
plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1],
            s=100, c='blue', label='Cluster 1')
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1],
            s=100, c='green', label='Cluster 2')
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1],
            s=100, c='red', label='Cluster 3')
plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1],
            s=100, c='cyan', label='Cluster 4')
plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1],
            s=100, c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], s=200, c='yellow', label='Centroids')
plt.title('Cluster of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
