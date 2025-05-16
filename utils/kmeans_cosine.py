import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans

class CosineKMeans:
    def __init__(self, n_clusters=50, max_iter=5000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cluster_centers_ = None
        self.labels_ = None

    def fit_predict(self, X):
        # Identify all-zero vectors
        all_zero_mask = np.all(X == 0, axis=1)

        # Filter out all-zero vectors
        X_non_zero = X[~all_zero_mask]

        # Initialize cluster centers
        n_samples, n_features = X_non_zero.shape
        initial_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X_non_zero[initial_indices]

        for _ in range(self.max_iter):
            # Compute distances between points and cluster centers
            distances = cosine_distances(X_non_zero, self.cluster_centers_)
            labels = np.argmin(distances, axis=1)

            # Compute new cluster centers
            new_centers = np.zeros((self.n_clusters, n_features))
            for i in range(self.n_clusters):
                cluster_points = X_non_zero[labels == i]
                if len(cluster_points) > 0:
                    new_centers[i] = cluster_points.mean(axis=0)

            # Check for convergence (i.e., if centers do not change)
            if np.allclose(self.cluster_centers_, new_centers):
                break

            self.cluster_centers_ = new_centers

        self.labels_ = -1 * np.ones(X.shape[0], dtype=int)
        self.labels_[~all_zero_mask] = labels

        return self.labels_

def kmeans(data, numOfClusters=50, max_iter=5000):
    cluster = CosineKMeans(n_clusters=numOfClusters, max_iter=max_iter)
    labels = cluster.fit_predict(data)
    return labels

if __name__ == '__main__':
    # 示例数据，包含全0向量
    data = np.array([
        [1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
        [8.0, 8.0], [1.0, 0.6], [9.0, 11.0],
        [8.0, 2.0], [10.0, 2.0], [9.0, 3.0],
        [0.0, 0.0]
    ])
    print(data.shape)
    k = 3
    # 实例化并拟合模型
    labels = kmeans(data, k)
    print(type(labels))
    print("簇标签:", labels)
