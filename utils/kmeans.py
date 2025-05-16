import random
import math
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def kmeans(data, numOfClusters=50, max_iter=5000):
    cluster = MiniBatchKMeans(n_clusters=numOfClusters, init='k-means++',
                              max_iter=max_iter, compute_labels=True,
                              init_size=None, n_init='auto', batch_size=100000, verbose=True)
    labels = cluster.fit_predict(data)
    return labels


if __name__ == '__main__':
    # 示例数据
    data = np.array([
        [1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
        [8.0, 8.0], [1.0, 0.6], [9.0, 11.0],
        [8.0, 2.0], [10.0, 2.0], [9.0, 3.0]
    ])
    print(data.shape)
    k = 3
    # 实例化并拟合模型
    labels = kmeans(data, k)

    print("簇标签:", type(labels))

