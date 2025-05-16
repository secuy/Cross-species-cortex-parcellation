import numpy as np
from sklearn.decomposition import PCA

def matPCA(mat, n_components):

    # 初始化 PCA，指定降维后的维度，比如降到2维
    pca = PCA(n_components=n_components)

    # 进行 PCA 并返回降维后的数据
    reduced_data = pca.fit_transform(mat)
    return reduced_data

if __name__ == '__main__':
    mat = np.random.rand(20, 10)
    reduced_data = matPCA(mat, 5)
    print(reduced_data.shape)