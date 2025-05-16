import numpy as np
from sklearn.decomposition import NMF

# mat是原始矩阵，n_components是分解后的向量长度
def matNMF(mat, n_components):
    # 初始化 NMF 模型
    model = NMF(n_components=n_components, init='random', random_state=0)

    # 拟合模型并转换数据
    W = model.fit_transform(mat)
    H = model.components_

    return W, H

if __name__ == '__main__':
    mat = np.random.rand(20, 10)
    W, H = matNMF(mat, 5)
    print(W, W.shape, H, H.shape)