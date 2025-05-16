import numpy as np

# 计算特征的余弦相似度矩阵(对称)
def calc_connect_sim(feat_A, feat_B):
    # feat_A: (M, N), M为特征维数, feat_B: (N, M)

    # 计算余弦相似度
    norms_A = np.linalg.norm(feat_A, axis=1, keepdims=True)
    norms_B = np.linalg.norm(feat_B, axis=0, keepdims=True)
    # 避免除以0：将所有为0的范数替换为一个小的非零数
    norms_A[norms_A == 0] = 1e-10
    norms_B[norms_B == 0] = 1e-10

    feat_norm_A = feat_A / norms_A
    feat_norm_B = feat_B / norms_B
    cosine_similarity_matrix = np.dot(feat_norm_A, feat_norm_B)

    similarity_matrix = (cosine_similarity_matrix + cosine_similarity_matrix.T) / 2  # 确保对称
    np.fill_diagonal(similarity_matrix, 1)  # 对角线元素设为1（自身相似性）
    # np.fill_diagonal(similarity_matrix, 0)
    return similarity_matrix
