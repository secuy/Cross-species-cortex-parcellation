import numpy as np

# 设置一个很小的正数来避免除零错误
epsilon = 1e-10

# 按总和归一化
def sum_normalize(matrix):
    row_sums = matrix.sum(axis=1)
    # 避免除零错误
    row_sums[row_sums == 0] = epsilon
    return matrix / row_sums[:, np.newaxis]

# 按向量长度归一化
def length_normalize(matrix):
    row_lengths = np.linalg.norm(matrix, axis=1)
    # 避免除零错误
    row_lengths[row_lengths == 0] = epsilon
    return matrix / row_lengths[:, np.newaxis]

# 最小-最大归一化
def maxmin_normalize(matrix):
    # 获取每一行的最大值和最小值
    row_max = np.max(matrix, axis=1)
    row_min = np.min(matrix, axis=1)

    # 处理全0特征行：最大值和最小值都为0时，避免除以0错误
    zero_rows = (row_max == 0) & (row_min == 0)

    # 计算每一行的范围，避免除以0错误
    row_range = row_max - row_min
    row_range[zero_rows] = 1  # 将全0特征行的范围设为1，避免除以0

    # 执行最大最小值归一化
    normalized_matrix = (matrix - row_min[:, np.newaxis]) / row_range[:, np.newaxis]

    # 将全0特征行恢复为全0
    normalized_matrix[zero_rows] = 0

    return normalized_matrix


if __name__ == '__main__':
    # 示例矩阵
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [0, 0, 0]  # 添加一行全零的情况进行测试
    ])

    # 归一化结果
    sum_normalized_matrix = sum_normalize(matrix)
    length_normalized_matrix = length_normalize(matrix)
    maxmin_normalize_matrix = maxmin_normalize(matrix)

    print("按总和归一化后的矩阵:")
    print(sum_normalized_matrix)

    print("\n按向量长度归一化后的矩阵:")
    print(length_normalized_matrix)

    print("\n按最大最小归一化后的矩阵:")
    print(maxmin_normalize_matrix)
