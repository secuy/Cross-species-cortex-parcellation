import numpy as np

def WFS_tracts(tract, para, k):
    """
    计算余弦级数表示的 3D 曲线。

    Parameters:
        tract (numpy.ndarray): 形状为 (3, n_vertex) 的 3D 曲线坐标。
        para (numpy.ndarray): 弧长参数化。
        k (int): 余弦级数的次数。

    Returns:
        wfs (numpy.ndarray): 余弦级数表示的曲线。
        beta (numpy.ndarray): 级数系数。

    """
    n_vertex = len(para)

    # 将 para 进行相反数操作拼接起来，长度变为原来的两倍减一，变成偶函数
    para_even = np.concatenate((-para[-2::-1], para))
    # 将 tract 进行相同的操作，区别是不加负号，长度变为原来的两倍减一，变偶
    tract_even = np.hstack((tract[:, -2::-1], tract))

    Y = np.zeros((2 * n_vertex - 1, k + 1))
    # 将 para 重复 k+1 遍
    para_even = np.tile(para_even, (k + 1, 1)).T
    # 0到k乘pi复制2倍的 vertex 的数量-1
    pi_factors = np.tile(np.arange(k + 1), (2 * n_vertex - 1, 1)) * np.pi
    Y = np.cos(para_even * pi_factors) * np.sqrt(2)

    # 计算 beta
    YTY = np.dot(Y.T, Y)
    YTY_inv = np.linalg.pinv(YTY)
    YTY_inv_YT = np.dot(YTY_inv, Y.T)
    beta = np.dot(YTY_inv_YT, tract_even.T)

    return beta


def parameterize_arclength(tract):
    """
    计算弧长并执行单位长度参数化。

    Parameters:
        tract (numpy.ndarray): 形状为(3, n_vertex)或(2, n_vertex)的输入曲线数组。

    Returns:
        arc_length (float): 总弧长
        para (numpy.ndarray): 映射曲线到单位区间[0, 1]的参数数组
    """
    n_vertex = tract.shape[1]
    # n_vertex 必须大于等于2才能正常运行此函数。

    p0 = tract[:, :-1]
    p1 = tract[:, 1:]
    disp = p1 - p0

    # 计算每一段的欧氏距离
    L2 = np.sqrt(np.sum(disp ** 2, axis=0))

    arc_length = np.sum(L2)

    # 计算参数化值
    cum_len = np.cumsum(L2) / arc_length
    para = np.zeros(n_vertex)
    para[1:] = cum_len

    return arc_length, para

def trans_cosine(tract, k):
    """
    计算余弦级数表示的 3D 曲线。

    Parameters:
        tract (numpy.ndarray): 形状为 (3, n_vertex) 的 3D 曲线坐标。
        k (int): 余弦级数的次数。

    Returns:
        wfs (numpy.ndarray): 余弦级数表示的曲线。
        beta (numpy.ndarray): 级数系数， 大小是(k+1, 3)。

    """

    arc_length, para = parameterize_arclength(tract)
    beta = WFS_tracts(tract, para, k)
    return beta

def point2cosine(fibers, k):
    # 将纤维曲线转化为余弦级数系数矩阵
    # fibers是多个纤维的数据点,数据类型为numpy数组,大小为：[N_fibers, N_point, features]
    # 返回值大小[N_fibers, (k+1)*3, features]
    # k是系数参数
    arr_list = []
    for fiber in fibers:
        arr_list.append(trans_cosine(fiber.T, k))
    result = np.stack(arr_list, axis=0)
    return result