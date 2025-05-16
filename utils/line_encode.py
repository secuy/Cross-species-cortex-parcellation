

import numpy as np
import time
from joblib import Parallel, delayed


def line_to_beta(index, line, k=3):

    direction = line[-1] - line[0]

    tract = line.T
    n_vertex = tract.shape[1]

    p0 = tract[:, :-1]
    p1 = tract[:, 1:]
    disp = p1 - p0

    L2 = np.sqrt(np.sum(disp ** 2, axis=0))

    arc_length = np.sum(L2)

    cum_len = np.cumsum(L2) / arc_length
    para = np.zeros(n_vertex)
    para[1:] = cum_len

    n_vertex = len(para)
    para_even = np.hstack((-para[::-1][1:], para))

    tract_even = np.hstack((tract[:, ::-1][:, 1:], tract))

    para_even = np.tile(para_even, (k + 1, 1)).T
    pi_factors = np.tile(np.arange(k + 1), (2 * n_vertex - 1, 1)) * np.pi
    Y = np.cos(para_even * pi_factors) * np.sqrt(2)

    beta = np.linalg.pinv(Y.T @ Y) @ Y.T @ tract_even.T
    return index, beta, direction


def lines_to_psd_multi_cpu(lines, number_of_jobs=1):
    result_betas = np.zeros((len(lines), 9))
    betas = np.zeros((len(lines), 4, 3))
    # time_start = time.time()
    results = Parallel(n_jobs=number_of_jobs, verbose=0)(
        delayed(line_to_beta)(
            index,
            line)
        for index, line in zip(range(0, len(lines)), lines))
    # print(time.time() - time_start)
    for result in results:
        index, beta, direction = result
        betas[index] = beta
        result_betas[index, 6:9] = direction
    # print(time.time() - time_start)
    vecs = np.sqrt(np.sum(betas[:, 1:, :] ** 2, axis=-1))
    result_betas[:, 3:6] = vecs
    result_betas[:, 0:3] = betas[:, 0, :]
    # print(time.time() - time_start)
    return result_betas


def lines_to_betas_multi_cpu(lines, number_of_jobs=1, k=3):
    result_betas = np.zeros((len(lines), k+1, 3))
    results = Parallel(n_jobs=number_of_jobs, verbose=0)(
        delayed(line_to_beta)(
            index,
            line, k)
        for index, line in zip(range(0, len(lines)), lines))
    # print(time.time() - time_start)
    for result in results:
        index, beta, direction = result
        result_betas[index] = beta
    return result_betas


def line_2_psd(line):
    _, beta, direction = line_to_beta(0, line)
    result = np.zeros((2, 9))
    result[0, 0:3] = beta[0]
    result[0, 3:6] = np.sqrt(np.sum(beta[1:] ** 2, axis=1))
    result[0, 6:9] = direction
    result[1] = result[0]
    result[1, 6:9] *= -1
    return result

