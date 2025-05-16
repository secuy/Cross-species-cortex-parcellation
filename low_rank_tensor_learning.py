import os
import numpy as np
from scipy.linalg import eigh as largest_eigh
import scipy.io as scio
from scipy import sparse
from sklearn import cluster
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.neighbors import kneighbors_graph


superLabels_human = np.load("")

def merge_super_vertex_feat(feat, super_labels):
    super_labels_uni = np.unique(super_labels)
    super_feat = []
    for i in super_labels_uni:
        if i != 0:
            super_feat.append(np.mean(feat[:, np.where(super_labels==i)[0]], axis=1))
    super_feat = np.array(super_feat)
    print(super_feat.shape)
    return super_feat

def prox_weight_tensor_nuclear_norm(Y, C):
    # calculate the weighted tensor nuclear norm
    # min_X ||X||_w* + 0.5||X - Y||_F^2
    n1, n2, n3 = np.shape(Y)
    X = np.zeros((n1, n2, n3), dtype=complex)
    # Y = np.fft.fft(Y, n3)
    Y = np.fft.fftn(Y)
    # Y = np.fft.fftn(Y, s=[n1, n2, n3])
    eps = 1e-6
    for i in range(n3):
        U, S, V = np.linalg.svd(Y[:, :, i], full_matrices=False)
        temp = np.power(S - eps, 2) - 4 * (C - eps * S)
        ind = np.where(temp > 0)
        ind = np.array(ind)
        r = np.max(ind.shape)
        if np.min(ind.shape) == 0:
            r = 0
        if r >= 1:
            temp2 = S[ind] - eps + np.sqrt(temp[ind])
            S = temp2.reshape(temp2.size, )
            X[:, :, i] = np.dot(np.dot(U[:, 0:r], np.diag(S)), V[:, 0:r].T)
    newX = np.fft.ifftn(X)
    # newX = np.fft.ifftn(X, s=[n1, n2, n3])
    # newX = np.fft.ifft(X, n3)

    return np.real(newX)


def cal_knn_graph(distance, neighbor_num, metric):
    # construct a knn graph
    neighbors_graph = kneighbors_graph(
        distance, neighbor_num, mode='connectivity', metric=metric, include_self=False)
    W = 0.5 * (neighbors_graph + neighbors_graph.T)
    return W


def low_rank(A, cluster_num, lambda_1, rho, iteration_num):
    # optimize the consensus graph learning problem
    # min_H, Z 0.5||A - H'H||_F^2 + 0.5||Z - hatHhatH'||_F^2 + ||Z||_w*
    # s.t. H'H = I_k
    sample_num, sample_num, view_num = np.shape(A)
    # initial variables
    H = np.zeros((sample_num, cluster_num, view_num))
    HH = np.zeros((sample_num, sample_num, view_num))
    hatH = np.zeros((sample_num, cluster_num, view_num))
    hatHH = np.zeros((sample_num, sample_num, view_num))
    Q = np.zeros((sample_num, sample_num, view_num))
    Z = np.zeros((sample_num, sample_num, view_num))
    obj = np.zeros((iteration_num, 1))
    # loop
    for iter in range(iteration_num):
        # update H
        temp = np.zeros((sample_num, sample_num, view_num))
        G = np.zeros((sample_num, sample_num, view_num))
        for view in range(view_num):
            temp[:, :, view] = np.dot(
                np.dot(Q[:, :, view], 0.5 * (Z[:, :, view] + Z[:, :, view].T) - 0.5 * hatHH[:, :, view])
                , Q[:, :, view]
            )
            G[:, :, view] = lambda_1 * A[:, :, view] + temp[:, :, view]
            _, H[:, :, view] = largest_eigh(
                G[:, :, view], subset_by_index=[sample_num - cluster_num, sample_num - 1]
            )
            HH[:, :, view] = np.dot(H[:, :, view], H[:, :, view].T)
            Q[:, :, view] = np.diag(1 / np.sqrt(np.diag(HH[:, :, view])))
            hatH[:, :, view] = np.dot(Q[:, :, view], H[:, :, view])
            hatHH[:, :, view] = np.dot(hatH[:, :, view], hatH[:, :, view].T)
        # update Z
        hatHH2 = hatHH.transpose((0, 2, 1))
        Z2 = prox_weight_tensor_nuclear_norm(hatHH2, rho)
        Z = Z2.transpose((0, 2, 1))
        # update obj
        f = np.zeros((view_num, 1))
        for view in range(view_num):
            f[view] = 0.5 * lambda_1 * np.linalg.norm(A[:, :, view] - HH[:, :, view], ord='fro') + \
                      0.5 * np.linalg.norm(Z[:, :, view] - hatHH[:, :, view], ord='fro')
        obj[iter] = np.sum(f)
        if iter > 0:
            val = abs(obj[iter] - obj[iter - 1]) / obj[iter - 1]
            print(f"{iter}:{val}")
            if val < 0.001:
                break

    view_labels = []
    Ws = []
    # construct knn graph
    for view in range(0, view_num, 2):
        # minkowski
        distance = hatHH[:, :, view] + hatHH[:, :, view + 1]
        W = cal_knn_graph(1 - distance, neighbor_num=15, metric="cosine")
        # W = cal_knn_graph(1 - H[:, :, view], 15)
        # perform spectral clustering
        laplacian = sparse.csgraph.laplacian(W, normed=True)
        # laplacian = sparse.csgraph.laplacian(H[:, :, view], normed=True)
        _, vec = sparse.linalg.eigsh(sparse.identity(
            laplacian.shape[0]) - laplacian, cluster_num, sigma=None, which='LA')
        embedding = normalize(vec)
        est = cluster.KMeans(n_clusters=cluster_num, n_init="auto").fit(embedding)
        view_labels.append(est.labels_)
        Ws.append(W)
        # reture results
    return Ws, np.array(view_labels)


if __name__ == '__main__':
    human_num = 5
    maca_num = 5
    for LR in ['L', 'R']:
        human_root_path = ""
        maca_root_path = ""
        human_VA_path = ""
        maca_VA_path = ""
        human_folders = sorted(os.listdir(human_root_path))
        maca_folders = sorted(os.listdir(maca_root_path))
        human_maca_feat = []
        for i in range(human_num):
            human_sub_path = os.path.join(human_root_path, human_folders[i], "fiber_vertices_mat_inflated.npy")
            human_va_sub_path = os.path.join(human_VA_path, human_folders[i], "vertice_atlas_mat_norm.npy")
            if LR == "L":
                human_feat = np.load(human_sub_path)[:, :10242]
                human_va_feat = np.load(human_va_sub_path).T[:, :10242]
                superLabels = superLabels_human[:10242]
            else:
                human_feat = np.load(human_sub_path)[:, 10242:]
                human_va_feat = np.load(human_va_sub_path).T[:, 10242:]
                superLabels = superLabels_human[10242:]
            human_super_avg_feat = merge_super_vertex_feat(human_feat, superLabels)
            human_va_super_avg_feat = merge_super_vertex_feat(human_va_feat, superLabels)
            human_maca_feat.append(human_super_avg_feat)
            human_maca_feat.append(human_va_super_avg_feat)
        
        for i in range(maca_num):
            maca_sub_path = os.path.join(maca_root_path, maca_folders[i], "fiber_vertices_mat_inflated.npy")
            maca_va_sub_path = os.path.join(maca_VA_path, maca_folders[i], "vertice_atlas_mat_norm.npy")
            if LR == "L":
                maca_feat = np.load(maca_sub_path)[:, :10242]
                maca_va_feat = np.load(maca_va_sub_path).T[:, :10242]
                superLabels = superLabels_human[:10242]
            else:
                maca_feat = np.load(maca_sub_path)[:, 10242:]
                maca_va_feat = np.load(maca_va_sub_path).T[:, 10242:]
                superLabels = superLabels_human[10242:]
            maca_super_avg_feat = merge_super_vertex_feat(maca_feat, superLabels)
            maca_va_super_avg_feat = merge_super_vertex_feat(maca_va_feat, superLabels)
            human_maca_feat.append(maca_super_avg_feat)
            human_maca_feat.append(maca_va_super_avg_feat)
        
        view_num = (human_num + maca_num) * 2
        sample_num = np.unique(superLabels).shape[0] - 1
        A = np.zeros((sample_num, sample_num, view_num))
        for view in range(view_num):
            knn_graph = cal_knn_graph(human_maca_feat[view], neighbor_num=15, metric='cosine')
            S = sparse.identity(knn_graph.shape[0]) - sparse.csgraph.laplacian(knn_graph, normed=True).toarray()
            A[:, :, view] = S
        print(A.shape)
    
        cluster_list = [60,70,80,90,100,150,200]
        for cluster_num in cluster_list:
            parameter_lambda = [1, 100, 500, 1000, 5000, 10000]
            parameter_rho = [1000]
            for i in range(len(parameter_lambda)):
                for j in range(len(parameter_rho)):
                    Ws, predict_labels = low_rank(A, cluster_num, parameter_lambda[i], parameter_rho[j], 100)
                    # 注意这里已经标签从0开始进行+1，变为从1开始
                    predict_labels = predict_labels + 1
                    print(f"cluster_num:{cluster_num}")
                    print(predict_labels.shape, np.unique(predict_labels))  # , np.unique(predict_labels)
                    np.save(f"./MICCAI_2025/super_labels/super_labels-{cluster_num}_lambda-{parameter_lambda[i]}_rho-{parameter_rho[j]}_{LR}.npy", predict_labels)