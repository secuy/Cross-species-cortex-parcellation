import os
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
from sklearn import cluster
from sklearn.preprocessing import normalize

from utils.calc_connect_sim import calc_connect_sim
from cluster_metrics import cluster_metric_4_single_sub

superLabels_L_human = loadmat("")['superLabels'].flatten()
superLabels_R_human = loadmat("")['superLabels'].flatten() + 1000

superLabels_human = np.concatenate([superLabels_L_human, superLabels_R_human])
superLabels_maca = np.concatenate([superLabels_L_maca, superLabels_R_maca])
print(np.unique(superLabels_human), np.unique(superLabels_maca))

# 这里要注意的是：sub_label的开始的标签为0,super_label开始的标签为1,最后生成的final_label应该与sub_label一致,其开始标签为0
def gen_final_label(super_label, sub_label):
    final_label = np.zeros(20484)
    uni_sub_label = np.unique(sub_label)
    for i in uni_sub_label:
        # 注意这里找到的superidx的位置应该要加1，因为superlabel的标签从1开始
        super_labels_idx = np.where(sub_label == i)[0] + 1
        for j in super_labels_idx:
            final_label[np.where(super_label == j)[0]] = i
    # print(np.where(final_label==0)[0].shape)
    return final_label

def merge_super_vertex_feat(vert_feat, super_labels):
    super_labels_length = np.unique(super_labels).shape[0]
    super_feat = []
    for i in range(1, super_labels_length):
        super_feat.append(np.mean(vert_feat[:, np.where(super_labels==i)[0]], axis=1))
    super_feat = np.array(super_feat)
    print(super_feat.shape)
    return super_feat

def cal_knn_graph(distance, neighbor_num, metric):
    # construct a knn graph
    neighbors_graph = kneighbors_graph(
        distance, neighbor_num, mode='connectivity', metric=metric, include_self=False)
    W = 0.5 * (neighbors_graph + neighbors_graph.T)
    return W

def spectral_cluster(feat, cluster_num):
    W = cal_knn_graph(feat, neighbor_num=25, metric="cosine")
    laplacian = sparse.csgraph.laplacian(W, normed=True)
    _, vec = sparse.linalg.eigsh(sparse.identity(laplacian.shape[0]) - laplacian, cluster_num, sigma=None, which='LA')
    embedding = normalize(vec)
    est = cluster.KMeans(n_clusters=cluster_num, n_init="auto").fit(embedding)
    return est.labels_

if __name__ == "__main__":
    human_num = 5
    maca_num = 5

    human_root_path = ""
    maca_root_path = ""

    human_folders = sorted(os.listdir(human_root_path))
    maca_folders = sorted(os.listdir(maca_root_path))
    print(len(human_folders))
    print(len(maca_folders))

    clusterNum_list = [4, 6, 8, 10, 12, 14, 16, 18, 20, 40, 60, 80, 100]

    for n_clusters in clusterNum_list:
        human_tot_DBI = 0
        maca_tot_DBI = 0
        human_tot_CHI = 0
        maca_tot_CHI = 0
        human_tot_homo = 0
        maca_tot_homo = 0
        for i in range(human_num):
            print(f"{human_folders[i]} is reading")
            human_sub_path = os.path.join(human_root_path, human_folders[i], "vertice_atlas_mat_norm.npy")
            human_data = np.load(human_sub_path)

            human_super_avg_feat = merge_super_vertex_feat(human_data, superLabels_human)
            labels_human = spectral_cluster(human_super_avg_feat, n_clusters)
            np.save(f"./new_baseline_super_labels/new_baseline_super_labels_{human_folders[i]}_{n_clusters}.npy", labels_human)
            
            final_labels_human = gen_final_label(superLabels_human, labels_human)
            human_DBI_score, human_CHI_score, human_homo = cluster_metric_4_single_sub(human_data, final_labels_human, "human")
            np.save(f"./new_baseline_final_labels/new_baseline_final_labels_{human_folders[i]}_{n_clusters}.npy", final_labels_human)
            human_tot_DBI += human_DBI_score
            human_tot_CHI += human_CHI_score
            human_tot_homo += human_homo

        for i in range(maca_num):
            print(f"{maca_folders[i]} is reading")
            maca_sub_path = os.path.join(maca_root_path, maca_folders[i], "vertice_atlas_mat_norm.npy")
            maca_data = np.load(maca_sub_path)

            maca_super_avg_feat = merge_super_vertex_feat(maca_data, superLabels_human)
            # maca_sim_mat = calc_connect_sim(maca_super_avg_feat, maca_super_avg_feat.T)
            # spectral_clustering = SpectralClustering(n_clusters=n_clusters,
            #                                          affinity='precomputed',
            #                                          assign_labels='kmeans',
            #                                          random_state=0)
            # labels_maca = spectral_clustering.fit_predict(maca_sim_mat)
            labels_maca = spectral_cluster(maca_super_avg_feat, n_clusters) 
            np.save(f"./new_baseline_super_labels/new_baseline_super_labels_{maca_folders[i]}_{n_clusters}.npy", labels_maca)

            final_labels_maca = gen_final_label(superLabels_human, labels_maca)
            np.save(f"./new_baseline_final_labels/new_baseline_final_labels_{maca_folders[i]}_{n_clusters}.npy", final_labels_maca)
            maca_DBI_score, maca_CHI_score, maca_homo = cluster_metric_4_single_sub(maca_data, final_labels_maca, "maca")
            maca_tot_DBI += maca_DBI_score
            maca_tot_CHI += maca_CHI_score
            maca_tot_homo += maca_homo
        print(f"cluster_num:{n_clusters}")
        print(f"Human\\Macaque Mean Overall DBI: {(human_tot_DBI / human_num):.4f}\\{(maca_tot_DBI / maca_num):.4f}")
        print(f"Human\\Macaque Mean Overall CHI: {(human_tot_CHI / human_num):.4f}\\{(maca_tot_CHI / maca_num):.4f}")
        print(f"Human\\Macaque Mean Overall homo: {(human_tot_homo / human_num):.4f}\\{(maca_tot_homo / maca_num):.4f}")

        