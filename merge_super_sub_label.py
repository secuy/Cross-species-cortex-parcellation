import os
import numpy as np
from scipy.io import loadmat
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cluster_metrics import cluster_metric_4_single_sub
from utils.mat_norm import maxmin_normalize

# 这里要注意的是：sub_label的开始的标签为0,super_label开始的标签为1,最后生成的final_label应该与sub_label一致,其开始标签为0
def gen_final_label(super_label, sub_label):
    final_label = np.zeros(super_label.shape[0])
    uni_sub_label = np.unique(sub_label)
    uni_super_label = np.unique(super_label)
    for i in uni_sub_label:
        # 注意这里找到的superidx的位置应该要加1，因为superlabel的标签从1开始
        super_labels_idx = np.where(sub_label == i)[0] + uni_super_label[1]
        for j in super_labels_idx:
            final_label[np.where(super_label == j)[0]] = i
    return final_label

superLabels = np.load("")

for LR in ['L', 'R']:
    if LR == "L":
        superLabels_LR = superLabels[:10242]
    else:
        superLabels_LR = superLabels[10242:]
    human_root_path = ""
    maca_root_path = ""
    human_VA_path = ""
    maca_VA_path = ""

    human_folders = sorted(os.listdir(human_root_path))
    maca_folders = sorted(os.listdir(maca_root_path))
    human_num = 5
    maca_num = 5
    human_data = []
    maca_data = []
    for i in range(human_num):
        human_va_sub_path = os.path.join(human_VA_path, human_folders[i], "vertice_atlas_mat_norm.npy")
        human_va_feat = np.load(human_va_sub_path)
        human_va_feat_norm = maxmin_normalize(human_va_feat)
        if LR == "L":
            human_data.append(human_va_feat_norm[:10242, :])
        else:
            human_data.append(human_va_feat_norm[10242:, :])

    for i in range(maca_num):
        maca_va_sub_path = os.path.join(maca_VA_path, maca_folders[i], "vertice_atlas_mat_norm.npy")
        maca_va_feat = np.load(maca_va_sub_path)
        maca_va_feat_norm = maxmin_normalize(maca_va_feat)
        if LR == "L":
            maca_data.append(maca_va_feat_norm[:10242, :])
        else:
            maca_data.append(maca_va_feat_norm[10242:, :])

    # parameter_lambda = [1, 10, 100, 500, 1000, 5000, 10000]
    # parameter_rho = [1, 1000, 5000]
    parameter_lambda = [1, 500, 5000] # [1, 100, 500, 1000, 5000, 10000]
    parameter_rho = [1000]
    cluster_list = [3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,150,200]
    for n_clusters in cluster_list:
        print(f"cluster_num:{n_clusters}")
        for lambda_para in parameter_lambda:
            for rho_para in parameter_rho:
                human_tot_DBI = 0
                maca_tot_DBI = 0
                human_tot_CHI = 0
                maca_tot_CHI = 0
                human_tot_homo = 0
                maca_tot_homo = 0
                for sub_num in range(1, 6):
                    human_sub_labels = np.load(f"./MICCAI_2025/super_labels_ablation/super_labels_ablation-va-{n_clusters}_lambda-{lambda_para}_rho-{rho_para}_{LR}.npy")[sub_num-1, :]

                    maca_sub_labels = np.load(f"./MICCAI_2025/super_labels_ablation/super_labels_ablation-va-{n_clusters}_lambda-{lambda_para}_rho-{rho_para}_{LR}.npy")[4 + sub_num, :]


                    human_final_label = gen_final_label(superLabels_LR, human_sub_labels)
                    human_final_label = human_final_label.astype(int)
                    np.save(f"./MICCAI_2025/final_labels_ablation/final_labels_ablation-va-{n_clusters}_lambda-{lambda_para}_rho-{rho_para}_human-{sub_num}_{LR}.npy", human_final_label)
                    maca_final_label = gen_final_label(superLabels_LR, maca_sub_labels)
                    maca_final_label = maca_final_label.astype(int)
                    np.save(f"./MICCAI_2025/final_labels_ablation/final_labels_ablation-va-{n_clusters}_lambda-{lambda_para}_rho-{rho_para}_maca-{sub_num}_{LR}.npy", maca_final_label)

                    _, human_DBI_score, human_CHI_score, human_homo = cluster_metric_4_single_sub(human_data[sub_num-1], human_final_label, "human")
                    human_tot_DBI += human_DBI_score
                    human_tot_CHI += human_CHI_score
                    human_tot_homo += human_homo

                    _, maca_DBI_score, maca_CHI_score, maca_homo = cluster_metric_4_single_sub(maca_data[sub_num-1], maca_final_label, "maca")
                    maca_tot_DBI += maca_DBI_score
                    maca_tot_CHI += maca_CHI_score
                    maca_tot_homo += maca_homo
                print(f"lambda_para:{lambda_para}, rho_para:{rho_para}")     
                print(f"Human\\Macaque Mean Overall DBI: {(human_tot_DBI / human_num):.4f}\\{(maca_tot_DBI / maca_num):.4f}")
                print(f"Human\\Macaque Mean Overall CHI: {(human_tot_CHI / human_num):.4f}\\{(maca_tot_CHI / maca_num):.4f}")
                print(f"Human\\Macaque Mean Overall homo: {(human_tot_homo / human_num):.4f}\\{(maca_tot_homo / maca_num):.4f}")
