import os
import fnmatch
import nibabel as nib
import numpy as np
import json
from sklearn.neighbors import KDTree
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

from utils.readTck_Trk_nib import get_tck_trk_streamlines

surf_name = "inflated"
surf_L_file = "*/S1200.L.{}_MSMAll.10k_fs_LR.surf.gii".format(surf_name)
surf_R_file = "*/S1200.R.{}_MSMAll.10k_fs_LR.surf.gii".format(surf_name)
base_path = "**"
out_path = "**"

human_parent_dirs = np.load("*")
with open("*", 'r') as json_file:
    human_child_dict = json.load(json_file)

CLUSTER_NUM = 33256

surf_L = nib.load(surf_L_file).darrays[0].data
surf_R = nib.load(surf_R_file).darrays[0].data
surf = np.concatenate((surf_L, surf_R), axis=0)

kdtree = KDTree(surf, leaf_size=1000)

def find_folders(base_path):
    paths = []
    sub_names = os.listdir(base_path)
    for sub_name in sub_names:
        paths.append(base_path + sub_name)
    return paths

def process_path(path):
    # 创建文件夹
    sub_name = path.split("/")[-1]
    if not os.path.exists(f"{out_path}{sub_name}"):
        os.makedirs(f"{out_path}{sub_name}", exist_ok=True)
    else:
        print(f"{path} skip")
        return

    cluster_num = 0

    fiber_vertices_mat = np.zeros((CLUSTER_NUM, surf.shape[0]))

    for cluster_folder in human_parent_dirs:
        cluster_f = os.path.join(path, cluster_folder, "Cluster_clean_in_yeo_space")  # , sub_folder[sub_idx]
        clusters = human_child_dict[cluster_folder]
        for cluster in clusters:
            print("cluster num:{}".format(cluster_num))
            if not os.path.exists(cluster_f + "/" + cluster):
                print(f"{sub_name} folder is broken")
                return
            streamlines = get_tck_trk_streamlines(cluster_f + "/" + cluster)
            if len(streamlines) == 0:
                cluster_num += 1
                continue
            streamlines = np.array(list(streamlines))

            for idx, line in enumerate(streamlines):
                dist1, ind1 = kdtree.query(line[1].reshape(1, -1), k=1)
                ind1 = ind1[0, 0]

                dist2, ind2 = kdtree.query(line[-2].reshape(1, -1), k=1)
                ind2 = ind2[0, 0]

                fiber_vertices_mat[cluster_num][ind1] += 1
                fiber_vertices_mat[cluster_num][ind2] += 1

            cluster_num += 1
    print(cluster_num)

    np.save(f"{out_path}{sub_name}/fiber_vertices_mat_{surf_name}.npy", fiber_vertices_mat)


def main():
    paths = find_folders(base_path)
    pool_size = 50
    with Pool(pool_size) as pool:
        pool.map(process_path, paths)


if __name__=='__main__':
    main()