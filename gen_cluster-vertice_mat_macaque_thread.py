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
surf_L_file = "*/MacaqueYerkes19.L.{}.10k_fs_LR.surf.gii".format(surf_name)
surf_R_file = "*/MacaqueYerkes19.R.{}.10k_fs_LR.surf.gii".format(surf_name)
base_path = "*"
out_path = "*"

maca_parent_dirs = np.load("*")
with open("*", 'r') as json_file:
    maca_child_dict = json.load(json_file)

folder_name = 'FC'
sub_folder = ['cluster_clean_2_in_Yerkes', 'cluster_clean_2_in_F99']
sub_idx = 1

CLUSTER_NUM = 12063

surf_L = nib.load(surf_L_file).darrays[0].data
surf_R = nib.load(surf_R_file).darrays[0].data
surf = np.concatenate((surf_L, surf_R), axis=0)

kdtree = KDTree(surf, leaf_size=1000)

def find_folders(base_path, folder_name):
    paths = []
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if fnmatch.fnmatch(dir_name, folder_name):
                paths.append(os.path.join(root, dir_name))
    return paths

def process_path(path):
    # 创建文件夹
    words_list = path.split("/")
    sub_name = f"{words_list[4]}_{words_list[5]}_{words_list[6]}"
    if not os.path.exists(f"{out_path}{sub_name}"):
        os.makedirs(f"{out_path}{sub_name}", exist_ok=True)

    cluster_num = 0

    fiber_vertices_mat = np.zeros((CLUSTER_NUM, surf.shape[0]))

    for cluster_folder in maca_parent_dirs:
        cluster_f = os.path.join(path, cluster_folder, sub_folder[sub_idx])
        clusters = maca_child_dict[cluster_folder]
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
    paths = find_folders(base_path, folder_name)
    pool_size = 53
    with Pool(pool_size) as pool:
        pool.map(process_path, paths)

if __name__=='__main__':
    main()