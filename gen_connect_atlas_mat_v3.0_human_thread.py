import json
import os
import fnmatch
import time
from functools import partial

import nibabel as nib
import numpy as np
from scipy.spatial import KDTree
from multiprocessing import Pool

from utils.readTck_Trk_nib import get_tck_trk_streamlines
from utils.voxel_to_world import voxel_to_world

atlas_path = "*/XTRACT_atlases-master/HCP_tracts_5/"
atlas_name = "xtract-HCP-tracts-5"
species = "human"
atlas_list = ['ac.nii.gz', 'af_l.nii.gz', 'af_r.nii.gz', 'ar_l.nii.gz', 'ar_r.nii.gz',
              'atr_l.nii.gz', 'atr_r.nii.gz', 'cbd_l.nii.gz', 'cbd_r.nii.gz', 'cbp_l.nii.gz',
              'cbp_r.nii.gz', 'cbt_l.nii.gz', 'cbt_r.nii.gz', 'cst_l.nii.gz', 'cst_r.nii.gz',
              'fa_l.nii.gz', 'fa_r.nii.gz', 'fma.nii.gz', 'fmi.nii.gz', 'fx_l.nii.gz', 'fx_r.nii.gz',
              'ifo_l.nii.gz', 'ifo_r.nii.gz', 'ilf_l.nii.gz', 'ilf_r.nii.gz', 'mcp.nii.gz',
              'mdlf_l.nii.gz', 'mdlf_r.nii.gz', 'or_l.nii.gz', 'or_r.nii.gz', 'slf1_l.nii.gz', 'slf1_r.nii.gz',
              'slf2_l.nii.gz', 'slf2_r.nii.gz', 'slf3_l.nii.gz', 'slf3_r.nii.gz', 'str_l.nii.gz', 'str_r.nii.gz',
              'uf_l.nii.gz', 'uf_r.nii.gz', 'vof_l.nii.gz', 'vof_r.nii.gz']

human_parent_dirs = np.load("*")
with open("*", 'r') as json_file:
    human_child_dict = json.load(json_file)

base_path = "*"
out_path = "*"
sub_folder = ['Cluster_clean_in_yeo_space']
sub_idx = 0

LEAF_SIZE = 1000
CLUSTER_NUM = 33256
RADIUS = 2

atlas_threshold = 0.5

def openAndTransAtlas(path):
    world_coords = []
    atlas_label = []
    atlas_voxel_num = []
    for idx, file in enumerate(atlas_list):

        print(file)
        datas = nib.load(path + file)
        data = datas.get_fdata()
        affine = datas.affine

        space_points = np.stack((np.where(data > atlas_threshold)[0],
                                 np.where(data > atlas_threshold)[1], np.where(data > atlas_threshold)[2]), axis=-1)

        print(space_points.shape)
        atlas_voxel_num.append(space_points.shape[0] ** (1 / 3))

        world_coord = voxel_to_world(space_points, affine)
        world_coords.append(world_coord)
        atlas_label.append(np.array([idx] * world_coord.shape[0], dtype=np.int32))

    world_coords = np.concatenate(world_coords, axis=0)
    kdtree = KDTree(world_coords, leafsize=LEAF_SIZE)
    atlas_label = np.concatenate(atlas_label)
    return kdtree, atlas_label, atlas_voxel_num


def find_folders(base_path):
    paths = []
    sub_names = os.listdir(base_path)
    for sub_name in sub_names:
        paths.append(base_path + sub_name)
    return paths

def process_path(path, kdtree, atlas_label, atlas_voxel_num):
    # 创建文件夹
    sub_name = path.split("/")[-1]
    if not os.path.exists(f"{out_path}{sub_name}"):
        os.makedirs(f"{out_path}{sub_name}", exist_ok=True)
    # else:
    #     print(f"{path} skip")
    #     return

    start = time.time()

    cluster_num = 0

    fiber_atlas_mat = np.zeros((CLUSTER_NUM, len(atlas_voxel_num)))

    for cluster_folder in human_parent_dirs:
        cluster_f = os.path.join(path, cluster_folder, sub_folder[sub_idx])  # , sub_folder[sub_idx]
        clusters = human_child_dict[cluster_folder]
        for cluster in clusters:
            if not os.path.exists(cluster_f + "/" + cluster):
                print(f"{sub_name} folder is broken")
                return
            print("{}".format(cluster_num))
            streamlines = get_tck_trk_streamlines(cluster_f + "/" + cluster)
            if len(streamlines) != 0:
                for line in streamlines:
                    idxs = kdtree.query_ball_point(line, RADIUS)
                    label_point_count = np.zeros(len(atlas_voxel_num))
                    if any(idxs):
                        for idz in idxs:
                            label_point_count[atlas_label[idz]] += 1
                    else:
                        continue
                    label_point_count /= atlas_voxel_num
                    fiber_atlas_mat[cluster_num] += label_point_count
                # fiber_atlas_mat[cluster_num] = fiber_atlas_mat[cluster_num] / len(streamlines)
            cluster_num += 1

    print(cluster_num)
    print(f"{path}, time is:{time.time() - start}")

    np.save(f"{out_path}{sub_name}/fiber_atlas_mat_{atlas_threshold}.npy", fiber_atlas_mat)


def main():
    kdtree, atlas_label, atlas_voxel_num = openAndTransAtlas(atlas_path)

    paths = find_folders(base_path)
    pool_size = 30

    process_path_with_params = partial(process_path, kdtree=kdtree, atlas_label=atlas_label, atlas_voxel_num=atlas_voxel_num)

    with Pool(pool_size) as pool:
        pool.map(process_path_with_params, paths)

if __name__=='__main__':
    main()
