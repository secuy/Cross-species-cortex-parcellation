import fnmatch
import json
import os
import time
import nibabel as nib
import numpy as np
from dipy.viz import window, actor
import colorsys
from scipy.spatial import KDTree
from multiprocessing import Pool
from functools import partial

from utils.readTck_Trk_nib import get_tck_trk_streamlines
from utils.voxel_to_world import voxel_to_world

atlas_path = "*/XTRACT_atlases-master/Macaque_tracts/"
species = "macaque"
atlas_name = "xtract-Macaque-tracts"
atlas_list = ['ac.nii.gz', 'af_l.nii.gz', 'af_r.nii.gz', 'ar_l.nii.gz', 'ar_r.nii.gz',
              'atr_l.nii.gz', 'atr_r.nii.gz', 'cbd_l.nii.gz', 'cbd_r.nii.gz', 'cbp_l.nii.gz',
              'cbp_r.nii.gz', 'cbt_l.nii.gz', 'cbt_r.nii.gz', 'cst_l.nii.gz', 'cst_r.nii.gz',
              'fa_l.nii.gz', 'fa_r.nii.gz', 'fma.nii.gz', 'fmi.nii.gz', 'fx_l.nii.gz', 'fx_r.nii.gz',
              'ifo_l.nii.gz', 'ifo_r.nii.gz', 'ilf_l.nii.gz', 'ilf_r.nii.gz', 'mcp.nii.gz',
              'mdlf_l.nii.gz', 'mdlf_r.nii.gz', 'or_l.nii.gz', 'or_r.nii.gz', 'slf1_l.nii.gz', 'slf1_r.nii.gz',
              'slf2_l.nii.gz', 'slf2_r.nii.gz', 'slf3_l.nii.gz', 'slf3_r.nii.gz', 'str_l.nii.gz', 'str_r.nii.gz',
              'uf_l.nii.gz', 'uf_r.nii.gz', 'vof_l.nii.gz', 'vof_r.nii.gz']

base_path = "*"
out_path = "*"

maca_parent_dirs = np.load("*")
with open("*", 'r') as json_file:
    maca_child_dict = json.load(json_file)

folder_name = 'FC'
sub_folder = ['cluster_clean_2_in_Yerkes', 'cluster_clean_2_in_F99']
sub_idx = 1
LEAF_SIZE = 1000
CLUSTER_NUM = 12063
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

        data_idx = np.where(data > atlas_threshold)
        space_points = np.stack((data_idx[0], data_idx[1], data_idx[2]), axis=-1)

        print(space_points.shape)
        atlas_voxel_num.append(space_points.shape[0] ** (1 / 3))

        world_coord = voxel_to_world(space_points, affine)
        world_coords.append(world_coord)
        atlas_label.append(np.array([idx] * world_coord.shape[0], dtype=np.int32))

    world_coords = np.concatenate(world_coords, axis=0)
    kdtree = KDTree(world_coords, leafsize=LEAF_SIZE)
    atlas_label = np.concatenate(atlas_label)
    return kdtree, atlas_label, atlas_voxel_num


def find_folders(base_path, folder_name):
    paths = []
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if fnmatch.fnmatch(dir_name, folder_name):
                paths.append(os.path.join(root, dir_name))
    return paths

def process_path(path, kdtree, atlas_label, atlas_voxel_num):
    words_list = path.split("/")
    sub_name = f"{words_list[4]}_{words_list[5]}_{words_list[6]}"
    if not os.path.exists(f"{out_path}{sub_name}"):
        os.makedirs(f"{out_path}{sub_name}", exist_ok=True)
    # else:
    #     print(f"{path} skip")
    #     return
    start = time.time()

    fiber_atlas_mat = np.zeros((CLUSTER_NUM, len(atlas_voxel_num)))

    cluster_num = 0
    for parent_cluster in maca_parent_dirs:
        child_clusters_path = os.path.join(path, parent_cluster, sub_folder[sub_idx])
        child_clusters = maca_child_dict[parent_cluster]
        for child_cluster in child_clusters:
            if not os.path.exists(os.path.join(child_clusters_path, child_cluster)):
                print(f"{sub_name} folder is broken")
                return
            streamlines = get_tck_trk_streamlines(os.path.join(child_clusters_path, child_cluster))

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
            print(cluster_num)
            cluster_num += 1

    print(cluster_num)
    print(fiber_atlas_mat.shape)
    print(f"{path}, time is:{time.time() - start}")

    np.save(f"{out_path}{sub_name}/fiber_atlas_mat_{atlas_threshold}.npy", fiber_atlas_mat)

def main():
    kdtree, atlas_label, atlas_voxel_num = openAndTransAtlas(atlas_path)

    paths = find_folders(base_path, folder_name)
    pool_size = 53

    process_path_with_params = partial(process_path, kdtree=kdtree, atlas_label=atlas_label, atlas_voxel_num=atlas_voxel_num)

    with Pool(pool_size) as pool:
        pool.map(process_path_with_params, paths)


if __name__ == '__main__':
    main()