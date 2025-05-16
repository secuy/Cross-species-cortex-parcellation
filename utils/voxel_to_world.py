import numpy as np


def voxel_to_world(voxel_coords, affine):
    """
    将体素坐标转换为世界坐标。

    参数：
    - voxel_coords: 体素坐标点云 (N, 3)
    - affine: 仿射矩阵

    返回：
    - 世界坐标点云 (N, 3)
    """
    voxel_coords_trans = np.concatenate((voxel_coords, np.ones((voxel_coords.shape[0], 1))), axis=1)
    world_coords_trans = np.dot(affine, voxel_coords_trans.T)
    return world_coords_trans[:3, :].T
