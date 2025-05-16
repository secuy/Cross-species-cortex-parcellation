import os

import numpy as np
from dipy.viz import window, actor
def rotate_x(points, angle):
    theta = np.radians(angle)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    return np.dot(points, R_x.T)

def rotate_y(points, angle):
    theta = np.radians(angle)
    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    return np.dot(points, R_y.T)

def rotate_z(points, angle):
    theta = np.radians(angle)
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return np.dot(points, R_z.T)


human_mean = []
human_path = "D:/njust/wuye/week14/gae_baseline/test_data/sub-015_npy/"
for i in range(1000):
    cluster = np.load(human_path + "cluser_{}.npy".format(i))
    human_mean.append(np.mean(cluster, axis=0))

human_mean = np.array(human_mean)
print(human_mean.shape)

maca_mean = []
maca_path = "D:/njust/wuye/week14/gae_baseline/test_data/sub-001_npy/"
for i in range(500):
    cluster = np.load(maca_path + "cluser_{}.npy".format(i))
    maca_mean.append(np.mean(cluster, axis=0))

maca_mean = np.array(maca_mean)
print(maca_mean.shape)



# 绕 z 轴旋转
rotated_points_z = rotate_z(maca_mean, 180)
