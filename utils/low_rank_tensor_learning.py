import numpy as np
import tensorly as tl
from tensorly.tenalg import svd_interface
from numpy.fft import fft, ifft

from utils.calc_connect_sim import calc_connect_sim

def t_svt(tensor, tau):
    # 对张量进行傅里叶变换
    tensor_fft = fft(tensor, axis=2)
    for i in range(tensor_fft.shape[-1]):  # 遍历最后一个维度的所有 slice
        # 使用新的 SVD 接口
        U, S, Vh = svd_interface(tensor_fft[:, :, i], n_eigenvecs=None)
        # 阈值处理奇异值
        S_thresholded = tl.clip(S - tau, a_min=0, a_max=None)
        tensor_fft[:, :, i] = tl.dot(U, tl.dot(tl.diag(S_thresholded), Vh))

    # 对张量进行逆傅里叶变换
    tensor_reconstructed = ifft(tensor_fft, axis=2).real
    print(type(tensor_reconstructed))
    return tensor_reconstructed

if __name__ == '__main__':
    nmf_feat = np.load('/data/zyz/group_connect_mat/joint_feat_NMF500.npy')
    nmf_feat_human = nmf_feat[:20484, :]
    nmf_feat_maca = nmf_feat[20484:, :]
    connect_human = calc_connect_sim(nmf_feat_human, nmf_feat_human.T)
    connect_maca = calc_connect_sim(nmf_feat_maca, nmf_feat_maca.T)
    # connect_human = np.random.rand(10, 10)
    # connect_maca = np.random.rand(10, 10)

    tensor_data = tl.tensor(np.stack([connect_human, connect_maca], axis=2))
    print(tensor_data.shape)
    taus = [1, 2, 5, 10]  # Threshold
    for tau in taus:
        result = t_svt(tensor_data, tau)
        mse = np.mean((result - tensor_data) ** 2)
        print(f"threshold:{tau}, Mean Squared Error (MSE): {mse}")