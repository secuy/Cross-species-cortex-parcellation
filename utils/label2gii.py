import nibabel as nib
import numpy as np

def labelnpy2gii(labels, file_path, LR):

    labels = labels.flatten().astype(np.int32)
    unique_labels = np.unique(labels)
    # print(unique_labels)
    # 创建标签表
    label_table = nib.gifti.GiftiLabelTable()
    for i in unique_labels:
        if i == 0:
            # 当顶点标签为0时，表示是大脑皮层的内侧部分，将这一部分设置为透明
            label = nib.gifti.GiftiLabel(key=i, red=1, green=1, blue=1, alpha=0)
        else:
            # 创建一个随机颜色的标签
            label = nib.gifti.GiftiLabel(key=i, red=np.random.rand(), green=np.random.rand(), blue=np.random.rand(), alpha=1)
        label.label = f"Label_{i}"  # 手动设置标签名称
        label_table.labels.append(label)
    # 创建 Gifti 数据结构
    gii_data = nib.gifti.GiftiDataArray(data=labels, intent='NIFTI_INTENT_LABEL')
    # 创建 Gifti Image
    gii_img = nib.gifti.GiftiImage(darrays=[gii_data], labeltable=label_table)
    # 设置皮层左右脑
    if LR == "L":
        gii_img.meta.data.append(nib.gifti.GiftiNVPairs("AnatomicalStructurePrimary", "CortexLeft"))
    else:
        gii_img.meta.data.append(nib.gifti.GiftiNVPairs("AnatomicalStructurePrimary", "CortexRight"))
    # 保存到 .gii 文件
    nib.save(gii_img, f"{file_path}.label.gii")
    
    print("标签已保存为 labels.gii 文件")

if __name__=='__main__':
    sub_num = 1
    species = "human"
    LR = "L"
    cluster_num = 3
    parameter_lambda = [500] # [1, 10, 100, 500, 1000, 5000, 10000]
    parameter_rho = [1000] # [1, 1000, 5000]
    for lamb in parameter_lambda:
        for rho in parameter_rho:
            file_name = f"align_final_labels-{cluster_num}_lambda-{lamb}_rho-{rho}_{species}-{sub_num}_{LR}.npy"
            labels = np.load(f"../MICCAI_2025/final_labels/{file_name}")

            labelnpy2gii(labels, f"../MICCAI_2025/gii_folder/{file_name.split('.')[0]}", LR)  # {file_name.split('.')[0]}
            # labelnpy2gii(labels_R, f"../gii_folder/{file_name.split('.')[0]}_R.label.gii", 'R')