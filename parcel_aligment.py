import numpy as np
import nibabel as nib

from cluster_metrics import dice_coef, dice_coef_joined

def parcel_align(human_label, maca_label):
    uni_human_label = np.unique(human_label)
    uni_maca_label = np.unique(maca_label)
    list_score = []
    list_idx = []
    for i in uni_human_label:
        max_score = 0
        max_idx = -1
        for j in uni_maca_label:
            human_idx = np.where(human_label==i)[0]
            maca_idx = np.where(maca_label==j)[0]
            intersection = np.intersect1d(human_idx, maca_idx)
            union = np.union1d(human_idx, maca_idx)
            if max_score < len(intersection)/len(union):
                max_score = len(intersection)/len(union)
                max_idx = j
        # print(max_score, (i, max_idx))
        list_score.append(max_score)
        list_idx.append(max_idx)
    return list_idx, list_score

def labelnpy2gii(labels_1, labels_2, file_path_1, file_path_2, LR, idx, score, cluster_num):


    labels_1 = labels_1.flatten().astype(np.int32)
    labels_2 = labels_2.flatten().astype(np.int32)
    labels_1_mask = (labels_1 != 0).astype(int)
    labels_2_mask = (labels_2 != 0).astype(int)

    _, labels_1, labels_2 = dice_coef_joined(labels_1, labels_2)
    labels_1 *= labels_1_mask
    labels_2 *= labels_2_mask
    np.save(f"./MICCAI_2025/final_labels/{file_path_1.split('/')[-1]}", labels_1)
    np.save(f"./MICCAI_2025/final_labels/{file_path_2.split('/')[-1]}", labels_2)
    label_table_1 = nib.gifti.GiftiLabelTable()
    label_table_2 = nib.gifti.GiftiLabelTable()
    unique_labels_1 = np.unique(labels_1)
    for i in unique_labels_1:
        if i == 0:
            # 当顶点标签为0时，表示是大脑皮层的内侧部分，将这一部分设置为透明
            label_1 = nib.gifti.GiftiLabel(key=i, red=1, green=1, blue=1, alpha=0)
            label_2 = nib.gifti.GiftiLabel(key=i, red=1, green=1, blue=1, alpha=0)
        else:
            # 创建一个随机颜色的标签
            red = np.random.rand()
            green = np.random.rand()
            blue = np.random.rand()
            label_1 = nib.gifti.GiftiLabel(key=i, red=red, green=green, blue=blue, alpha=1)
            label_2 = nib.gifti.GiftiLabel(key=i, red=red, green=green, blue=blue, alpha=1)
        label_1.label = f"Label_{i}"  # 手动设置标签名称
        label_table_1.labels.append(label_1)
        label_2.label = f"Label_{i}"
        label_table_2.labels.append(label_2)
    # 创建 Gifti 数据结构
    gii_data_1 = nib.gifti.GiftiDataArray(data=labels_1, intent='NIFTI_INTENT_LABEL')
    gii_data_2 = nib.gifti.GiftiDataArray(data=labels_2, intent='NIFTI_INTENT_LABEL')
    # 创建 Gifti Image
    gii_img_1 = nib.gifti.GiftiImage(darrays=[gii_data_1], labeltable=label_table_1)
    gii_img_2 = nib.gifti.GiftiImage(darrays=[gii_data_2], labeltable=label_table_2)
    # 设置皮层左右脑
    if LR == "L":
        gii_img_1.meta.data.append(nib.gifti.GiftiNVPairs("AnatomicalStructurePrimary", "CortexLeft"))
        gii_img_2.meta.data.append(nib.gifti.GiftiNVPairs("AnatomicalStructurePrimary", "CortexLeft"))
    else:
        gii_img_1.meta.data.append(nib.gifti.GiftiNVPairs("AnatomicalStructurePrimary", "CortexRight"))
        gii_img_2.meta.data.append(nib.gifti.GiftiNVPairs("AnatomicalStructurePrimary", "CortexRight"))
    # 保存到 .gii 文件
    nib.save(gii_img_1, f"{file_path_1}.label.gii")
    nib.save(gii_img_2, f"{file_path_2}.label.gii")
    
    print("标签已保存为 labels.gii 文件")

if __name__=="__main__":
    for i in range(1, 2):
        human_sub_num = i
        maca_sub_num = i
        LRs = ["L", "R"]
        cluster_nums = [3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]
        parameter_lambda = [1, 100, 500, 5000]
        parameter_rho = [1, 5000]
        for LR in LRs:
            for cluster_num in cluster_nums:
                for lamb in parameter_lambda:
                    for rho in parameter_rho:
                        super_label = np.load(f"./MICCAI_2025/super_labels/super_labels-{cluster_num}_lambda-{lamb}_rho-{rho}_{LR}.npy")
                        human = super_label[0 + human_sub_num - 1]
                        maca = super_label[5 + maca_sub_num - 1]
                        print(super_label.shape)
                        idx, score = parcel_align(human, maca)
                        print(len(score))
                        file_name_1 = f"final_labels-{cluster_num}_lambda-{lamb}_rho-{rho}_human-{human_sub_num}_{LR}.npy"
                        file_name_2 = f"final_labels-{cluster_num}_lambda-{lamb}_rho-{rho}_maca-{maca_sub_num}_{LR}.npy"
                        labels_1 = np.load(f"./MICCAI_2025/final_labels/{file_name_1}")
                        labels_2 = np.load(f"./MICCAI_2025/final_labels/{file_name_2}")
                        file_path_1 = f"./MICCAI_2025/gii_folder/{LR}.Human-{cluster_num}.ZYZ_ISBI_2025_lambda-{lamb}_rho-{rho}"
                        file_path_2 = f"./MICCAI_2025/gii_folder/{LR}.Macaque-{cluster_num}.ZYZ_ISBI_2025_lambda-{lamb}_rho-{rho}"
                        labelnpy2gii(labels_1, labels_2, file_path_1, file_path_2, LR, idx, score, cluster_num)