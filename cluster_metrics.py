import nibabel as nib
import numpy as np

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

from utils.compute_vertex_nhood import compute_vertex_nhood


human_L_path = "*"
human_R_path = ""
human_L_mesh = nib.load(human_L_path).darrays[1].data
human_L_data = nib.load(human_L_path).darrays[0].data
human_R_mesh = nib.load(human_R_path).darrays[1].data
human_R_data = nib.load(human_R_path).darrays[0].data

maca_L_path = ""
maca_R_path = ""
maca_L_mesh = nib.load(maca_L_path).darrays[1].data
maca_L_data = nib.load(maca_L_path).darrays[0].data
maca_R_mesh = nib.load(maca_R_path).darrays[1].data
maca_R_data = nib.load(maca_R_path).darrays[0].data

human_adj_L,_ = compute_vertex_nhood(human_L_data, human_L_mesh)
human_adj_R,_ = compute_vertex_nhood(human_R_data, human_R_mesh)
N = human_L_data.shape[0]
human_adj = np.zeros([2 * N, 2 * N])
human_adj[:N, :N] = human_adj_L.toarray()
human_adj[N:, N:] = human_adj_R.toarray()
# print(np.unique(human_adj))

maca_adj_L,_ = compute_vertex_nhood(maca_L_data, maca_L_mesh)
maca_adj_R,_ = compute_vertex_nhood(maca_R_data, maca_R_mesh)
N = maca_L_data.shape[0]
maca_adj = np.zeros([2 * N, 2 * N])
maca_adj[:N, :N] = maca_adj_L.toarray()
maca_adj[N:, N:] = maca_adj_R.toarray()
# print(np.unique(maca_adj))

def get_parcel_graph(vert_graph, label):
    # 其中label从1开始
    parcel_len = np.unique(label).shape[0]
    parcel_adj = np.zeros([parcel_len, parcel_len])
    for i in range(parcel_len):
        idx = np.where(label == i+1)[0]
        parcel_sub_adj = vert_graph[idx]
        merge_parcel = np.sum(parcel_sub_adj, axis=0)
        vert_j = np.where(merge_parcel != 0)[0]
        parcel_adj[i, label[vert_j] - 1] = 1
        parcel_adj[label[vert_j] - 1, i] = 1
    return parcel_adj


def find_dist_to_neighs(ids, parcels, D, in_members):
    """
    Compute the average distance to vertices in adjacent parcels.

    Parameters:
    ids: array-like
        The IDs of adjacent parcels.
    parcels: array-like
        The parcellation labels for each vertex.
    D: array-like
        The distance matrix.
    in_members: array-like
        Boolean array indicating vertices in the current parcel.

    Returns:
    votes_out: array-like
        The average distances to adjacent parcels for each vertex in the current parcel.
    """
    out_members = np.zeros_like(parcels, dtype=bool)
    for j in ids:
        out_members |= parcels == j

    nk = np.sum(out_members)
    corrs = D[np.ix_(in_members, out_members)]
    votes_out = np.sum(corrs, axis=1) / nk
    return votes_out

def calc_parcel_SI(feat, label, species):
    if species == 'human':
        adj = human_adj
    else:
        adj = maca_adj
    label = label.astype(int)
    # print(label.shape)
    parcel_adj = get_parcel_graph(adj, label)
    np.fill_diagonal(parcel_adj, 0)
    return np.mean(silhouette_coef(label, cosine_distances(feat), parcel_adj))

def homogeneity(parcels, Z):
    """
    Compute the homogeneity of each parcel.

    Parameters:
    -----------
    parcels : ndarray
        An array of size (N,) indicating the parcellation assignment of each vertex.
    Z : ndarray
        An (N, N) correlation matrix (ideally Fisher's r-to-z transformed).

    Returns:
    --------
    homogeneities : ndarray
        A (K,) array where the kth element indicates the homogeneity value of the kth parcel.
    """
    K = np.max(parcels)  # Number of parcels
    homogeneities = np.zeros(K)  # Initialize homogeneities

    for i in range(1, K + 1):  # Parcels are assumed to be indexed from 1
        in_members = parcels == i  # Boolean mask for vertices in the current parcel
        nk = np.sum(in_members)  # Number of vertices in the parcel

        if nk < 2:  # Singleton parcel or empty
            ak = 1
        else:
            # Extract the submatrix for the current parcel
            corrs = Z[np.ix_(in_members, in_members)]
            np.fill_diagonal(corrs, 0)  # Set diagonal to 0
            means_in = np.sum(corrs, axis=1) / (nk - 1)  # Average similarity per vertex
            ak = np.mean(means_in)  # Average similarity across all vertices in the parcel

        homogeneities[i - 1] = ak  # Store homogeneity for parcel i

    return homogeneities

def relabel(parcels):
    unique_elems = np.unique(parcels)
    ids = unique_elems[unique_elems != 0]
    K = len(ids)
    if K == 0:
        return np.zeros_like(parcels), 0
    if np.max(parcels) == K:
        return parcels.copy(), K
    else:
        relabeled = np.zeros_like(parcels)
        new_id = 1
        for old_id in ids:
            relabeled[parcels == old_id] = new_id
            new_id += 1
        return relabeled, K

def count_unique_elements(v):
    v_flat = v.flatten()
    if len(v_flat) == 0:
        return np.array([]), np.array([])
    uniqs, counts = np.unique(v_flat, return_counts=True)
    sorted_indices = np.argsort(-counts)
    uniqs_sorted = uniqs[sorted_indices]
    counts_sorted = counts[sorted_indices]
    return uniqs_sorted, counts_sorted

def dice_coef(U, V):
    parcels1, K1 = relabel(U)
    parcels2, K2 = relabel(V)
    
    Umatched = np.zeros_like(parcels1)
    Vmatched = np.zeros_like(parcels2)
    
    check1 = np.zeros(K1, dtype=int)
    check2 = np.zeros(K2, dtype=int)
    
    max_k = max(K1, K2)
    pairs = np.zeros((max_k, 2), dtype=int)
    overlaps = np.zeros(max_k, dtype=int)
    dices = np.zeros(max_k)
    
    # First pass
    for i in range(1, K1 + 1):
        id1 = i
        mask = parcels1 == id1
        v = parcels2[mask]
        if v.size == 0:
            continue
        uniqs, counts = count_unique_elements(v)
        if uniqs.size == 0:
            continue
        overlap = counts[0]
        id2 = uniqs[0]
        mask_v = parcels2 == id2
        v_rev = parcels1[mask_v]
        uniqs_rev, counts_rev = count_unique_elements(v_rev)
        if uniqs_rev.size == 0:
            cmp = -1
        else:
            cmp = uniqs_rev[0]
        if cmp == id1:
            check1[id1-1] = 1
            check2[id2-1] = 1
            pairs[id1-1, 0] = id1
            pairs[id1-1, 1] = id2
            overlaps[id1-1] = overlap
            size_id1 = np.sum(mask)
            size_id2 = np.sum(mask_v)
            dices[id1-1] = (2 * overlap) / (size_id1 + size_id2) if (size_id1 + size_id2) != 0 else 0
    
    # Second pass
    unassigned_ids = np.where(check2 == 0)[0] + 1
    for id2 in unassigned_ids:
        mask = parcels2 == id2
        v = parcels1[mask]
        if v.size == 0:
            continue
        uniqs, counts = count_unique_elements(v)
        if uniqs.size == 0:
            continue
        id1 = uniqs[0]
        if check1[id1-1] == 0:
            check1[id1-1] = 1
            check2[id2-1] = 1
            pairs[id1-1, 0] = id1
            pairs[id1-1, 1] = id2
            overlap = counts[0]
            overlaps[id1-1] = overlap
            size_id1 = np.sum(parcels1 == id1)
            size_id2 = np.sum(mask)
            dices[id1-1] = (2 * overlap) / (size_id1 + size_id2) if (size_id1 + size_id2) != 0 else 0
    
    # Third pass
    unassigned_ids = np.where(check1 == 0)[0] + 1
    for id1 in unassigned_ids:
        mask = parcels1 == id1
        v = parcels2[mask]
        if v.size == 0:
            continue
        uniqs, counts = count_unique_elements(v)
        if uniqs.size == 0:
            continue
        id2 = uniqs[0]
        if check2[id2-1] == 0:
            check1[id1-1] = 1
            check2[id2-1] = 1
            pairs[id1-1, 0] = id1
            pairs[id1-1, 1] = id2
            overlap = counts[0]
            overlaps[id1-1] = overlap
            size_id1 = np.sum(mask)
            size_id2 = np.sum(parcels2 == id2)
            dices[id1-1] = (2 * overlap) / (size_id1 + size_id2) if (size_id1 + size_id2) != 0 else 0
    
    # Fourth pass
    unassigned_ids = np.where(check2 == 0)[0] + 1
    for id2 in unassigned_ids:
        mask = parcels2 == id2
        v = parcels1[mask]
        if v.size == 0:
            continue
        uniqs, counts = count_unique_elements(v)
        if uniqs.size == 0:
            continue
        id1 = uniqs[0]
        if check1[id1-1] == 0:
            check1[id1-1] = 1
            check2[id2-1] = 1
            pairs[id1-1, 0] = id1
            pairs[id1-1, 1] = id2
            overlap = counts[0]
            overlaps[id1-1] = overlap
            size_id1 = np.sum(parcels1 == id1)
            size_id2 = np.sum(mask)
            dices[id1-1] = (2 * overlap) / (size_id1 + size_id2) if (size_id1 + size_id2) != 0 else 0
    
    # Sort dices and pairs
    sorted_indices = np.argsort(-dices)
    dices[:] = dices[sorted_indices]
    pairs[:] = pairs[sorted_indices]
    
    # Assign matched labels
    current_id = 1
    for a, b in pairs:
        if b == 0:
            continue
        Umatched[parcels1 == a] = current_id
        Vmatched[parcels2 == b] = current_id
        current_id += 1
    
    # Assign unmatched labels
    unique_p1 = np.unique(parcels1)
    unique_p1 = unique_p1[unique_p1 != 0]
    matched_p1 = np.where(check1 == 1)[0] + 1
    unassigned1 = np.setdiff1d(unique_p1, matched_p1)
    
    unique_p2 = np.unique(parcels2)
    unique_p2 = unique_p2[unique_p2 != 0]
    matched_p2 = np.where(check2 == 1)[0] + 1
    unassigned2 = np.setdiff1d(unique_p2, matched_p2)
    
    max_un = max(len(unassigned1), len(unassigned2))
    for i in range(max_un):
        if i < len(unassigned1):
            Umatched[parcels1 == unassigned1[i]] = current_id
        if i < len(unassigned2):
            Vmatched[parcels2 == unassigned2[i]] = current_id
        current_id += 1
    
    # Final check for remaining pairs
    rems = np.where(dices == 0)[0]
    for rem in rems:
        label = rem + 1  # Assuming rem is 0-based index
        aa1 = Umatched == label
        aa2 = Vmatched == label
        intersection = np.logical_and(aa1, aa2).sum()
        sum_aa1 = aa1.sum()
        sum_aa2 = aa2.sum()
        if sum_aa1 + sum_aa2 > 0:
            dice_val = (2 * intersection) / (sum_aa1 + sum_aa2)
            dices[rem] = dice_val
    
    dice = np.mean(dices[dices > 0]) if np.any(dices > 0) else 0.0
    
    return dice, Umatched, Vmatched

def dice_coef_joined(U, V):
    parcels1, K1 = relabel(U)
    parcels2, K2 = relabel(V)
    
    Ujoined = np.zeros_like(parcels1)
    Vjoined = np.zeros_like(parcels2)
    
    check1 = np.zeros(K1, dtype=int)
    check2 = np.zeros(K2, dtype=int)
    
    max_k = max(K1, K2)
    pairs = np.zeros((max_k, 2), dtype=int)
    overlaps = np.zeros(max_k, dtype=int)
    dices = np.zeros(max_k)
    
    # JOIN: First pass
    for i in range(1, K1 + 1):
        mask = parcels1 == i
        v = parcels2[mask]
        if v.size == 0:
            continue
        instances, values = count_unique_elements(v)
        sums = np.array([np.sum(parcels2 == x) for x in instances])
        res = (values / sums) >= 0.5
        if np.sum(res) > 1:
            merged = instances[res]
            newid = merged[0]
            for m in merged:
                parcels2[parcels2 == m] = newid
    
    # JOIN: Second pass
    clusters2 = np.unique(parcels2)
    for cluster in clusters2:
        mask = parcels2 == cluster
        v = parcels1[mask]
        if v.size == 0:
            continue
        instances, values = count_unique_elements(v)
        sums = np.array([np.sum(parcels1 == x) for x in instances])
        res = (values / sums) >= 0.5
        if np.sum(res) > 1:
            merged = instances[res]
            newid = merged[0]
            for m in merged:
                parcels1[parcels1 == m] = newid
    
    # MATCH: First pass
    clusters1 = np.unique(parcels1)
    for id1 in clusters1:
        mask = parcels1 == id1
        v = parcels2[mask]
        if v.size == 0:
            continue
        uniqs, counts = count_unique_elements(v)
        overlap = counts[0]
        id2 = uniqs[0]
        mask_v = parcels2 == id2
        v_rev = parcels1[mask_v]
        uniqs_rev, counts_rev = count_unique_elements(v_rev)
        cmp = uniqs_rev[0] if uniqs_rev.size > 0 else -1
        if cmp == id1:
            check1[id1-1] = 1
            check2[id2-1] = 1
            pairs[id1-1, 0] = id1
            pairs[id1-1, 1] = id2
            overlaps[id1-1] = overlap
            size_id1 = np.sum(mask)
            size_id2 = np.sum(mask_v)
            dices[id1-1] = (2 * overlap) / (size_id1 + size_id2) if (size_id1 + size_id2) != 0 else 0
    
    # MATCH: Second pass
    unassigned_ids = np.setdiff1d(np.unique(parcels2), np.where(check2 == 1)[0] + 1)
    for id2 in unassigned_ids:
        mask = parcels2 == id2
        v = parcels1[mask]
        if v.size == 0:
            continue
        uniqs, counts = count_unique_elements(v)
        overlap = counts[0]
        id1 = uniqs[0]
        if check1[id1-1] == 0:
            check1[id1-1] = 1
            check2[id2-1] = 1
            pairs[id1-1, 0] = id1
            pairs[id1-1, 1] = id2
            overlaps[id1-1] = overlap
            size_id1 = np.sum(parcels1 == id1)
            size_id2 = np.sum(mask)
            dices[id1-1] = (2 * overlap) / (size_id1 + size_id2) if (size_id1 + size_id2) != 0 else 0
    
    # MATCH: Third pass
    unassigned_ids = np.setdiff1d(np.unique(parcels1), np.where(check1 == 1)[0] + 1)
    for id1 in unassigned_ids:
        mask = parcels1 == id1
        v = parcels2[mask]
        if v.size == 0:
            continue
        uniqs, counts = count_unique_elements(v)
        overlap = counts[0]
        id2 = uniqs[0]
        if check2[id2-1] == 0:
            check1[id1-1] = 1
            check2[id2-1] = 1
            pairs[id1-1, 0] = id1
            pairs[id1-1, 1] = id2
            overlaps[id1-1] = overlap
            size_id1 = np.sum(mask)
            size_id2 = np.sum(parcels2 == id2)
            dices[id1-1] = (2 * overlap) / (size_id1 + size_id2) if (size_id1 + size_id2) != 0 else 0
    
    # Remove empty pairs
    valid_pairs = pairs[:, 0] != 0
    pairs = pairs[valid_pairs]
    dices = dices[valid_pairs]
    
    # Sort by Dice score
    sorted_indices = np.argsort(-dices)
    dices = dices[sorted_indices]
    pairs = pairs[sorted_indices]
    
    # Assign matched labels
    current_id = 1
    for a, b in pairs:
        Ujoined[parcels1 == a] = current_id
        Vjoined[parcels2 == b] = current_id
        current_id += 1
    
    # Assign unmatched labels
    unassigned1 = np.setdiff1d(np.unique(parcels1), np.where(check1 == 1)[0] + 1)
    unassigned2 = np.setdiff1d(np.unique(parcels2), np.where(check2 == 1)[0] + 1)
    max_un = max(len(unassigned1), len(unassigned2))
    for i in range(max_un):
        if i < len(unassigned1):
            Ujoined[parcels1 == unassigned1[i]] = current_id
        if i < len(unassigned2):
            Vjoined[parcels2 == unassigned2[i]] = current_id
        current_id += 1
    
    # Final Dice score calculation
    dices = np.append(dices, np.zeros(max(np.max(Ujoined), np.max(Vjoined)) - len(dices)))
    dice = np.mean(dices)
    
    return dice, Ujoined, Vjoined

def dice_atlas_align(parcels, atlas):
    """
    对齐目标分区与参考图谱，计算Dice相似度系数
    :param parcels: 目标分区 (numpy数组)
    :param atlas: 参考图谱 (numpy数组)
    :return: dice系数, 对齐后的分区
    """
    # ========== 预处理阶段 ==========
    parcels_orig = parcels.copy()
    
    # 将非重叠区域置零
    parcels[atlas == 0] = 0
    
    # 重新编号目标分区标签
    new_parcels = np.zeros_like(parcels)
    unique_parcels = np.unique(parcels)
    unique_parcels = unique_parcels[unique_parcels != 0]
    for idx, p in enumerate(unique_parcels, 1):
        new_parcels[parcels_orig == p] = idx
    
    # 重新编号参考图谱标签
    joined = np.zeros_like(atlas)
    unique_atlas = np.unique(atlas)
    unique_atlas = unique_atlas[unique_atlas != 0]
    for idx, a in enumerate(unique_atlas, 1):
        joined[atlas == a] = idx
    
    parcels = new_parcels
    atlas = joined
    
    # ========== 合并阶段 ==========
    # 合并与参考图谱高度重叠的区域
    atlas_clusters = np.unique(atlas)
    atlas_clusters = atlas_clusters[atlas_clusters != 0]
    
    for cluster in atlas_clusters:
        mask = atlas == cluster
        v = parcels[mask]
        instances, values = count_unique_elements(v)
        if len(instances) == 0:
            continue
        
        # 计算重叠比例
        sums = np.array([np.sum(parcels == x) for x in instances])
        res = (values / sums) >= 0.5
        
        if np.sum(res) > 1:
            merged = instances[res]
            new_id = merged[0]
            for m in merged:
                parcels[parcels == m] = new_id

    # ========== 匹配阶段 ==========
    # 动态设置check_parcels的大小
    max_parcel_id = np.max(parcels)
    check_parcels = np.zeros(max_parcel_id + 1, dtype=int)  # 确保大小足够
    check_atlas = np.zeros(len(atlas_clusters) + 1, dtype=int)
    
    pairs = np.zeros((len(atlas_clusters) + 1, 2), dtype=int)  # 存储匹配对
    dices = np.zeros(len(atlas_clusters) + 1)
    
    # 主要匹配过程
    for cluster in atlas_clusters:
        mask = atlas == cluster
        v = parcels[mask]
        instances, counts = count_unique_elements(v)
        if len(instances) == 0:
            continue
        
        # 计算加权得分
        sums = np.array([np.sum(parcels == x) for x in instances])
        scores = counts / sums
        best_idx = np.argmax(scores)
        parcel_id = instances[best_idx]
        
        if parcel_id == 0:
            continue
        
        # 反向验证匹配
        mask_parcel = parcels == parcel_id
        v_atlas = atlas[mask_parcel]
        atlas_instances, atlas_counts = count_unique_elements(v_atlas)
        if len(atlas_instances) == 0:
            continue
        
        atlas_sums = np.array([np.sum(atlas == x) for x in atlas_instances])
        atlas_scores = atlas_counts / atlas_sums
        best_atlas_idx = np.argmax(atlas_scores)
        
        if atlas_instances[best_atlas_idx] == cluster:
            # 记录匹配对
            check_parcels[parcel_id] = 1
            check_atlas[cluster] = 1
            pairs[cluster] = [cluster, parcel_id]
            
            # 计算Dice系数
            intersection = np.sum(mask & mask_parcel)
            total = np.sum(mask) + np.sum(mask_parcel)
            dices[cluster] = 2 * intersection / total if total != 0 else 0

    # ========== 后处理 ==========
    # 生成最终对齐结果
    aligned = np.zeros_like(atlas)
    current_id = 1
    for a_cluster in atlas_clusters:
        p_id = pairs[a_cluster][1]
        if p_id != 0:
            aligned[parcels == p_id] = current_id
            current_id += 1
    
    # 处理未匹配区域
    unmatched_parcels = np.setdiff1d(np.unique(parcels), np.where(check_parcels)[0])
    for p in unmatched_parcels:
        if p != 0:
            aligned[parcels == p] = current_id
            current_id += 1

    # 计算最终Dice系数
    valid_dices = dices[dices != 0]
    final_dice = np.mean(valid_dices) if len(valid_dices) > 0 else 0.0
    
    return final_dice, aligned

def cluster_metric_4_single_sub(feature, label, species):
    # species include: human and maca(macaque)
    DBI_score = davies_bouldin_score(feature, label)
    CHI_score = calinski_harabasz_score(feature, label)
    homo = np.mean(homogeneity(label, cosine_similarity(feature)))
    return DBI_score, CHI_score, homo

def calc_species_transfer_percent(my_label, my_trans_label, atlas_label, atlas_trans_label):
    atlas_map_table = {}
    for idx, l in enumerate(atlas_label):
        trans_idx  = np.where(atlas_trans_label == l)[0]
        atlas_map_table[idx] = trans_idx

    tot_map_num = 0
    for idx, l in enumerate(my_label):
        for k in atlas_map_table[idx]:
            if my_trans_label[k] == my_label[idx]:
                tot_map_num += 1
                break
    return tot_map_num / len(my_label)

if __name__ == "__main__":
    parameter_lambda = [1, 10, 100, 500, 1000, 5000, 10000]
    parameter_rho = [1, 1000, 5000]
    dice_avg = []
    dice_joined_avg = []
    human_atlas_list = ["Brodmann09.human.10k_fs_LR.label.gii",
                        "B05.monkey-to-human.10k_fs_LR.label.gii",
                        "Markov.monkey-to-human.10k_fs_LR.label.gii"]
    maca_atlas_list = ["Brodmann09.human-to-monkey.10k_fs_LR.label.gii",
                    "B05.monkey.10k_fs_LR.label.gii",
                    "Markov.monkey.10k_fs_LR.label.gii"]
    human_parcel_list = ["Schaefer2018_100Parcels_17Networks_order.human.10k_fs_LR.label.gii",
                        "Schaefer2018_200Parcels_17Networks_order.human.10k_fs_LR.label.gii",
                        "Schaefer2018_300Parcels_17Networks_order.human.10k_fs_LR.label.gii",
                        "Schaefer2018_400Parcels_17Networks_order.human.10k_fs_LR.label.gii",
                        "Glasser2016.human.10k_fs_LR.label.gii",
                        "Yeo2011_17Networks_N1000.human.10k_fs_LR.label.gii",
                        "Yeo2011_7Networks_N1000.human.10k_fs_LR.label.gii"]
    maca_parcel_list = ["Schaefer2018_100Parcels_17Networks_order.human-to-monkey.10k_fs_LR.label.gii",
                       "Schaefer2018_200Parcels_17Networks_order.human-to-monkey.10k_fs_LR.label.gii",
                       "Schaefer2018_300Parcels_17Networks_order.human-to-monkey.10k_fs_LR.label.gii",
                       "Schaefer2018_400Parcels_17Networks_order.human-to-monkey.10k_fs_LR.label.gii",
                       "Glasser2016.human-to-monkey.10k_fs_LR.label.gii",
                       "Yeo2011_17Networks_N1000.human-to-monkey.10k_fs_LR.label.gii",
                       "Yeo2011_7Networks_N1000.human-to-monkey.10k_fs_LR.label.gii"]
    
    label_1 = np.load(f"")
    label_2 = np.load(f"")
    dice, Umatched, Vmatched = dice_coef(label_1, label_2)
    dice_joined, Ujoined, Vjoined = dice_coef_joined(label_1, label_2)
    print(dice, dice_joined)