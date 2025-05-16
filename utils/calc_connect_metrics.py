import bct
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.metrics import r2_score
from sklearn.neighbors import KernelDensity
from math import ceil

def norm_mat(mat):
    mat = (mat + mat.T) / 2  # 保证对称
    np.fill_diagonal(mat, 0)  # 对角线置为0，避免自环


def calc_metric(mat):
    norm_mat(mat)
    # 从numpy矩阵创建图
    G = nx.from_numpy_array(mat)

    mat_metrics = {}
    # 计算图的基本指标
    density = nx.density(G)
    print(f"Density: {density}")
    mat_metrics['density'] = density

    clustering_coeff_average_binary = nx.average_clustering(G)
    print(f"clustering_coeff_average(binary): {clustering_coeff_average_binary}")
    mat_metrics['cca'] = clustering_coeff_average_binary

    # clustering_coeff_average_weighted = nx.average_clustering(G, weight='weight')
    # print(f"clustering_coeff_average(weighted): {clustering_coeff_average_weighted}")
    transitivity_binary = nx.transitivity(G)
    print(f"transitivity(binary): {transitivity_binary}")
    mat_metrics['transitivity'] = transitivity_binary

    # transitivity_weighted = nx.transitivity(G, weight='weight')
    # print(f"transitivity(weighted): {transitivity_weighted}")

    # 检查图是否连通
    if not nx.is_connected(G):
        print("Graph is not connected. Extracting the largest connected component.")
        largest_cc = max(nx.connected_components(G), key=len)  # 最大连通子图
        G = G.subgraph(largest_cc).copy()

    # 仅在图是连通的情况下计算路径相关指标
    if nx.is_connected(G):
        network_characteristic_path_length_binary = nx.average_shortest_path_length(G)
        print(f"network_characteristic_path_length(binary): {network_characteristic_path_length_binary}")
        mat_metrics['ncpl'] = network_characteristic_path_length_binary
        # network_characteristic_path_length_weighted = nx.average_shortest_path_length(G, weight='weight')
        # print(f"network_characteristic_path_length(weighted): {network_characteristic_path_length_weighted}")

        global_efficiency_binary = nx.global_efficiency(G)
        print(f"global_efficiency(binary): {global_efficiency_binary}")
        mat_metrics['ge'] = global_efficiency_binary
        # global_efficiency_weighted = nx.global_efficiency(G, weight='weight')
        # print(f"global_efficiency(weighted): {global_efficiency_weighted}")

        diameter = nx.diameter(G)
        print(f'Diameter: {diameter}')
        mat_metrics['diameter'] = diameter

        radius = nx.radius(G)
        print(f'Radius: {radius}')
        mat_metrics['radius'] = radius
    else:
        print("Graph is not fully connected, skipping path-based metrics.")

    return mat_metrics

def small_worldness(adj_mat):
    '''
    Calculate small-worldness for a given network

    See Humphries & Gurney (2008):
        https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0002051

    Parameters
    ----------
    adj_mat (numpy.array) : Binary matrix of super-threshold correlations
                            Can be found using get_adjacency_matrix()

    Returns
    -------
    S_delta_g (float) : Small-worlness metric value
    '''
    # rand_adj_mat = bct.randmio_und(adj_mat, itr=1)[0]
    # rand_adj_mat = bct.makerandCIJ_und(n=adj_mat.shape[0], k=int(adj_mat.sum()/2))
    rand_adj_mat = nx.convert_matrix.to_numpy_matrix(
        nx.generators.gnm_random_graph(
            n=adj_mat.shape[0],
            m=int(adj_mat.sum()/2)
        )
    )
    C_delta = bct.transitivity_bu(adj_mat)
    C_delta_rand = bct.transitivity_bu(rand_adj_mat)
    L = bct.charpath(bct.distance_bin(adj_mat), include_infinite=False)[0]
    L_rand = bct.charpath(bct.distance_bin(rand_adj_mat), include_infinite=False)[0]

    # Avoid divide by zero error
    # gamma_delta_g = C_delta/C_delta_rand if C_delta_rand>0.0 else C_delta/1e-9
    # lambda_g = L/L_rand if L_rand>0.0 else L/1e-9
    gamma_delta_g = C_delta/C_delta_rand
    lambda_g = L/L_rand

    S_delta_g = gamma_delta_g/lambda_g
    S_delta_g = np.nan if np.isinf(S_delta_g) else S_delta_g
    return S_delta_g


def scale_free_topology_index(adj_mat):
    '''
    Scale-free topology index.

    Basically R^2 score for log-scaled degree vs log-scaled density of degree
       See: https://pdfs.semanticscholar.org/75da/d533fb111ba8ca278967b03f137fc2ff6e8e.pdf

    Parameters
    ----------
    adj_mat (numpy.array) : Binary matrix of super-threshold correlations
                            Can be found using get_adjacency_matrix()

    Returns
    -------
    sft_index (float) : Scale-free topology index metric
    '''

    degrees = bct.degrees_und(adj_mat)
    # Use histogram of degrees to find densities:
    kde = KernelDensity(kernel='gaussian').fit(degrees.reshape(-1, 1))
    degrees_log_dens = kde.score_samples(degrees.reshape(-1, 1))
    degrees_dens = np.exp(degrees_log_dens)

    degrees_log10 = np.log10(degrees)
    degrees_log10 = np.where(np.isnan(degrees_log10), 0, degrees_log10)
    degrees_log10 = np.where(np.isinf(degrees_log10), 0, degrees_log10)

    degrees_dens_log10 = np.log10(degrees_dens)
    degrees_dens_log10 = np.where(np.isnan(degrees_dens_log10), 0, degrees_dens_log10)
    degrees_dens_log10 = np.where(np.isinf(degrees_dens_log10), 0, degrees_dens_log10)

    # # Use R^2
    sft_index = r2_score(degrees_log10, degrees_dens_log10)
    # # Instead of R^2 just use linear regression slope
    # regr = linear_model.LinearRegression()
    # regr.fit(degrees_log10.reshape(-1,1), degrees_dens_log10.reshape(-1,1))
    # sft_index = regr.coef_
    return sft_index

def adjacency_matrix_metrics(adj_mat):
    '''
    Compute a number of network-level metrics on the adjacency matrix

    Uses Brain Connectivity Toolbox functions for :
        density -- 'bct.density_und'
        transitivity -- 'bct.transitivity_bu'
        efficiency -- 'bct.efficiency_bin'
        community q -- 'bct.community_louvain'
        modularity metric -- 'bct.modularity_und'
        assortativity -- 'bct.assortativity_bin'
        characteristic path length -- 'bct.charpath'

    And metrics from this file:
        scale free topology index -- scale_free_topology_index()
        small-worldness -- small_worldness()


    Parameters
    ----------
    adj_mat (numpy.array) : Binary matrix of super-threshold correlations
                            Can be found using get_adjacency_matrix()

    Returns
    -------
    metrics (dict) : Dictionary of calculated metric values
    '''
    metrics = {}
    metrics['density'], metrics['vertices'], metrics['edges'] = bct.density_und(adj_mat)
    print(metrics['density'], metrics['vertices'], metrics['edges'])

    metrics['transitivity'] = bct.transitivity_bu(adj_mat)
    print(metrics['transitivity'])

    metrics['efficiency'] = bct.efficiency_bin(adj_mat)
    print(metrics['efficiency'])

    metrics['community_q'] = bct.community_louvain(adj_mat)[1]
    print(metrics['community_q'])

    metrics['modularity_metric'] = bct.modularity_und(adj_mat)[1]
    print(metrics['modularity_metric'])

    metrics['assortativity'] = bct.assortativity_bin(adj_mat)
    print(metrics['assortativity'])

    metrics['characteristic_path_length'] = bct.charpath(
        bct.distance_bin(adj_mat),
        include_infinite=False
    )[0]

    metrics['scale_free_topology_index'] = scale_free_topology_index(adj_mat)
    print(metrics['scale_free_topology_index'])

    metrics['small_worldness'] = small_worldness(adj_mat)
    print(metrics['small_worldness'])
    return metrics

if __name__ == '__main__':
    h_connect_mat = np.load("../NMF_vertices/NMF_for_human_vertices.npy")
    m_connect_mat = np.load("../NMF_vertices/NMF_for_maca_vertices.npy")
    print(h_connect_mat.shape, m_connect_mat.shape)

    human_conn_mat = np.dot(h_connect_mat[:, :2000].T, h_connect_mat[:, :2000])
    human_metrics = adjacency_matrix_metrics(human_conn_mat)
    print(human_metrics)
    print("--------------------------")

    maca_conn_mat = np.dot(m_connect_mat[:, :1000].T, m_connect_mat[:, :1000])
    maca_metrics = adjacency_matrix_metrics(maca_conn_mat)
    print(maca_metrics)
