import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import lil_matrix


def compute_vertex_nhood(verticalCoords, meshFaces):
    """
    Compute the adjacency matrices mapping the neighbouring vertices and their Euclidean distance.

    Args:
    verticalCoords (ndarray): Array of shape (nVertices, 3) representing the coordinates of the vertices.
    meshFaces (ndarray): Array of shape (nFaces, 3) representing the faces of the mesh (each face is a triangle defined by 3 vertex indices).

    Returns:
    adj (sparse matrix): Binary adjacency matrix of shape (nVertices, nVertices).
    adj_weighted (sparse matrix): Weighted adjacency matrix of shape (nVertices, nVertices), with weights being the Euclidean distances.
    """

    # print("Compute the adjacency matrices mapping the neighbouring vertices and their Euclidean distance.")
    nVertices = len(verticalCoords)
    dynamicDists = []  # List to store the dynamic distances as tuples (vertex1, vertex2, distance)

    # Iterate over each vertex
    for i in range(nVertices):
        # Find the faces that contain the vertex 'i'
        idx = np.where((meshFaces == i).any(axis=1))[0]

        # Get the neighboring vertices from the faces
        neighs = np.unique(meshFaces[idx, :])

        # Calculate Euclidean distances between vertex 'i' and its neighbours
        dists = cdist([verticalCoords[i]], verticalCoords[neighs])[0]

        # Append the distances to dynamicDists
        for neigh, dist in zip(neighs, dists):
            dynamicDists.append((i, neigh, dist))

    # Convert the dynamicDists to a numpy array
    dynamicDists = np.array(dynamicDists)
    # Create the weighted adjacency matrix using sparse format
    adj_weighted = lil_matrix((nVertices, nVertices))
    for i, neigh, dist in dynamicDists:
        adj_weighted[int(i), int(neigh)] = dist

    # Create the binary adjacency matrix
    adj = adj_weighted.copy()
    adj[adj > 0] = 1

    return adj, adj_weighted