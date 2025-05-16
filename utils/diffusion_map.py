import numpy as np
import scipy.spatial.distance as dist
from scipy.linalg import eigh


def diffusion_map(X, n_components=2, epsilon=1.0):
    """
    Perform Diffusion Map dimensionality reduction.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        The input data.
    n_components : int, optional, default: 2
        The number of dimensions in the reduced space.
    epsilon : float, optional, default: 1.0
        The kernel scale parameter.

    Returns:
    Y : array, shape (n_samples, n_components)
        The reduced dimensionality data.
    """
    # Compute the pairwise distance matrix
    pairwise_dists = dist.squareform(dist.pdist(X))

    # Compute the affinity matrix
    W = np.exp(-pairwise_dists ** 2 / (2 * epsilon ** 2))

    # Normalize the affinity matrix to get the Markov matrix
    D = np.diag(W.sum(axis=1))
    D_inv = np.linalg.inv(D)
    K = D_inv @ W @ D_inv

    # Eigen decomposition
    evals, evecs = eigh(K)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    evals, evecs = evals[::-1], evecs[:, ::-1]

    # Compute the diffusion coordinates
    Y = evecs[:, 1:n_components + 1] * evals[1:n_components + 1]

    return Y


# Example usage:
if __name__ == "__main__":
    # Create some sample data
    np.random.seed(0)
    X = np.random.rand(100, 10)

    # Perform diffusion map dimensionality reduction
    Y = diffusion_map(X, n_components=2, epsilon=0.1)

    print("Original Data Shape:", X.shape)
    print("Reduced Data Shape:", Y.shape)
