import numpy as np

def optimize(x, cityCoor, cityDist):
    """
    Calculate the total travel distance for a set of paths.

    Args:
    x (numpy.ndarray): An array representing the set of paths.
    cityCoor (numpy.ndarray): An array containing the coordinates of different cities or points.
    cityDist (numpy.ndarray): An array representing the distances between different cities or points.

    Returns:
    numpy.ndarray: An array containing the calculated total travel distances for each path.
    """

    m = x.shape[0]
    n = cityCoor.shape[0]
    Optim = np.zeros(m)

    for i in range(m):
        for j in range(n - 1):
            Optim[i] += cityDist[x[i, j] - 1, x[i, j + 1] - 1]

        Optim[i] += cityDist[x[i, 0] - 1, x[i, n - 1] - 1]

    return Optim