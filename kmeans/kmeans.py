import numpy as np


def kmeans(X: np.ndarray, k: int, centroids=None, max_iter=30, tolerance=1e-2):
    X = X.astype("float64")
    if centroids == None:
        centroids = X[np.random.choice(len(X), size=k, replace=False)]
    elif centroids == "kmeans++":
        centroids = select_centroids(X, k)
    clusters = [[] for i in range(k)]

    centroids_tmp = centroids
    for i in range(max_iter):
        labels = []
        for x in X:
            j = np.argmin(np.sum(np.square(x - centroids), axis=1))
            clusters[j].append(x)
            labels.append(j)
        for j in range(k):
            centroids[j] = np.mean(clusters[j], axis=0)
        if (
            np.mean(np.sqrt(np.sum(np.square(centroids - centroids_tmp), axis=1)))
            < tolerance
        ):
            break
        centroids_tmp = centroids

    return centroids, labels


def select_centroids(X, k):
    """
    kmeans++ algorithm to select initial points:

    1. Pick first point randomly
    2. Pick next k-1 points by selecting points that maximize the minimum
       distance to all existing clusters. So for each point, compute distance
       to each cluster and find that minimum.  Among the min distances to a cluster
       for each point, find the max distance. The associated point is the new centroid.

    Return centroids as k x p array of points from X.
    """
    centroids = np.array([X[np.random.randint(len(X))]])

    for i in range(k - 1):
        distances = []
        for x in X:
            distances.append(np.min(np.sum(np.square(x - centroids), axis=1)))
        centroids = np.append(centroids, [X[np.argmax(distances)]], axis=0)

    return centroids.astype("float64")