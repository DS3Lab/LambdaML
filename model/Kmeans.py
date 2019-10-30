import numpy as np
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Kmeans:
    def __init__(self, data, centroids, error=np.iinfo(np.int16).max):
        self.X = data
        self.centroids = centroids
        self.error = error

    def l2_dist(self, a, b, ax=1):
        return np.mean(np.linalg.norm(a - b, axis=ax))

    def closest_centroid(self, points, centroids):
        """returns an array containing the index to the nearest centroid for each point"""
        distances = np.sqrt(((points - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def move_centroids(self, points, closest, centroids):
        """returns the new centroids assigned from the points closest to them"""
        return np.array([points[closest == k].mean(axis=0) for k in range(centroids.shape[0])])

    def find_nearest_cluster(self):
        closest = self.closest_centroid(self.X, self.centroids)
        new_centroids_vec = self.move_centroids(self.X, closest, self.centroids)
        self.error = self.l2_dist(self.centroids, new_centroids_vec)
        self.centroids = new_centroids_vec
        return
