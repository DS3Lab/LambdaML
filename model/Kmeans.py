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
        batch_size = 5000
        remaining = points.shape[0]
        current_batch = 0
        argmin_dist = []

        while True:
            print(remaining)
            dist = np.full((remaining, centroids.shape[0]), np.inf)
            if remaining <= batch_size:
                for i in range(centroids.shape[0]):
                    dist[:, i] = np.sum(np.square(points[current_batch:current_batch + remaining] - centroids[i, :]),
                                        axis=1)
                argmin_dist.append(np.argmin(dist, axis=1))
                break
            else:
                for i in range(centroids.shape[0]):
                    dist[:, i] = np.sum(np.square(points[current_batch:current_batch + batch_size] - centroids[i, :]),
                                        axis=1)
                argmin_dist.append(np.argmin(dist, axis=1))
                current_batch += batch_size
                remaining -= batch_size

        res = argmin_dist[0]
        print(len(argmin_dist))
        for i in range(1, len(argmin_dist)):
            print(argmin_dist[i].shape)
            res = np.concatenate((res, argmin_dist[i]), axis=None)
        return res

    def move_centroids(self, points, closest, centroids):
        x = np.array([points[closest == k].mean(axis=0) for k in range(centroids.shape[0])])
        return np.nan_to_num(x)

    def find_nearest_cluster(self):
        # centroids.shape: nr_clusters, data_dim
        closest = self.closest_centroid(self.X, self.centroids)
        new_centroids_vec = self.move_centroids(self.X, closest, self.centroids)
        self.error = self.l2_dist(self.centroids, new_centroids_vec)
        self.centroids = new_centroids_vec
        return
