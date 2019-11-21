import numpy as np
import torch


from data_loader.LibsvmDataset import SparseLibsvmDataset


class SparseKmeans(object):

    def __init__(self, _data, _centroids, _nr_feature, _nr_cluster, _error=np.iinfo(np.int16).max):
        self.data = _data
        self.nr_feature = _nr_feature
        self.centroids = _centroids
        self.nr_cluster = _nr_cluster
        self.error = _error
        self.model = torch.zeros(self.nr_feature, 1)

    def l2_dist_sq(self, x1, x2):
        diff = torch.sparse.FloatTensor.sub(x1, x2)
        sq_diff = torch.sparse.FloatTensor.mul(diff, diff)
        sum = torch.sparse.sum(sq_diff)
        return sum

    def closest_centroid(self, cent):
        argmin_dist = []
        for i in range(len(self.data)):
            min_sum = np.inf
            idx = 0
            for j in range(len(cent)):
                tmp = self.l2_dist_sq(train_data.ins_list[i], cent[j])
                if tmp < min_sum:
                    idx = j
                    min_sum = tmp
            argmin_dist.append(idx)
        return argmin_dist

    def move_centroids(self, closest, cent):
        c_mean = [torch.sparse.FloatTensor(cent[0].size()[0], cent[0].size()[1]) for i in range(self.nr_cluster)]
        c_count = [0 for i in range(self.nr_cluster)]
        for i in range(len(self.data)):
            c_mean[closest[i]] = torch.sparse.FloatTensor.add(self.data.ins_list[i], c_mean[closest[i]])
            c_count[closest[i]] += 1
        for i in range(self.nr_cluster):
            c_mean[i] = torch.sparse.FloatTensor.div(c_mean[i], c_count[i])
        return c_mean

    def get_error(self, old_cent, new_cent):
        tmp = self.l2_dist_sq(new_cent[0], old_cent[0])
        for i in range(1,self.nr_cluster):
            tmp = torch.sparse.FloatTensor.add(self.l2_dist_sq(new_cent[i], old_cent[i]), tmp)
        return torch.sparse.FloatTensor.div(tmp, self.nr_cluster)

    def find_nearest_cluster(self):
        # centroids.shape: nr_clusters, data_dim
        closest = self.closest_centroid(self.centroids)
        new_centroids = self.move_centroids(closest, self.centroids)
        self.error = self.get_error(self.centroids, new_centroids)
        self.centroids = new_centroids
        return


def get_centroids(train_data, nr_cluster):
    ins_list = []
    for i in range(nr_cluster):
        ins, label = train_data.__getitem__(i)
        ins_list.append(ins)
    return ins_list

if __name__ == "__main__":
    train_file = "../dataset/agaricus_127d_train.libsvm"
    test_file = "../dataset/agaricus_127d_test.libsvm"
    dim = 127
    train_data = SparseLibsvmDataset(train_file, dim)
    test_data = SparseLibsvmDataset(test_file, dim)
    nr_cluster = 10
    centroids = get_centroids(train_data, nr_cluster)
    kmeans_model = SparseKmeans(train_data, centroids, dim, nr_cluster)
    kmeans_model.find_nearest_cluster()
