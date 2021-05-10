from storage import memcachedStorage
from communicator import Communicator
from communicator import memcached_primitive, memcached_primitive_nn


class memcachedCommunicator(Communicator):

    def __init__(self, _storage, _tmp_bucket, _merged_bucket, _num_workers, _worker_index):
        super(memcachedCommunicator, self).__init__(_storage)
        assert isinstance(self.storage, MemcachedStorage)
        self.tmp_bucket = _tmp_bucket
        self.merged_bucket = _merged_bucket
        self.num_workers = _num_workers
        self.worker_index = _worker_index

    def reduce_batch(self, vector, postfix=""):
        return memcached_primitive.reduce_batch(self.storage, vector, self.tmp_bucket, self.merged_bucket,
                                         self.num_workers, self.worker_index, postfix)

    def reduce_batch_nn(self, data_bytes, postfix=""):
        return memcached_primitive_nn.reduce_batch(self.storage, data_bytes, self.tmp_bucket, self.merged_bucket,
                                            self.num_workers, self.worker_index, postfix)

    def reduce_epoch(self, vector, postfix=""):
        return memcached_primitive.reduce_epoch(self.storage, vector, self.tmp_bucket, self.merged_bucket,
                                         self.num_workers, self.worker_index, postfix)

    def reduce_epoch_nn(self, data_bytes, postfix=""):
        return memcached_primitive_nn.reduce_epoch(self.storage, data_bytes, self.tmp_bucket, self.merged_bucket,
                                            self.num_workers, self.worker_index, postfix)

    def reduce_scatter_batch(self, vector, postfix=""):
        return memcached_primitive.reduce_scatter_batch(self.storage, vector, self.tmp_bucket, self.merged_bucket,
                                                 self.num_workers, self.worker_index, postfix)

    def reduce_scatter_epoch(self, vector, postfix=""):
        return memcached_primitive.reduce_scatter_epoch(self.storage, vector, self.tmp_bucket, self.merged_bucket,
                                                 self.num_workers, self.worker_index, postfix)

    def delete_expired_batch(self, cur_epoch, cur_batch):
        return memcached_primitive.delete_expired_batch(self.storage, self.merged_bucket, cur_epoch, cur_batch)

    def delete_expired_epoch(self, cur_epoch):
        return memcached_primitive.delete_expired_epoch(self.storage, self.merged_bucket, cur_epoch)

    def async_reduce(self, vector, object_name=""):
        return memcached_primitive.async_reduce(self.storage, vector, self.merged_bucket, object_name)

    def async_reduce_nn(self, data_bytes, object_name=""):
        return memcached_primitive_nn.async_reduce(self.storage, data_bytes, self.merged_bucket, object_name)


