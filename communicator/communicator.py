from storage.storage import S3Storage, RedisStorage, MemcachedStorage
from communicator import s3_primitive


class Communicator(object):

    def __init__(self, __storage):
        self.storage = __storage

    def reduce_batch(self, vector, **kwargs):
        """

        :param vector: merged tensor
        :param kwargs: other params
        :return:
        """
        return

    def reduce_epoch(self, vector, **kwargs):
        """

        :param vector: merged tensor
        :param kwargs: other params
        :return:
        """
        return

    def reduce_scatter_batch(self, vector, **kwargs):
        """

        :param vector: merged tensor
        :param kwargs: other params
        :return:
        """
        return

    def reduce_scatter_epoch(self, vector, **kwargs):
        """

        :param vector: merged tensor
        :param kwargs: other params
        :return:
        """
        return


class S3Communicator(Communicator):

    def __init__(self, __storage, __tmp_bucket, __merged_bucket, __num_workers, __worker_index):
        super(S3Communicator, self).__init__()
        assert isinstance(self.storage, S3Storage)
        self.tmp_bucket = __tmp_bucket
        self.merged_bucket = __merged_bucket
        self.num_workers = __num_workers
        self.worker_index = __worker_index

    def reduce_batch(self, vector, postfix=""):
        return s3_primitive.reduce_batch(self.storage, vector, self.tmp_bucket, self.merged_bucket,
                                         self.num_workers, self.worker_index, postfix)

    def reduce_epoch(self, vector, postfix=""):
        return s3_primitive.reduce_epoch(self.storage, vector, self.tmp_bucket, self.merged_bucket,
                                         self.num_workers, self.worker_index, postfix)

    def reduce_scatter_batch(self, vector, postfix=""):
        return s3_primitive.reduce_scatter_batch(self.storage, vector, self.tmp_bucket, self.merged_bucket,
                                                 self.num_workers, self.worker_index, postfix)

    def reduce_scatter_epoch(self, vector, postfix=""):
        return s3_primitive.reduce_scatter_epoch(self.storage, vector, self.tmp_bucket, self.merged_bucket,
                                                 self.num_workers, self.worker_index, postfix)

    def delete_expired_batch(self, cur_epoch, cur_batch):
        return s3_primitive.delete_expired_batch(self.storage, self.merged_bucket, cur_epoch, cur_batch)

    def delete_expired_epoch(self, cur_epoch):
        return s3_primitive.delete_expired_epoch(self.storage, self.merged_bucket, cur_epoch)
