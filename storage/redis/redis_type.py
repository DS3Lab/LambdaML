# NOTE(milos) This type is obsolete, should switch to RedisCluster
from storage.base import BaseStorage
from storage.redis.redis_operator import RedisOperator


class RedisStorage(BaseStorage, RedisOperator):

    def __init__(self, name, ip, port):
        super(RedisStorage, self).__init__(name)
        self.client = self._get_client(ip, port)

    def save(self, src_data, keys, **kwargs):
        return self._set_object(self.client, keys, src_data)

    def save_v2(self, src_data, keys, bucket_name, **kwargs):
        return self._hset_object(self.client, bucket_name, keys, src_data)

    def load(self, key, **kwargs):
        return self._get_object(self.client, key)

    def load_or_wait(self, key, sleep_time=0.1, time_out=60):
        return self._get_object_or_wait(self.client, key, sleep_time, time_out)

    def load_v2(self, key, bucket_name, **kwargs):
        return self._hget_object(self.client, bucket_name, key)

    def load_or_wait_v2(self, key, bucket_name, sleep_time=0.1, time_out=60):
        return self._hget_object_or_wait(self.client, bucket_name, key, sleep_time, time_out)

    def delete(self, keys, **kwargs):
        return self._delete_keys(self.client, keys)

    def delete_v2(self, keys, bucket_name):
        return self._hdelete_keys(self.client, bucket_name, keys)

    def clear(self, **kwargs):
        return self._clear_all(self.client)

    def list(self):
        return self._list_keys(self.client, count=10000)

    def list_v2(self, bucket_name):
        return self._hlist_keys(self.client, bucket_name, count=10000)
