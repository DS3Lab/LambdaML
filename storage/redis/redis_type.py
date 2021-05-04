from storage.base import BaseStorage
from storage.redis import redis_operator


class RedisStorage(BaseStorage):

    def __init__(self, name, ip, port):
        super(RedisStorage, self).__init__(name)
        self.client = redis_operator.get_client(ip, port)

    def save(self, src_data, key, **kwargs):
        return redis_operator.set_object(self.client, key, src_data)

    def load(self, key, **kwargs):
        return redis_operator.get_object(self.client, key)

    def load_or_wait(self, key, sleep_time=0.1, time_out=60):
        return redis_operator.get_object_or_wait(self.client, key, sleep_time, time_out)

    def delete(self, key, **kwargs):
        return redis_operator.delete_keys(self.client, key)

    def delete_v2(self, key, fields):
        return redis_operator.delete_fields(self.client, key, fields)

    def clear(self, **kwargs):
        return redis_operator.clear_all(self.client)

    def list(self):
        return redis_operator.list_keys(self.client, count=10000)
