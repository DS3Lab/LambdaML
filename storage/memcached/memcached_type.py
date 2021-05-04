from storage.base import BaseStorage

class MemcachedStorage(BaseStorage):

    def __init__(self, name, ip, port=11211):
        super(MemcachedStorage, self).__init__(name)
        self.client = memcached_operator._get_client(ip, port)

    def save(self, src_data, key, **kwargs):
        return memcached_operator._set_object(self.client, key, src_data)

    def load(self, key, **kwargs):
        return memcached_operator._get_object(self.client, key)

    def load_v2(self, key, field):
        return memcached_operator._get_object_v2(self.client, key, field)

    def load_or_wait(self, key, sleep_time=0.1, time_out=60):
        return memcached_operator._get_object_or_wait(self.client, key, sleep_time, time_out)

    def load_or_wait_v2(self, key, field, sleep_time=0.1, time_out=60):
        return memcached_operator._get_object_or_wait_v2(self.client, key, field, sleep_time, time_out)

    def delete(self, key, **kwargs):
        if isinstance(key, str):
            return memcached_operator._delete_key(self.client, key)
        elif isinstance(key, list):
            return memcached_operator._delete_keys(self.client, key)

    def clear(self, **kwargs):
        return memcached_operator._clear_all(self.client)

    def list(self, keys=[""]):
        return memcached_operator._list_keys(self.client, keys)

