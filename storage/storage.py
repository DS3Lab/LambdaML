import boto3
from storage import s3_operator, redis_operator, memcached_operator


class BaseStorage(object):

    def __init__(self, __name, **kwargs):
        self.name = __name

    def save(self, src_data, **kwargs):
        return

    def load(self, **kwargs):
        return

    def load_or_wait(self, **kwargs):
        return

    def delete(self, key, **kwargs):
        return

    def clear(self, **kwargs):
        return

    def list(self, **kwargs):
        return


class S3Storage(BaseStorage):

    def __init__(self, name):
        super(S3Storage, self).__init__(name)
        self.client = s3_operator.get_client()

    def save(self, src_data, bucket_name="", object_name=""):
        s3_operator.put_object(self.client, bucket_name, object_name, src_data)

    def load(self, bucket_name="", object_name=""):
        return s3_operator.get_object(self.client, bucket_name, object_name)

    def load_or_wait(self, bucket_name="", object_name="", sleep_time=0.1, time_out=60):
        return s3_operator.get_object_or_wait(self.client, bucket_name, object_name, sleep_time, time_out)

    def delete(self, key, bucket_name=""):
        if isinstance(key, str):
            s3_operator.delete_object(self.client, bucket_name, key)
        elif isinstance(key, list):
            s3_operator.delete_objects(self.client, bucket_name, key)


class RedisStorage(BaseStorage):

    def __init__(self, name, ip, port):
        super(RedisStorage, self).__init__(name)
        self.client = redis_operator.get_client(ip, port)

    def save(self, src_data, key=""):
        redis_operator.set_object(self.client, key, src_data)
        return True

    def load(self, key=""):
        return redis_operator.get_object(self.client, key)

    def load_or_wait(self, key="", sleep_time=0.1, time_out=60):
        return redis_operator.get_object_or_wait(self.client, key, sleep_time, time_out)


class MemcachedStorage(BaseStorage):

    def __init__(self, name, ip, port=11211):
        super(MemcachedStorage, self).__init__(name)
        self.client = memcached_operator.get_client(ip, port)

    def save(self, src_data, key=""):
        memcached_operator.set_object(self.client, key, src_data)
        return True

    def load(self, key=""):
        return memcached_operator.get_object(self.client, key)

    def load_v2(self, key="", field=""):
        return memcached_operator.get_object_v2(self.client, key, field)

    def load_or_wait(self, key="", sleep_time=0.1, time_out=60):
        return memcached_operator.get_object_or_wait(self.client, key, sleep_time, time_out)

    def load_or_wait_v2(self, key="", field="", sleep_time=0.1, time_out=60):
        return memcached_operator.get_object_or_wait_v2(self.client, key, field, sleep_time, time_out)


