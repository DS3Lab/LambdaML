from storage import s3_operator, redis_operator, memcached_operator


class BaseStorage(object):

    def __init__(self, __name, **kwargs):
        self.name = __name

    def save(self, src_data, key, **kwargs):
        return

    def load(self, key, **kwargs):
        return

    def load_or_wait(self, key, **kwargs):
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

    def save(self, src_data, object_name, bucket_name=""):
        return s3_operator.put_object(self.client, bucket_name, object_name, src_data)

    def load(self, object_name, bucket_name=""):
        return s3_operator.get_object(self.client, bucket_name, object_name)

    def load_or_wait(self, object_name, bucket_name="", sleep_time=0.1, time_out=60):
        return s3_operator.get_object_or_wait(self.client, bucket_name, object_name, sleep_time, time_out)

    def delete(self, key, bucket_name=""):
        if isinstance(key, str):
            return s3_operator.delete_object(self.client, bucket_name, key)
        elif isinstance(key, list):
            return s3_operator.delete_objects(self.client, bucket_name, key)

    def clear(self, bucket_name=""):
        return s3_operator.clear_bucket(self.client, bucket_name)

    def list(self, bucket_name=""):
        return s3_operator.list_bucket_objects(self.client, bucket_name)

    def download_file(self, bucket_name, object_name, local_path):
        return s3_operator.download_file(self.client, bucket_name, object_name, local_path)

    def upload_file(self, bucket_name, object_name, local_path):
        return s3_operator.upload_file(self.client, bucket_name, object_name, local_path)


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


class MemcachedStorage(BaseStorage):

    def __init__(self, name, ip, port=11211):
        super(MemcachedStorage, self).__init__(name)
        self.client = memcached_operator.get_client(ip, port)

    def save(self, src_data, key, **kwargs):
        return memcached_operator.set_object(self.client, key, src_data)

    def load(self, key, **kwargs):
        return memcached_operator.get_object(self.client, key)

    def load_v2(self, key, field):
        return memcached_operator.get_object_v2(self.client, key, field)

    def load_or_wait(self, key, sleep_time=0.1, time_out=60):
        return memcached_operator.get_object_or_wait(self.client, key, sleep_time, time_out)

    def load_or_wait_v2(self, key, field, sleep_time=0.1, time_out=60):
        return memcached_operator.get_object_or_wait_v2(self.client, key, field, sleep_time, time_out)

    def delete(self, key, **kwargs):
        if isinstance(key, str):
            return memcached_operator.delete_key(self.client, key)
        elif isinstance(key, list):
            return memcached_operator.delete_keys(self.client, key)

    def clear(self, **kwargs):
        return memcached_operator.clear_all(self.client)

    def list(self, keys=[""]):
        return memcached_operator.list_keys(self.client, keys)


