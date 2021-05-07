# NOTE(milos) This type is obsolete, should switch to S3Bucket
from storage.s3.s3_operator import S3Operator
from storage.base import BaseStorage


class S3Storage(BaseStorage, S3Operator):

    def __init__(self):
        self.client = self._get_client()

    def save(self, src_data, object_name, bucket_name=""):
        return self._put_object(self.client, bucket_name, object_name, src_data)

    def load(self, object_name, bucket_name=""):
        return self._get_object(self.client, bucket_name, object_name)

    def load_or_wait(self, object_name, bucket_name="", sleep_time=0.1, time_out=60):
        return self._get_object_or_wait(self.client, bucket_name, object_name, sleep_time, time_out)

    def delete(self, key, bucket_name=""):
        if isinstance(key, str):
            return self._delete_object(self.client, bucket_name, key)
        elif isinstance(key, list):
            return self._delete_objects(self.client, bucket_name, key)

    def clear(self, bucket_name=""):
        return self._clear_bucket(self.client, bucket_name)

    def list(self, bucket_name=""):
        return self._list_bucket_objects(self.client, bucket_name)

    def download(self, bucket_name, object_name, local_path):
        return self._download_file(self.client, bucket_name, object_name, local_path)

    def upload(self, bucket_name, object_name, local_path):
        return self._upload_file(self.client, bucket_name, object_name, local_path)
