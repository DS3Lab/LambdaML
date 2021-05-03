# NOTE(milos) This is deprecated, we should switch to S3Bucket everywhere
from storage.s3 import s3_operator
from storage.base import BaseStorage

class S3Storage(BaseStorage):

    # TODO(milos) allow user to specify aws credentials in any they want, and read from ~/.aws/credentials by default
    # We should follow "configuring credentials" as described in the documentation
    def __init__(self, name):
        # TODO(milos) I think this is not a proper way to call super constructor, check this
        super(S3Storage, self).__init__(name)
        self.client = s3_operator._get_client()

    def save(self, src_data, object_name, bucket_name=""):
        return s3_operator._put_object(self.client, bucket_name, object_name, src_data)

    def load(self, object_name, bucket_name=""):
        return s3_operator._get_object(self.client, bucket_name, object_name)

    def load_or_wait(self, object_name, bucket_name="", sleep_time=0.1, time_out=60):
        return s3_operator._get_object_or_wait(self.client, bucket_name, object_name, sleep_time, time_out)

    def delete(self, key, bucket_name=""):
        if isinstance(key, str):
            return s3_operator._delete_object(self.client, bucket_name, key)
        elif isinstance(key, list):
            return s3_operator._delete_objects(self.client, bucket_name, key)

    def clear(self, bucket_name=""):
        return s3_operator._clear_bucket(self.client, bucket_name)

    def list(self, bucket_name=""):
        return s3_operator._list_bucket_objects(self.client, bucket_name)

    # TODO(milos) it's missing clear function

