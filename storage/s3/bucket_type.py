from storage.s3.s3_operator import S3Operator
from storage.base import BaseStorage
from typing import Union, Iterator, List
from boto3 import Session
import uuid


class S3Bucket(BaseStorage, S3Operator):

    def __init__(self, client: Session = None, name: str = "", timeout: int = 60,
                 create_new: bool = False) -> None:
        self.client = self._get_client() if client is None else client
        self.name = "lambdaml-" + str(uuid.uuid4()) if name == "" else name
        self.timeout = timeout
        if create_new or name == "":
            self.__create_new_bucket()

    def save(self, src_data: Union[str, bytes], key: str) -> bool:
        return self._put_object(self.client, self.name, key, src_data)

    def load(self, key: str) -> Iterator[bytes]:
        return self._get_object(self.client, self.name, key)

    def load_or_wait(self, key: str, sleep_time: float = 0.1) -> Iterator[bytes]:
        return self._get_object_or_wait(self.client, self.name, key, sleep_time, self.timeout)

    def delete(self, key: Union[str, List[str]]) -> bool:
        if isinstance(key, str):
            return self._delete_object(self.client, self.name, key)
        return self._delete_objects(self.client, self.name, key)

    def clear(self, delete_bucket: bool = False) -> bool:
        ok = self._clear_bucket(self.client, self.name)
        if ok and delete_bucket:
            return self._delete_bucket(self.client, self.name)
        return ok

    def list(self) -> List[Iterator[bytes]]:
        return self._list_bucket_objects(self.client, self.name)

    def __create_new_bucket(self) -> None:
        self._create_bucket(self.client, self.name)

    # TODO(milos) check if these two methods are redundant, and we can just use save/load
    # If they are not redundant, add a test for them in tests/test_storage_s3.py
    def download_file(self, key: str, local_path: str) -> bool:
        return self._download_file(self.client, self.name, key, local_path)

    def upload_file(self, key: str, local_path: str) -> bool:
        return self._upload_file(self.client, self.name, key, local_path)
