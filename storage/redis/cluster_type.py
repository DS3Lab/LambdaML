from storage.base import BaseStorage
from storage.redis.redis_operator import RedisOperator
from typing import Union, Iterator, List, Dict, Any
import uuid
import boto3


class RedisCluster(BaseStorage, RedisOperator):

    def __init__(self, create_new: bool = False, new_cluster_args: Dict[str, Any] = None,
                 host: str = "", port: int = 6379) -> None:
        """TODO(milos) describe the arguments required for creating new cluster """
        if create_new:
            if new_cluster_args is None:
                new_cluster_args = dict()
            cluster_id = self.__create_new_cluster(new_cluster_args)
            # now wait while the cluster is available, and then extract the
        self.client = self._get_client(host, port)

    def save(self, src_data: Union[str, bytes], key: str) -> bool:
        """TODO(milos) documentation"""

    def load(self, key: str) -> Iterator[bytes]:
        """TODO(milos) documentation"""

    def load_or_wait(self, key: str) -> Iterator[bytes]:
        """TODO(milos) documentation"""

    def delete(self, key: Union[str, List[str]]) -> bool:
        """TODO(milos) documentation"""

    def clear(self) -> bool:
        """TODO(milos) documentation"""

    def list(self) -> List[Iterator[bytes]]:
        """TODO(milos) documentation"""

    @staticmethod
    def __create_new_cluster(new_cluster_args: Dict[str, Any]) -> str:
        default_arg_values = {
            "CacheClusterId": "lambdaml-" + str(uuid.uuid4()),
            "Engine": "redis",
            "CacheNodeType": "cache.t3.medium",
            "NumCacheNodes": 1,
        }
        for key in default_arg_values:
            if key not in new_cluster_args:
                new_cluster_args[key] = default_arg_values[key]
        if new_cluster_args["Engine"] != "redis":
            raise ValueError("Value of 'Engine' has to be 'redis'")
        ec_client = boto3.client("elasticache")
        response = ec_client.create_cache_cluster(**new_cluster_args)
        return response['CacheCluster']['CacheClusterId']