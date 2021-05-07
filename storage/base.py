from abc import ABC, abstractmethod
from typing import Union, Iterator, List


class BaseStorage(ABC):

    @abstractmethod
    def save(self, src_data: Union[str, bytes], key: str) -> bool:
        """TODO(milos) documentation"""

    @abstractmethod
    def load(self, key: str) -> Iterator[bytes]:
        """TODO(milos) documentation"""

    @abstractmethod
    def load_or_wait(self, key: str) -> Iterator[bytes]:
        """TODO(milos) documentation"""

    @abstractmethod
    def delete(self, key: Union[str, List[str]]) -> bool:
        """TODO(milos) documentation"""

    @abstractmethod
    def clear(self) -> bool:
        """TODO(milos) documentation"""

    @abstractmethod
    def list(self) -> List[Iterator[bytes]]:
        """TODO(milos) documentation"""
