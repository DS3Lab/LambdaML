from abc import ABC, abstractmethod

class BaseStorage(ABC):

    # TODO(milos) maybe also timeout should be here, since we require derived class to have load_or_wait
    def __init__(self, __name, **kwargs):
        # TODO(milos) maybe rename this to type and don't pass it? But maybe it's completely useless, and delete it
        self.name = __name

    @abstractmethod
    def save(self, src_data, key, **kwargs):
        return

    @abstractmethod
    def load(self, key, **kwargs):
        return

    @abstractmethod
    def load_or_wait(self, key, **kwargs):
        return

    @abstractmethod
    def delete(self, key, **kwargs):
        return

    @abstractmethod
    def clear(self, **kwargs):
        return

    @abstractmethod
    def list(self, **kwargs):
        return
