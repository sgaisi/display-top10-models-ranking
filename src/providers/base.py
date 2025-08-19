from abc import ABC, abstractmethod
from typing import Iterable, Tuple


class Provider(ABC):
    name: str

    @abstractmethod
    def is_ready(self) -> bool: ...

    @abstractmethod
    def list_models(self) -> Iterable[str]: ...

    @abstractmethod
    def run(self, prompt: str, model_id: str) -> Tuple[str, float]: ...
