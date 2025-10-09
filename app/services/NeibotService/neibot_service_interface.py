from abc import ABC
from typing import List, Tuple

class NeibotServiceInterface(ABC):
    def get_response(self, history: List[Tuple[str,str]]) -> str:
        raise NotImplementedError("This method should be overridden by subclasses.")