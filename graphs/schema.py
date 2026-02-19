from dataclasses import dataclass
from typing import List,Optional,Tuple

@dataclass
class RoomNode:
    id: int
    type: str
    role: Optional[str]=None
    area: Optional[float]=None
    centroid: Optional[Tuple[float,float]]= None

@dataclass
class FloorplanGraph:
    nodes: List[RoomNode]
    edges: List[RoomNode]
