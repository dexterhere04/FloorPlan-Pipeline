from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class RoomNode:
    id: str
    type: str
    role: Optional[str] = None
    area: Optional[float] = None
    centroid: Optional[Tuple[float, float]] = None


@dataclass
class RoomEdge:
    src: str
    dst: str
    relation: str  
    direction: Optional[str] = None


@dataclass
class FloorplanGraph:
    nodes: List[RoomNode]
    edges: List[RoomEdge]