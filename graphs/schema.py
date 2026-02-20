from dataclasses import dataclass
from typing import List, Optional, Tuple


# -----------------------------
# Node: Room / architectural element
# -----------------------------

@dataclass
class RoomNode:
    id: str                     # e.g. "living_0"
    type: str                   # living | bedroom | kitchen | bathroom | balcony | front_door
    role: Optional[str] = None  # master, guest, etc. (future)
    area: Optional[float] = None
    centroid: Optional[Tuple[float, float]] = None


# -----------------------------
# Edge: Spatial / connectivity relation
# -----------------------------

@dataclass
class RoomEdge:
    src: str
    dst: str
    relation: str               # adjacency | via_door | via_window | direct
    direction: Optional[str] = None  # left_of | right_of | above | below (optional future)


# -----------------------------
# Graph container
# -----------------------------

@dataclass
class FloorplanGraph:
    nodes: List[RoomNode]
    edges: List[RoomEdge]