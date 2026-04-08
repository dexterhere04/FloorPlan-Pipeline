from pydantic import BaseModel, Field
from typing import List, Optional


class FloorplanRoom(BaseModel):
    id: str = Field(
        description="Unique identifier for the room, e.g. 'living_0', 'dining_0', 'kitchen_0'"
    )
    type: str = Field(
        description="Room type: living, dining, kitchen, bedroom, bathroom, balcony, or front_door"
    )
    role: Optional[str] = Field(
        default=None, description="Optional role: master, guest, etc."
    )


class FloorplanEdge(BaseModel):
    src: str = Field(description="Source room id")
    dst: str = Field(description="Destination room id")
    relation: str = Field(description="Relation type: adjacency, via_door, or direct")
    direction: Optional[str] = Field(
        default=None,
        description="Spatial direction: left_of, right_of, above, or below",
    )


class FloorplanGraphSchema(BaseModel):
    rooms: List[FloorplanRoom] = Field(description="List of all rooms in the floorplan")
    edges: List[FloorplanEdge] = Field(
        description="List of all spatial relationships between rooms"
    )
