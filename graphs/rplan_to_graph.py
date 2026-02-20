from shapely.geometry import Polygon
from .schema import RoomNode, RoomEdge, FloorplanGraph
from .extract_adjacency import are_adjacent, spatial_direction


def rplan_to_graph(rooms):
    """
    rooms: list of dict {id, type, polygon}
    """

    nodes = []
    edges = []

    # build nodes
    for r in rooms:
        poly = Polygon(r["polygon"])
        cx, cy = poly.centroid.x, poly.centroid.y

        nodes.append(
            RoomNode(
                id=r["id"],
                type=r["type"],
                area=poly.area,
                centroid=(cx, cy),
            )
        )

    # adjacency
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            poly_i = Polygon(rooms[i]["polygon"])
            poly_j = Polygon(rooms[j]["polygon"])

            if are_adjacent(poly_i, poly_j):
                dir_ij = spatial_direction(nodes[i].centroid, nodes[j].centroid)

                edges.append(
                    RoomEdge(
                        src=nodes[i].id,
                        dst=nodes[j].id,
                        relation="adjacent",
                        direction=dir_ij,
                    )
                )

    return FloorplanGraph(nodes=nodes, edges=edges)
