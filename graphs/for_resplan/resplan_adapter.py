from graphs.schema import RoomNode, RoomEdge, FloorplanGraph
from dataset.resplan_utils import plan_to_graph


def resplan_to_floorgraph(plan):
    """
    Convert ResPlan plan dict → our FloorplanGraph schema
    """

    G = plan_to_graph(plan)

    nodes = []
    edges = []

    # --- nodes ---
    for nid, data in G.nodes(data=True):
        geom = data.get("geometry")
        if geom is None or geom.is_empty:
            continue

        c = geom.centroid

        nodes.append(
            RoomNode(
                id=nid,
                type=data.get("type"),
                area=data.get("area"),
                centroid=(c.x, c.y),
            )
        )

    # --- edges ---
    node_centroids = {n.id: n.centroid for n in nodes}

    for u, v, data in G.edges(data=True):
        c1 = node_centroids.get(u)
        c2 = node_centroids.get(v)
        direction = None
        if c1 and c2:
            dx = c2[0] - c1[0]
            dy = c2[1] - c1[1]

            if abs(dx) > abs(dy):
                direction = "right_of" if dx > 0 else "left_of"
            else:
                direction = "below" if dy > 0 else "above"
        edges.append(
            RoomEdge(
                src=u,
                dst=v,
                relation=data.get("type", "adjacent"),
                direction=direction
            )
        )

    return FloorplanGraph(nodes=nodes, edges=edges)
