from graphs.schema import RoomNode, RoomEdge, FloorplanGraph
from resplan_utils import plan_to_graph


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
    for u, v, data in G.edges(data=True):
        edges.append(
            RoomEdge(
                src=u,
                dst=v,
                relation=data.get("type", "adjacent"),
            )
        )

    return FloorplanGraph(nodes=nodes, edges=edges)
