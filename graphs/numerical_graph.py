import numpy as np

ROOM_TYPES = ["living","kitchen","bedroom","bathroom","balcony","front_door"]
REL_TYPES  = ["adjacency","via_door","direct"]
DIR_TYPES  = ["left_of","right_of","above","below"]


def onehot(value, vocab):
    vec = np.zeros(len(vocab), dtype=np.float32)
    if value in vocab:
        vec[vocab.index(value)] = 1.0
    return vec


def floorgraph_to_numeric(graph, plan_width):
    """
    Convert FloorplanGraph → numeric graph tensors
    """

    nodes = []
    node_index = {}

    # ---------- Nodes ----------
    for i, n in enumerate(graph.nodes):
        node_index[n.id] = i

        type_vec = onehot(n.type, ROOM_TYPES)

        area = (n.area or 0.0) / (plan_width**2 + 1e-6)

        cx, cy = n.centroid if n.centroid else (0.0, 0.0)
        cx /= plan_width
        cy /= plan_width

        # bbox approx from area
        size = np.sqrt(n.area or 0.0) / plan_width

        feat = np.concatenate([
            type_vec,
            [area, cx, cy, size, size]
        ])

        nodes.append(feat)

    x = np.vstack(nodes).astype(np.float32)

    # ---------- Edges ----------
    edge_index = []
    edge_attr = []

    for e in graph.edges:
        i = node_index[e.src]
        j = node_index[e.dst]

        c1 = graph.nodes[i].centroid
        c2 = graph.nodes[j].centroid

        dx = (c2[0] - c1[0]) / plan_width
        dy = (c2[1] - c1[1]) / plan_width
        dist = np.sqrt(dx*dx + dy*dy)

        rel_vec = onehot(e.relation, REL_TYPES)
        dir_vec = onehot(e.direction, DIR_TYPES)

        feat = np.concatenate([
            rel_vec,
            [dx, dy, dist],
            dir_vec
        ])

        # undirected
        edge_index.append([i, j])
        edge_attr.append(feat)

        edge_index.append([j, i])
        edge_attr.append(feat)

    edge_index = np.array(edge_index).T.astype(np.int64)
    edge_attr = np.vstack(edge_attr).astype(np.float32)

    return x, edge_index, edge_attr