def graph_to_canonical_text(graph):
    """
    Deterministic graph serialization.
    Used as ground truth target for LLM training.
    """

    lines = []

    # Nodes
    for n in sorted(graph.nodes, key=lambda x: x.id):
        lines.append(f"{n.id}: {n.type}")

    # Edges
    for e in graph.edges:
        lines.append(f"{e.src} {e.relation} {e.dst}")

    return "\n".join(lines)