def graph_to_text(graph):
    lines = []

    for n in graph.nodes:
        lines.append(f"Room {n.id}: {n.type}")

    for e in graph.edges:
        if e.direction:
            lines.append(
                f"{e.src} {e.relation} {e.dst} {e.direction}"
            )
        else:
            lines.append(
                f"{e.src} {e.relation} {e.dst}"
            )

    return "\n".join(lines)
