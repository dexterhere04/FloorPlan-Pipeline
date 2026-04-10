from langchain_openai import ChatOpenAI
from pathlib import Path

from text_to_graph.prompt_expander import create_prompt_expander, ExpandedPrompt
from text_to_graph.graph_generator import create_graph_generator, parse_graph_output
from text_to_graph.schemas import FloorplanGraphSchema
from text_to_graph.edge_extractor import (
    create_edge_extractor,
    ExplicitEdge,
    ExplicitEdgeExtraction,
)
from text_to_graph.edge_completion import EdgeCompletionPredictor
from text_to_graph.edge_completion import CompletionConfig
from graphs.schema import RoomNode, RoomEdge, FloorplanGraph


def _matches_extracted_edge(
    edge: RoomEdge,
    source_type: str,
    destination_type: str,
    extracted_edge: ExplicitEdge,
) -> bool:
    relation_match = (
        extracted_edge.relation is None or edge.relation == extracted_edge.relation
    )
    direction_match = (
        extracted_edge.direction is None or edge.direction == extracted_edge.direction
    )

    if not relation_match or not direction_match:
        return False

    if extracted_edge.direction is not None:
        return (
            source_type == extracted_edge.src_type
            and destination_type == extracted_edge.dst_type
        )

    return (
        (source_type == extracted_edge.src_type and destination_type == extracted_edge.dst_type)
        or (source_type == extracted_edge.dst_type and destination_type == extracted_edge.src_type)
    )


def _filter_edges_by_extracted(
    graph_schema: FloorplanGraphSchema, extracted_edges: list[ExplicitEdge]
) -> FloorplanGraphSchema:
    if not extracted_edges:
        return graph_schema.model_copy(update={"edges": []})

    room_type_by_id = {room.id: room.type for room in graph_schema.rooms}
    filtered_edges = []
    for edge in graph_schema.edges:
        source_type = room_type_by_id.get(edge.src)
        destination_type = room_type_by_id.get(edge.dst)
        if not source_type or not destination_type:
            continue

        if any(
            _matches_extracted_edge(edge, source_type, destination_type, extracted_edge)
            for extracted_edge in extracted_edges
        ):
            filtered_edges.append(edge)

    return graph_schema.model_copy(update={"edges": filtered_edges})


class FloorplanWorkflow:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
        edge_completion_model_path: str | None = None,
        edge_completion_threshold: float = 0.45,
        edge_completion_max_added_edges: int = 24,
    ):
        self.llm = ChatOpenAI(api_key=api_key, model=model, base_url=base_url)
        self.expander = create_prompt_expander(self.llm)
        self.edge_extractor = create_edge_extractor(self.llm)
        self.generator = create_graph_generator(self.llm)

        self.edge_completer = None
        default_path = Path("trained_models/best_edge_completion.pt")
        model_path = Path(edge_completion_model_path) if edge_completion_model_path else default_path
        if model_path.exists():
            try:
                self.edge_completer = EdgeCompletionPredictor(
                    model_path,
                    config=CompletionConfig(
                        threshold=edge_completion_threshold,
                        max_added_edges=edge_completion_max_added_edges,
                    ),
                )
            except Exception:
                self.edge_completer = None

    def expand_prompt(self, user_prompt: str) -> ExpandedPrompt:
        return self.expander.invoke({"user_prompt": user_prompt})

    def extract_explicit_edges(
        self, user_prompt: str, expanded_prompt: ExpandedPrompt
    ) -> ExplicitEdgeExtraction:
        return self.edge_extractor.invoke(
            {
                "expanded_rooms": expanded_prompt.all_rooms,
                "user_prompt": user_prompt,
            }
        )

    def generate_graph(
        self,
        expanded_prompt: ExpandedPrompt,
        original_prompt: str,
        extracted_edges: ExplicitEdgeExtraction,
    ) -> FloorplanGraphSchema:
        result = self.generator.invoke(
            {
                "expanded_prompt": ", ".join(expanded_prompt.all_rooms),
                "original_prompt": original_prompt,
            }
        )

        if isinstance(result, dict):
            if "raw" in result:
                graph_schema = parse_graph_output(result["raw"])
                filtered = _filter_edges_by_extracted(graph_schema, extracted_edges.edges)
                if self.edge_completer is not None:
                    return self.edge_completer.complete_graph(filtered)
                return filtered
            if "parsed" in result:
                filtered = _filter_edges_by_extracted(result["parsed"], extracted_edges.edges)
                if self.edge_completer is not None:
                    return self.edge_completer.complete_graph(filtered)
                return filtered

        raw_text = getattr(result, "content", result)
        graph_schema = parse_graph_output(raw_text)
        filtered = _filter_edges_by_extracted(graph_schema, extracted_edges.edges)
        if self.edge_completer is not None:
            return self.edge_completer.complete_graph(filtered)
        return filtered

    def run(self, user_prompt: str) -> FloorplanGraph:
        expanded = self.expand_prompt(user_prompt)
        extracted_edges = self.extract_explicit_edges(user_prompt, expanded)
        graph_schema = self.generate_graph(expanded, user_prompt, extracted_edges)

        nodes = [
            RoomNode(id=room.id, type=room.type, role=room.role)
            for room in graph_schema.rooms
        ]

        edges = [
            RoomEdge(
                src=edge.src,
                dst=edge.dst,
                relation=edge.relation,
                direction=edge.direction,
            )
            for edge in graph_schema.edges
        ]

        return FloorplanGraph(nodes=nodes, edges=edges)

    def run_with_text_output(self, user_prompt: str) -> tuple[str, FloorplanGraph]:
        expanded = self.expand_prompt(user_prompt)
        extracted_edges = self.extract_explicit_edges(user_prompt, expanded)
        graph_schema = self.generate_graph(expanded, user_prompt, extracted_edges)

        serialized_lines = []
        for room in sorted(graph_schema.rooms, key=lambda x: x.id):
            serialized_lines.append(f"{room.id}: {room.type}")

        for edge in graph_schema.edges:
            if edge.direction:
                serialized_lines.append(
                    f"{edge.src} {edge.relation} {edge.direction} {edge.dst}"
                )
            else:
                serialized_lines.append(f"{edge.src} {edge.relation} {edge.dst}")

        serialized_text = "\n".join(serialized_lines)

        nodes = [
            RoomNode(id=room.id, type=room.type, role=room.role)
            for room in graph_schema.rooms
        ]

        edges = [
            RoomEdge(
                src=edge.src,
                dst=edge.dst,
                relation=edge.relation,
                direction=edge.direction,
            )
            for edge in graph_schema.edges
        ]

        return serialized_text, FloorplanGraph(nodes=nodes, edges=edges)
