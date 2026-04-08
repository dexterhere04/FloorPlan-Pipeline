from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from text_to_graph.prompt_expander import create_prompt_expander, ExpandedPrompt
from text_to_graph.graph_generator import create_graph_generator, parse_graph_output
from text_to_graph.schemas import FloorplanGraphSchema
from graphs.schema import RoomNode, RoomEdge, FloorplanGraph


class FloorplanWorkflow:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
    ):
        self.llm = ChatOpenAI(api_key=api_key, model=model, base_url=base_url)
        self.expander = create_prompt_expander(self.llm)
        self.generator = create_graph_generator(self.llm)

    def expand_prompt(self, user_prompt: str) -> ExpandedPrompt:
        return self.expander.invoke({"user_prompt": user_prompt})

    def generate_graph(
        self, expanded_prompt: ExpandedPrompt, original_prompt: str
    ) -> FloorplanGraphSchema:
        result = self.generator.invoke(
            {
                "expanded_prompt": expanded_prompt.all_rooms,
                "original_prompt": original_prompt,
            }
        )

        if isinstance(result, dict):
            if "raw" in result:
                return parse_graph_output(result["raw"])
            if "parsed" in result:
                return result["parsed"]

        raw_text = getattr(result, "content", result)
        return parse_graph_output(raw_text)

    def run(self, user_prompt: str) -> FloorplanGraph:
        expanded = self.expand_prompt(user_prompt)
        graph_schema = self.generate_graph(expanded, user_prompt)

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
        graph_schema = self.generate_graph(expanded, user_prompt)

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
