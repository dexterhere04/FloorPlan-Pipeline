from text_to_graph.workflow import FloorplanWorkflow
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY in .env file")

workflow = FloorplanWorkflow(api_key=api_key)

user_prompt = input("Enter floorplan prompt: ")

print("\n" + "=" * 60)
print("USER PROMPT:")
print("=" * 60)
print(user_prompt)

expanded = workflow.expand_prompt(user_prompt)
print("\n" + "=" * 60)
print("EXPANDED PROMPT:")
print("=" * 60)
print(f"Original rooms: {expanded.original_rooms}")
print(f"Added rooms: {expanded.added_rooms}")
print(f"All rooms: {expanded.all_rooms}")
print(f"Reasoning: {expanded.reasoning}")
print(f"Spatial context: {expanded.spatial_context}")

graph = workflow.run(user_prompt)
print("\n" + "=" * 60)
print("FLOORPLAN GRAPH:")
print("=" * 60)
print(f"Nodes ({len(graph.nodes)}):")
for node in graph.nodes:
    print(f"  - {node.id}: {node.type}")
print(f"Edges ({len(graph.edges)}):")
for edge in graph.edges:
    direction = f" {edge.direction}" if edge.direction else ""
    print(f"  - {edge.src} --[{edge.relation}{direction}]--> {edge.dst}")
