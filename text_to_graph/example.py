from pathlib import Path
import os
import sys

import requests
import base64

from dotenv import load_dotenv


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from text_to_graph import FloorplanWorkflow
from graphs.for_resplan.serialize_graph import graph_to_canonical_text


invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = True


load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


def main():
    api_key = (
        os.environ.get("api_key")
        or os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError("Please set api_key in the project root .env file")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream" if stream else "application/json",
    }

    payload = {
        "model": "mistralai/mistral-small-4-119b-2603",
        "reasoning_effort": "high",
        "messages": [{"role": "user", "content": ""}],
        "max_tokens": 16384,
        "temperature": 0.10,
        "top_p": 1.00,
        "stream": stream,
    }

    response = requests.post(invoke_url, headers=headers, json=payload)

    if stream:
        for line in response.iter_lines():
            if line:
                print(line.decode("utf-8"))
    else:
        print(response.json())

    workflow = FloorplanWorkflow(
        api_key=api_key,
        model="mistralai/mistral-small-4-119b-2603",
        base_url="https://integrate.api.nvidia.com/v1",
    )

    user_prompt = input("Enter floorplan prompt: ")

    print("=" * 60)
    print("USER PROMPT:")
    print("=" * 60)
    print(user_prompt)
    print()

    print("=" * 60)
    print("EXPANDED PROMPT (from expander agent):")
    print("=" * 60)
    expanded = workflow.expand_prompt(user_prompt)
    print(f"Original rooms: {expanded.original_rooms}")
    print(f"Added rooms: {expanded.added_rooms}")
    print(f"All rooms: {expanded.all_rooms}")
    print(f"Reasoning: {expanded.reasoning}")
    print()

    print("=" * 60)
    print("EXPLICIT EDGES (from edge extractor agent):")
    print("=" * 60)
    extracted_edges = workflow.extract_explicit_edges(user_prompt, expanded)
    print(f"Extracted edges: {len(extracted_edges.edges)}")
    for edge in extracted_edges.edges:
        print(f"  - {edge.src_type} -> {edge.dst_type} (relation: {edge.relation}, direction: {edge.direction})")
    print()

    serialized_graph, graph = workflow.run_with_text_output(user_prompt)

    print("=" * 60)
    print("SERIALIZED GRAPH:")
    print("=" * 60)
    print(serialized_graph)
    print()

    print("=" * 60)
    print("FLOORPLAN GRAPH OBJECT:")
    print("=" * 60)
    print(f"Nodes: {len(graph.nodes)}")
    for node in graph.nodes:
        print(f"  - {node.id}: {node.type}")
    print(f"Edges: {len(graph.edges)}")
    for edge in graph.edges:
        if edge.direction:
            print(f"  - {edge.src} --[{edge.relation} {edge.direction}]--> {edge.dst}")
        else:
            print(f"  - {edge.src} --[{edge.relation}]--> {edge.dst}")


if __name__ == "__main__":
    main()
