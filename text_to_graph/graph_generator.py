from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from text_to_graph.schemas import FloorplanGraphSchema


GRAPH_GENERATOR_SYSTEM = """You are a floorplan graph generation assistant. Your job is to convert a floorplan description into a structured graph representation.

AVAILABLE ROOM TYPES:
- living: Main living area/living room (hall/foyer/hallway map to living)
- dining: Dedicated dining room
- kitchen: Kitchen/cooking area
- bedroom: Bedroom(s)
- bathroom: Bathroom(s)
- balcony: Balcony/terrace
- front_door: Entry/exit point

RELATION TYPES:
- adjacency: Rooms share a wall or are directly next to each other
- via_door: Rooms are connected through a door
- direct: Rooms have a direct spatial relationship (no wall between them)

DIRECTION TYPES (optional):
- left_of: The destination room is to the left of the source room
- right_of: The destination room is to the right of the source room
- above: The destination room is above the source room
- below: The destination room is below the source room

GRAPH SERIALIZATION FORMAT:
The output must follow this canonical format:
1. First list all room nodes sorted by ID:
   {{room_id}}: {{room_type}}
2. Then list all edges:
   {{src}} {{relation}} {{direction}} {{dst}}   (if direction exists)
   {{src}} {{relation}} {{dst}}               (if no direction)

EXAMPLE OUTPUT:
living_0: living
kitchen_0: kitchen
bedroom_0: bedroom
bathroom_0: bathroom
front_door_0: front_door
living_0 adjacency below kitchen_0
kitchen_0 via_door left_of living_0
bedroom_0 via_door above bathroom_0
front_door_0 direct right_of living_0

IMPORTANT RULES:
- Include ALL room types from the expanded list (nodes should represent all expanded rooms)
- Use unique IDs for each room (e.g., living_0, kitchen_0, bedroom_0, bedroom_1)
- Only use the exact room types listed above (lowercase, with underscores)
- Only use the exact relation types listed above
- Only use the exact direction types listed above when applicable
- Create edges only when the original user prompt explicitly states a relationship
- Do not infer, invent, or complete any missing relationships
- If the original prompt does not explicitly mention any room-to-room relationships, output no edges
- Sort nodes alphabetically by ID for deterministic output
- The output MUST be parseable as a structured graph

Generate only the explicitly mentioned relationships from the original prompt, but include all room types in the expanded list as nodes."""


GRAPH_GENERATOR_USER = """Expanded floorplan rooms (ALL of these must appear as nodes):
{expanded_prompt}

Original user request (use this to identify explicit relationships only):
{original_prompt}

TASK:
1. Include ALL rooms listed in the expanded floorplan rooms as nodes (one node per room type; e.g., if 'bedroom' appears once, create bedroom_0; if mentioned twice, create bedroom_0 and bedroom_1)
2. Extract ONLY the explicit room-to-room relationships mentioned in the original user request
3. Do NOT infer or create any additional relationships
4. Output in the canonical format"""


def create_graph_generator(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages(
        [("system", GRAPH_GENERATOR_SYSTEM), ("human", GRAPH_GENERATOR_USER)]
    )
    return prompt | llm


def parse_graph_output(raw_output: str) -> FloorplanGraphSchema:
    """Parse the canonical text format back into a FloorplanGraphSchema."""
    from text_to_graph.schemas import FloorplanRoom, FloorplanEdge

    rooms = []
    edges = []

    lines = raw_output.strip().split("\n")

    for line in lines:
        parts = line.split()
        if len(parts) == 2 and ":" in line:
            room_id = parts[0].rstrip(":")
            room_type = parts[1]
            rooms.append(FloorplanRoom(id=room_id, type=room_type))
        elif len(parts) >= 3:
            src = parts[0]
            relation = parts[1]

            if len(parts) == 4:
                direction = parts[2]
                dst = parts[3]
            else:
                direction = None
                dst = parts[2]

            edges.append(
                FloorplanEdge(src=src, dst=dst, relation=relation, direction=direction)
            )

    return FloorplanGraphSchema(rooms=rooms, edges=edges)
