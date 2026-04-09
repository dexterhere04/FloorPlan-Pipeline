from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class ExplicitEdge(BaseModel):
    src_type: str = Field(
        description="Source room type. Must be one of the expanded room types."
    )
    dst_type: str = Field(
        description="Destination room type. Must be one of the expanded room types."
    )
    relation: str | None = Field(
        default=None,
        description="adjacency, via_door, or direct. Use null when user only said generic connected.",
    )
    direction: str | None = Field(
        default=None,
        description="left_of, right_of, above, or below when explicitly mentioned.",
    )


class ExplicitEdgeExtraction(BaseModel):
    edges: list[ExplicitEdge] = Field(
        description="Only edges explicitly stated in the original user prompt."
    )


EDGE_EXTRACTOR_SYSTEM = """You are an extraction assistant for floorplan prompts. Your ONLY job is to extract room-to-room relationships explicitly stated in the original user prompt.

INSTRUCTIONS:
1. Extract ONLY room-to-room edges explicitly mentioned in the original user prompt
2. Use the expanded room list to normalize room type names to canonical types
3. DO NOT infer any missing edges or complete the graph
4. DO NOT add edges that aren't explicitly stated

ALLOWED ROOM TYPES (canonical names):
living, dining, kitchen, bedroom, bathroom, balcony, front_door

EDGE STRUCTURE (for each extracted edge):
- src_type: source room type (one of the allowed types)
- dst_type: destination room type (one of the allowed types)
- relation: adjacency | via_door | direct | null (use null for "connected to", "attached to", etc.)
- direction: left_of | right_of | above | below | null (for spatial positioning when explicitly stated)

EXAMPLES:
- User says "bedroom connected to kitchen" → {{"src_type": "bedroom", "dst_type": "kitchen", "relation": null, "direction": null}}
- User says "living room is left of kitchen" → {{"src_type": "kitchen", "dst_type": "living", "relation": null, "direction": "left_of"}}
- User says "no relationships mentioned" → edges array is empty

OUTPUT FORMAT (REQUIRED - This is the ONLY thing you output):
{{"edges": [<list of edge objects>]}}

If no explicit relationships are mentioned, return:
{{"edges": []}}

OUTPUT RULES (CRITICAL):
- Return ONLY the JSON object, nothing else
- NO markdown formatting, NO code fences, NO explanation text
- Start with the opening brace and end with closing brace
- Every field must be a valid JSON value
- If you cannot extract a valid edge, return empty list"""


EDGE_EXTRACTOR_USER = """Expanded room types: {expanded_rooms}

Original user prompt: {user_prompt}

Extract only the explicit room-to-room relationships from this prompt.
Return ONLY the JSON object structure, nothing else."""


def create_edge_extractor(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages(
        [("system", EDGE_EXTRACTOR_SYSTEM), ("human", EDGE_EXTRACTOR_USER)]
    )
    return prompt | llm.with_structured_output(ExplicitEdgeExtraction)
