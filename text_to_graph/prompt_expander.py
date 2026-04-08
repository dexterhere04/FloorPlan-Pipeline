from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
import json


class ExpandedPrompt(BaseModel):
    original_rooms: List[str] = Field(
        description="Rooms explicitly mentioned by the user"
    )
    added_rooms: List[str] = Field(description="Essential rooms added by the agent")
    all_rooms: List[str] = Field(description="Complete list of rooms for the floorplan")
    reasoning: str = Field(description="Explanation of why essential rooms were added")
    spatial_context: str | dict = Field(
        description="Additional spatial context from the original prompt"
    )


EXPANDER_SYSTEM = """You are a floorplan design assistant. Your job is to understand what the user wants and expand their initial prompt with essential rooms that a normal residential floorplan requires.

AVAILABLE ROOM TYPES:
- living: Main living area/living room
- dining: Dedicated dining room
- kitchen: Kitchen/cooking area
- bedroom: Bedroom(s)
- bathroom: Bathroom(s)
- balcony: Balcony/terrace
- front_door: Entry/exit point

ESSENTIAL ROOMS FOR A COMPLETE FLOORPLAN:
1. At least one living room (central gathering space)
2. At least one kitchen (cooking area)
3. At least one bedroom (sleeping area)
4. At least one bathroom (sanitary facility)
5. At least one front_door (entry point to the home)

OPTIONAL BUT COMMONLY NEEDED:
- Additional bedrooms (for families, guests)
- Additional bathrooms (ensuite, guest bathroom)
- Balcony (outdoor access)

RULES:
- Always include the front_door - every home needs an entry point
- Always include at least one of each essential room type
- Treat hall/foyer/hallway as synonyms of living (single canonical type: living)
- Do not place a room in added_rooms if it is already present in original_rooms, even if it is phrased differently
- Do not add synonyms or alternate names for rooms the user already asked for
- Infer room sizes and relationships from context
- Consider the flow between spaces (e.g., kitchen near living area, bedrooms separate from living areas)
- Only add rooms that make sense for the described floorplan type and size

IMPORTANT: You MUST respond with ONLY a valid JSON object. No text before or after. Format:
{{"original_rooms": ["room1", "room2"], "added_rooms": ["room3"], "all_rooms": ["room1", "room2", "room3"], "reasoning": "explanation", "spatial_context": "context"}}"""


EXPANDER_USER = """Expand this floorplan description:

{user_prompt}

Return ONLY a valid JSON object with the fields: original_rooms, added_rooms, all_rooms, reasoning, spatial_context."""


def create_prompt_expander(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages(
        [("system", EXPANDER_SYSTEM), ("human", EXPANDER_USER)]
    )

    def normalize_expanded_prompt(expanded: ExpandedPrompt) -> ExpandedPrompt:
        synonym_map = {
            "living room": "living",
            "living_room": "living",
            "lounge": "living",
            "hall": "living",
            "hallway": "living",
            "foyer": "living",
            "entry": "front_door",
            "entryway": "front_door",
            "dining room": "dining",
            "dining_room": "dining",
            "master bedroom": "bedroom",
            "bath": "bathroom",
            "washroom": "bathroom",
        }

        def canonicalize(room: str) -> str:
            normalized = room.strip().lower().replace("-", " ")
            normalized = normalized.replace(" ", "_")
            normalized = synonym_map.get(normalized.replace("_", " "), normalized)
            return normalized

        original_rooms = []
        seen_rooms = set()
        for room in expanded.original_rooms:
            canonical = canonicalize(room)
            if canonical and canonical not in seen_rooms:
                original_rooms.append(canonical)
                seen_rooms.add(canonical)

        added_rooms = []
        for room in expanded.added_rooms:
            canonical = canonicalize(room)
            if canonical and canonical not in seen_rooms:
                added_rooms.append(canonical)
                seen_rooms.add(canonical)

        spatial_context = expanded.spatial_context
        if isinstance(spatial_context, dict):
            spatial_context = json.dumps(spatial_context, ensure_ascii=False)
        else:
            spatial_context = str(spatial_context)

        return expanded.model_copy(
            update={
                "original_rooms": original_rooms,
                "added_rooms": added_rooms,
                "all_rooms": original_rooms + added_rooms,
                "spatial_context": spatial_context,
            }
        )

    return prompt | llm.with_structured_output(ExpandedPrompt) | RunnableLambda(
        normalize_expanded_prompt
    )
