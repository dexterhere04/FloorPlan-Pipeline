import pickle
import numpy as np
from tqdm import tqdm

from graphs.for_resplan.resplan_adapter import resplan_to_floorgraph
from graphs.for_resplan.serialize_graph import graph_to_canonical_text
from graphs.numerical_graph import floorgraph_to_numeric
from dataset.resplan_utils import get_plan_width

INPUT = "dataset/ResPlan.pkl"
OUT   = "dataset/resplan_numeric.npz"

with open(INPUT,"rb") as f:
    plans = pickle.load(f)

all_nodes = []
all_edges = []
all_edge_index = []
all_text = []


def to_object_array(items):
    array = np.empty(len(items), dtype=object)
    array[:] = items
    return array

for plan in tqdm(plans):
    graph = resplan_to_floorgraph(plan)
    width = get_plan_width(plan)

    x, edge_index, edge_attr = floorgraph_to_numeric(graph, width)
    text = graph_to_canonical_text(graph)

    all_nodes.append(x)
    all_edge_index.append(edge_index)
    all_edges.append(edge_attr)
    all_text.append(text)

np.savez(
    OUT,
    nodes=to_object_array(all_nodes),
    edge_index=to_object_array(all_edge_index),
    edge_attr=to_object_array(all_edges),
    text=to_object_array(all_text)
)

print("Saved numeric dataset:", OUT)