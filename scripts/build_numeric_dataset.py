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
    nodes=all_nodes,
    edge_index=all_edge_index,
    edge_attr=all_edges,
    text=all_text
)

print("Saved numeric dataset:", OUT)