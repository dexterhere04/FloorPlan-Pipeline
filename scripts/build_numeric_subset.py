import pickle
import random
import numpy as np
from tqdm import tqdm

from graphs.for_resplan.resplan_adapter import resplan_to_floorgraph
from graphs.for_resplan.serialize_graph import graph_to_canonical_text
from graphs.numerical_graph import floorgraph_to_numeric
from dataset.resplan_utils import get_plan_width

INPUT = "dataset/ResPlan.pkl"
OUT = "dataset/resplan_numeric_100.pkl"
N     = 100


with open(INPUT, "rb") as f:
    plans = pickle.load(f)

print(f"Loaded {len(plans)} plans")

subset_idx = random.sample(range(len(plans)), N)

nodes_list = []
edge_index_list = []
edge_attr_list = []
text_list = []

for idx in tqdm(subset_idx):
    plan = plans[idx]

    graph = resplan_to_floorgraph(plan)
    width = get_plan_width(plan)

    if width <= 0:
        continue

    x, edge_index, edge_attr = floorgraph_to_numeric(graph, width)
    text = graph_to_canonical_text(graph)

    nodes_list.append(x)
    edge_index_list.append(edge_index)
    edge_attr_list.append(edge_attr)
    text_list.append(text)

print("Collected:", len(nodes_list), "graphs")

# quick sanity checks
print("Example node shape:", nodes_list[0].shape)
print("Example edge_attr shape:", edge_attr_list[0].shape)

data= {
    "nodes": nodes_list,
    "edge_index": edge_index_list,
    "edge_attr": edge_attr_list,
    "text": text_list,
    "plan_idx": subset_idx
}

with open(OUT,"wb") as f:
    pickle.dump(data,f)

print("Saved subset →", OUT)