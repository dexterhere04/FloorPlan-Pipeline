from graphs.for_resplan.serialize_graph import graph_to_canonical_text
import pickle
import random
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from graphs.for_resplan.resplan_adapter import resplan_to_floorgraph
from dataset.resplan_utils import plot_plan

with open("dataset/ResPlan.pkl", "rb") as f:
    plans = pickle.load(f)
idx = random.randrange(len(plans))
graph = resplan_to_floorgraph(plans[idx])
text=graph_to_canonical_text(graph)

print(text)

ax = plot_plan(plans[idx], title=f'Plan #{idx}')
plt.show()
