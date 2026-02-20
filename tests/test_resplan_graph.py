import pickle
from graphs.for_resplan.resplan_adapter import resplan_to_floorgraph
import random
import matplotlib.pyplot as plt
from dataset.resplan_utils import plot_plan
# load one ResPlan sample
with open("dataset/ResPlan.pkl", "rb") as f:
    plans = pickle.load(f)

print(f'Loaded {len(plans)} plans')
idx = random.randrange(len(plans))

plan=plans[idx]

graph = resplan_to_floorgraph(plan)

ax = plot_plan(plan, title=f'Plan #{idx}')
plt.show()

print("Nodes:")
for n in graph.nodes:
    print(n)

print("\nEdges:")
for e in graph.edges:
    print(e)
