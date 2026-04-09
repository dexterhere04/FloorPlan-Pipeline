# model_testing.ipynb - Complete Documentation

This notebook demonstrates the complete pipeline for testing and training a Graph Neural Network (GNN) model for floorplan layout prediction. The model takes room graphs (nodes and edges) and predicts spatial layout coordinates (centroid and bounding box) for each room.

---

## Overview

The notebook performs the following tasks:
1. **Data Loading** - Load numeric floorplan data (nodes, edges, text)
2. **Visualization** - Visualize floorplan graphs and compare with actual floorplans
3. **Dataset Construction** - Convert data to PyTorch Geometric format
4. **Model Definition** - Define a GNN architecture (RoughLayoutGNN)
5. **Training** - Train the model with various loss functions
6. **Evaluation** - Evaluate and visualize predictions

---

## Section 1: Data Loading

```python
import pickle

with open("dataset/resplan_numeric_100.pkl", "rb") as f:
    data = pickle.load(f)

nodes = data["nodes"]       # Node features (num_samples × num_nodes × 11)
edge_index = data["edge_index"]  # Edge connectivity
edge_attr = data["edge_attr"]    # Edge features
text = data["text"]          # Serialized graph text
```

**Data Structure:**
- `nodes`: List of numpy arrays, each shape `(num_rooms, 11)` containing:
  - Room type (one-hot, 6 dims): living, kitchen, bedroom, bathroom, balcony, front_door
  - Area (normalized)
  - Centroid (cx, cy) normalized
  - Bounding box (w, h) normalized
- `edge_index`: List of arrays, shape `(2, num_edges)`
- `edge_attr`: Edge features (10 dims): relation type, dx, dy, distance, direction

---

## Section 2: Graph Visualization

### 2.1 Simple NetworkX Graph

Creates a basic visualization of the room graph structure:

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
# Add nodes with positions from centroid features (columns 7, 8)
cx = x[:, 7]  # centroid x
cy = x[:, 8]  # centroid y
for n in range(len(x)):
    G.add_node(n, pos=(cx[n], cy[n]))
# Add edges
for s, d in edge_index_1.T:
    G.add_edge(int(s), int(d))
# Draw
pos = nx.get_node_attributes(G, "pos")
nx.draw(G, pos, node_size=300)
```

**Output:** NetworkX graph showing room connectivity with node positions from centroids.

### 2.2 Floorplan Visualization with Graph Overlay

Overlays the graph on the actual floorplan image:

```python
from dataset.resplan_utils import plot_plan

# Load original ResPlan data
with open("dataset/ResPlan.pkl", "rb") as f:
    plans = pickle.load(f)

plan = plans[orig_idx]
plot_plan(plan, title=f"ResPlan #{orig_idx}")
```

**Key steps:**
1. Gets original floorplan bounds from matplotlib axis
2. Maps normalized node coordinates (0-1) to floorplan axis coordinates
3. Overlays graph edges (red) and nodes (yellow) on floorplan
4. Sets z-order to ensure graph appears above floorplan

---

## Section 3: Dataset Construction

Converts numpy data to PyTorch Geometric `Data` objects:

```python
import torch
from torch_geometric.data import Data

def build_dataset(nodes, edge_index, edge_attr):
    dataset = []
    for i in range(len(nodes)):
        x = torch.tensor(nodes[i], dtype=torch.float32)
        ei = torch.tensor(edge_index[i], dtype=torch.long)
        ea = torch.tensor(edge_attr[i], dtype=torch.float32)
        
        # Target: cx, cy, w, h (columns 7-10)
        y = x[:, 7:11]
        
        data = Data(x=x, edge_index=ei, edge_attr=ea, y=y)
        dataset.append(data)
    return dataset
```

**Features used:**
- Input (x): All 11 node features
- Target (y): Spatial features only (cx, cy, w, h)

---

## Section 4: Train/Test Split

```python
train_dataset = dataset[:80]  # 80 samples
test_dataset = dataset[80:]    # 20 samples
```

**DataLoaders:**

```python
from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)
```

---

## Section 5: Model Definition - RoughLayoutGNN

A 3-layer Graph Isomorphism Network (GINE) for layout prediction:

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv

class RoughLayoutGNN(nn.Module):
    def __init__(self, node_dim=11, edge_dim=10, hidden=128):
        super().__init__()
        
        def mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim)
            )
        
        # Three GINE convolution layers
        self.conv1 = GINEConv(mlp(node_dim, hidden), edge_dim=edge_dim)
        self.conv2 = GINEConv(mlp(hidden, hidden), edge_dim=edge_dim)
        self.conv3 = GINEConv(mlp(hidden, hidden), edge_dim=edge_dim)
        
        # Prediction head: outputs 4 values (cx, cy, w, h)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4)  # cx, cy, w, h
        )
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        
        x = self.conv3(x, edge_index, edge_attr)
        x = torch.relu(x)
        
        out = self.head(x)
        return out
```

**Architecture:**
- **GINEConv**: Graph Isomorphism convolution that incorporates edge features
- **3 layers**: Progressive message passing through the graph
- **Hidden dimension**: 128
- **Output**: 4 values per node (centroid x, centroid y, width, height)

---

## Section 6: Training Setup

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RoughLayoutGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

---

## Section 7: Loss Functions

### 7.1 Basic L1 Loss

```python
def layout_loss(pred, target):
    return torch.nn.functional.l1_loss(pred, target)
```

### 7.2 Advanced Loss Functions

The notebook implements several geometric-aware loss functions:

#### Graph Geometry Loss
Encourages predicted edge distances and angles to match ground truth:

```python
def graph_geom_loss(pred, target, edge_index):
    cx_p, cy_p = pred[:, 0], pred[:, 1]
    cx_t, cy_t = target[:, 0], target[:, 1]
    
    loss = 0.0
    E = edge_index.shape[1]
    
    for k in range(E):
        i = edge_index[0, k]
        j = edge_index[1, k]
        
        # Distance loss
        dxp = cx_p[i] - cx_p[j]
        dyp = cy_p[i] - cy_p[j]
        dxt = cx_t[i] - cx_t[j]
        dyt = cy_t[i] - cy_t[j]
        
        dp = torch.sqrt(dxp**2 + dyp**2 + 1e-6)
        dt = torch.sqrt(dxt**2 + dyt**2 + 1e-6)
        ld = torch.abs(dp - dt)
        
        # Angle loss
        ang_p = torch.atan2(dyp, dxp)
        ang_t = torch.atan2(dyt, dxt)
        la = torch.abs(torch.sin((ang_p - ang_t) / 2))
        
        loss += ld + la
    
    return loss / E
```

#### Box Overlap Loss
Penalizes overlapping predicted boxes:

```python
def box_overlap(cx, cy, w, h):
    N = cx.shape[0]
    overlap = 0.0
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = torch.abs(cx[i] - cx[j])
            dy = torch.abs(cy[i] - cy[j])
            
            ox = torch.relu((w[i] + w[j]) / 2 - dx)
            oy = torch.relu((h[i] + h[j]) / 2 - dy)
            
            overlap += ox * oy
    
    return overlap / (N * (N - 1) / 2 + 1e-6)
```

#### Disconnect Loss
Ensures connected rooms have small gaps:

```python
def disconnect_loss(pred, edge_index):
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    
    loss = 0.0
    E = edge_index.shape[1]
    
    for k in range(E):
        i = edge_index[0, k]
        j = edge_index[1, k]
        
        dx = cx[i] - cx[j]
        dy = cy[i] - cy[j]
        dist = torch.sqrt(dx * dx + dy * dy + 1e-6)
        
        min_touch = (w[i] + w[j]) / 2
        gap = torch.relu(dist - min_touch)
        
        loss += gap
    
    return loss / E
```

#### Combined Layout Loss

```python
def layout_loss(pred, target, edge_index,
                l_geom=1.0, l_overlap=2.0, l_disc=1.0):
    
    l1 = torch.nn.functional.l1_loss(pred, target)
    
    g = graph_geom_loss(pred, target, edge_index)
    o = scaled_overlap_loss(pred)
    d = disconnect_loss(pred, edge_index)
    
    return l1 + l_geom * g + l_overlap * o + l_disc * d
```

### 7.3 Alternative Loss Functions (Second Approach)

#### Overlap Loss
Direct overlap calculation between boxes:

```python
def overlap_loss(boxes):
    cx, cy, w, h = boxes.T
    
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    
    loss = 0.0
    N = boxes.shape[0]
    
    for i in range(N):
        for j in range(i + 1, N):
            ix1 = torch.max(x1[i], x1[j])
            iy1 = torch.max(y1[i], y1[j])
            ix2 = torch.min(x2[i], x2[j])
            iy2 = torch.min(y2[i], y2[j])
            
            iw = torch.clamp(ix2 - ix1, min=0)
            ih = torch.clamp(iy2 - iy1, min=0)
            
            inter = iw * ih
            loss += inter
    
    return loss / (N * N + 1e-6)
```

#### Adjacency Loss
Encourages connected rooms to be close:

```python
def adjacency_loss(boxes, edge_index):
    cx, cy, w, h = boxes.T
    
    loss = 0.0
    M = edge_index.shape[1]
    
    for k in range(M):
        i = edge_index[0, k]
        j = edge_index[1, k]
        
        dx = torch.abs(cx[i] - cx[j])
        dy = torch.abs(cy[i] - cy[j])
        
        tx = (w[i] + w[j]) / 2
        ty = (h[i] + h[j]) / 2
        
        dist = torch.relu(dx - tx) + torch.relu(dy - ty)
        loss += dist
    
    return loss / (M + 1e-6)
```

---

## Section 8: Training Loop

### 8.1 Basic Training (50 epochs)

```python
def train_epoch():
    model.train()
    total_loss = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = layout_loss(pred, data.y, data.edge_index)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def eval_epoch():
    model.eval()
    total_loss = 0
    
    for data in test_loader:
        data = data.to(device)
        pred = model(data)
        loss = layout_loss(pred, data.y, data.edge_index)
        total_loss += loss.item()
    
    return total_loss / len(test_loader)

for epoch in range(50):
    train_loss = train_epoch()
    test_loss = eval_epoch()
    print(f"Epoch {epoch:03d} | train {train_loss:.4f} | test {test_loss:.4f}")
```

### 8.2 Training with Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4, path="best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.stop = False
        self.path = path
    
    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

early_stopper = EarlyStopping()

for epoch in range(200):
    tr = train_epoch()
    te = eval_epoch()
    print(f"{epoch:03d} | train {tr:.4f} | test {te:.4f}")
    early_stopper.step(te, model)
    if early_stopper.stop:
        print(f"\nEarly stopping triggered at epoch {epoch}")
        break

# Restore best model
model.load_state_dict(torch.load("best_model.pt"))
```

---

## Section 9: Visualization Functions

### 9.1 Simple Prediction Visualization

```python
@torch.no_grad()
def visualize_prediction(sample_idx=0):
    data = test_dataset[sample_idx].to(device)
    pred = model(data).cpu().numpy()
    gt = data.y.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Draw ground truth (green) and predicted (red) boxes
    for box, color in [(gt, "green"), (pred, "red")]:
        for cx, cy, w, h in box:
            x0 = cx - w / 2
            y0 = cy - h / 2
            rect = plt.Rectangle(
                (x0, y0), w, h,
                fill=False,
                edgecolor=color,
                linewidth=2
            )
            ax.add_patch(rect)
    
    # Auto-fit canvas
    ax.set_aspect("equal")
    plt.show()
```

### 9.2 Sample with Floorplan Overlay

```python
@torch.no_grad()
def visualize_sample(i, model, dataset, plans, plan_idx):
    model.eval()
    
    data = dataset[i].to(next(model.parameters()).device)
    pred = model(data).cpu().numpy()
    gt = data.y.cpu().numpy()
    
    # Get original floorplan
    orig = plan_idx[i]
    plan = plans[orig]
    plot_plan(plan, title=f"Dataset {i} → ResPlan {orig}")
    ax = plt.gca()
    
    # Get floorplan bounds
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    
    # Create graph
    x = data.x.cpu().numpy()
    cx = x[:, 7]
    cy = x[:, 8]
    
    G = nx.Graph()
    for n in range(len(x)):
        G.add_node(n, pos=(cx[n], cy[n]))
    
    ei = data.edge_index.cpu().numpy()
    for s, d in ei.T:
        G.add_edge(int(s), int(d))
    
    # Map to floorplan coordinates
    pos = nx.get_node_attributes(G, "pos")
    pos_plan = {
        k: (
            xmin + float(v[0]) * (xmax - xmin),
            ymin + float(v[1]) * (ymax - ymin),
        )
        for k, v in pos.items()
    }
    
    # Draw graph
    nx.draw_networkx_edges(G, pos_plan, ax=ax, edge_color="red", width=2)
    nx.draw_networkx_nodes(
        G, pos_plan, ax=ax,
        node_color="yellow", edgecolors="black", node_size=120
    )
    
    # Draw boxes
    def draw_boxes(boxes, color):
        for cx, cy, w, h in boxes:
            x0 = xmin + (cx - w/2) * (xmax - xmin)
            y0 = ymin + (cy - h/2) * (ymax - ymin)
            ww = w * (xmax - xmin)
            hh = h * (ymax - ymin)
            
            rect = plt.Rectangle(
                (x0, y0), ww, hh,
                fill=False,
                edgecolor=color,
                linewidth=2
            )
            ax.add_patch(rect)
    
    draw_boxes(gt, "green")   # Ground truth
    draw_boxes(pred, "blue")  # Predicted
    
    plt.tight_layout()
    plt.show()
```

### 9.3 Predicted Boxes Only

```python
@torch.no_grad()
def visualize_pred_boxes(i, model, dataset):
    model.eval()
    
    device = next(model.parameters()).device
    data = dataset[i].to(device)
    pred = model(data).cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    for cx, cy, w, h in pred:
        x0 = cx - w / 2
        y0 = cy - h / 2
        
        rect = plt.Rectangle(
            (x0, y0), w, h,
            fill=False,
            edgecolor="blue",
            linewidth=2
        )
        ax.add_patch(rect)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    plt.title(f"Predicted layout — sample {i}")
    plt.show()
```

---

## Section 10: Model Export

Exports trained model weights to a pickle file:

```python
model_was_training = model.training
model_cpu = model.to("cpu")
model_payload = {
    "model_class": "RoughLayoutGNN",
    "state_dict": model_cpu.state_dict(),
}

with open("trained_models/rough_layout_gnn.pkl", "wb") as f:
    pickle.dump(model_payload, f)

model.to(device)
```

---

## Summary of Key Components

| Component | Description |
|-----------|-------------|
| **Input** | Room graphs with node features (type, area, centroid, bbox) and edge features |
| **Output** | Predicted layout coordinates (cx, cy, w, h) for each room |
| **Model** | 3-layer GINE (Graph Isomorphism Network with Edge features) |
| **Loss Functions** | L1 loss, geometry loss, overlap loss, disconnect loss, adjacency loss |
| **Training** | Adam optimizer, early stopping, batch processing |
| **Visualization** | Floorplan overlay, box comparison (GT vs predicted) |

---

## Data Flow

```
ResPlan.pkl
    ↓
resplan_numeric_100.pkl (preprocessed)
    ↓
build_dataset() → PyTorch Geometric Data
    ↓
Train/Test Split
    ↓
RoughLayoutGNN (3-layer GINE)
    ↓
Predicted Layout (cx, cy, w, h per room)
    ↓
Visualization + Evaluation
```
