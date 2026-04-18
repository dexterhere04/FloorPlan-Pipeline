# FloorPlan-Pipeline

> Generate structured floorplans from plain-English prompts using LLMs + Graph Neural Networks.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)
![License](https://img.shields.io/badge/License-GPLv3-blue)

An end-to-end pipeline that converts natural language descriptions into floorplan graphs and predicted room layouts.

## 📌 At a Glance

- Turn prompts like _"2BHK with open kitchen and balcony"_ into a structured graph
- Convert graphs into GNN-ready tensors
- Predict room boxes (`cx, cy, w, h`) with a trained model
- Visualize outputs in notebooks

## 📚 Table of Contents

- [🎯 Overview](#-overview)
- [🏗️ Architecture](#️-architecture)
- [🚀 Getting Started](#-getting-started)
- [📊 Project Structure](#-project-structure)
- [🔑 Key Workflows](#-key-workflows)
- [🧪 Testing](#-testing)
- [📚 Documentation](#-documentation)
- [🔧 Configuration](#-configuration)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🎯 Overview

FloorPlan-Pipeline is a comprehensive system for automated floorplan generation from text descriptions. The pipeline:

1. **Expands user prompts** into complete room specifications using LLMs
2. **Generates topological graphs** representing room relationships and adjacencies
3. **Converts graphs to numerical tensors** for machine learning processing
4. **Predicts spatial layouts** using Graph Neural Networks (GNN)
5. **Visualizes floorplans** with room boxes and relationships

### Use Cases

- Automated architectural planning from text descriptions
- Real estate floorplan generation from written specifications
- Interior design space planning
- Building information modeling (BIM) assistance
- Layout optimization and constraint satisfaction

## 🏗️ Architecture

### Workflow Pipeline

```
Natural Language Input
        ↓
    Prompt Expansion (LLM)
        ↓
    Graph Generation (LLM)
        ↓
    FloorplanGraph (Canonical)
        ↓
    Numerical Tensors (GNN-ready)
        ↓
    RoughLayoutGNN (PyTorch)
        ↓
    Predicted Spatial Layout
        ↓
    Visualization & Output
```

### Key Components

#### Text-to-Graph Pipeline (`text_to_graph/`)
- **`workflow.py`**: Orchestrates prompt expansion, graph generation, and parsing
- **`prompt_expander.py`**: Expands user prompts into complete room descriptions with spatial context
- **`graph_generator.py`**: Converts expanded prompts into structured room graphs with relationships
- **`schemas.py`**: Pydantic models for LLM exchange format and validation

#### Graph Processing (`graphs/`)
- **`schema.py`**: Core data classes (`RoomNode`, `RoomEdge`, `FloorplanGraph`)
- **`numerical_graph.py`**: Converts graph objects to numeric tensors for GNN input
- **`for_resplan/`**: ResPlan dataset integration and graph serialization
  - `serialize_graph.py`: Deterministic canonical text serialization
  - `resplan_adapter.py`: ResPlan dataset adapter
- **`for_rplan/`**: RPlan dataset integration (reference/legacy)

#### Model Training & Inference
- **`complete_workflow.ipynb`**: End-to-end Jupyter notebook demonstrating the full pipeline
- **`model_notebook_training/model_testing.ipynb`**: GNN model training, evaluation, and visualization
- **`best_model.pt`**: Pre-trained checkpoint for layout prediction

#### Dataset & Utilities (`dataset/`)
- **`resplan_utils.py`**: Utilities for ResPlan dataset visualization and processing
- **`__init__.py`**: Package initialization

### Model Architecture

The **RoughLayoutGNN** predicts 4D bounding boxes (center_x, center_y, width, height) for each room:

```
Graph Input (Node & Edge Features)
    ↓
GINEConv Layer 1 (node_dim → hidden)
    ↓
GINEConv Layer 2 (hidden → hidden)
    ↓
GINEConv Layer 3 (hidden → hidden)
    ↓
MLP Head (hidden → 4D box output)
    ↓
Room Coordinates [cx, cy, w, h]
```

**Input Features:**
- **Node features** (11D): Room type (one-hot encoded across 6 room types), role
- **Edge features** (10D): Relationship type, spatial direction, adjacency information

**Output:** 4D bounding box for each room (normalized 0-1 range)

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ (tested with Python 3.11+)
- pip or conda for package management
- An LLM API key (OpenAI, NVIDIA, or compatible API)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd FloorPlan-Pipeline
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional ML dependencies** (if needed for your environment)
   ```bash
   pip install torch torchvision torchaudio
   pip install torch_geometric
   pip install langchain langchain-openai
   ```

5. **Set up environment variables**
   ```bash
   # Create a .env file in the repo root and add:
   # - OPENAI_API_KEY or NVIDIA_API_KEY
   # - FLOORPLAN_LLM_MODEL (optional, defaults to Mistral)
   # - OPENAI_BASE_URL (optional, for custom endpoints)
   ```

### Quick Start

#### 1. Using the Complete Workflow Notebook

```bash
jupyter notebook complete_workflow.ipynb
```

This notebook demonstrates the full pipeline:
- Prompt expansion
- Graph generation
- Tensor conversion
- Layout prediction
- Visualization

#### 2. Using the Python API

```python
from text_to_graph.workflow import FloorplanWorkflow
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize workflow
workflow = FloorplanWorkflow(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",
    base_url=os.getenv("OPENAI_BASE_URL")
)

# Expand a natural language prompt
prompt = "A modern 2BHK apartment with a living room, kitchen, two bedrooms, and a bathroom"
expanded = workflow.expand_prompt(prompt)

# Generate graph
graph_schema = workflow.generate_graph(expanded, prompt)

# Convert to canonical floorplan graph
from graphs.schema import RoomNode, RoomEdge, FloorplanGraph
from graphs.for_resplan.serialize_graph import graph_to_canonical_text

graph = FloorplanGraph(
    nodes=[RoomNode(id=room.id, type=room.type, role=room.role) for room in graph_schema.rooms],
    edges=[RoomEdge(src=edge.src, dst=edge.dst, relation=edge.relation, direction=edge.direction) 
           for edge in graph_schema.edges]
)

# Serialize to text
serialized = graph_to_canonical_text(graph)
print(serialized)
```

#### 2.5 Minimal sanity check

```bash
python -m pytest tests/ -q
```

#### 3. Using the Interactive Example Script

```bash
cd text_to_graph
python example.py
```

## 📊 Project Structure

```
FloorPlan-Pipeline/
├── README.md                              # This file
├── LICENSE                                # GNU General Public License v3
├── requirements.txt                       # Python dependencies
├── best_model.pt                          # Pre-trained layout prediction model
├── .env                                   # Environment configuration (API keys)
├── .env.example                           # Environment template
│
├── complete_workflow.ipynb                # End-to-end pipeline demonstration
│
├── text_to_graph/                         # LLM-based graph generation
│   ├── __init__.py
│   ├── workflow.py                        # Main orchestrator
│   ├── prompt_expander.py                 # Prompt expansion with LLM
│   ├── graph_generator.py                 # Graph generation with LLM
│   ├── schemas.py                         # Pydantic schemas for LLM I/O
│   └── example.py                         # Interactive example script
│
├── graphs/                                # Graph processing and utilities
│   ├── schema.py                          # Core FloorplanGraph dataclasses
│   ├── numerical_graph.py                 # Graph → tensor conversion
│   ├── for_resplan/                       # ResPlan dataset integration
│   │   ├── resplan_adapter.py
│   │   └── serialize_graph.py             # Canonical serialization
│   └── for_rplan/                         # RPlan dataset (legacy)
│       ├── extract_adjacency.py
│       ├── rplan_to_graph.py
│       └── serialize.py
│
├── dataset/                               # Dataset utilities
│   ├── __init__.py
│   ├── resplan_utils.py                   # ResPlan visualization & processing
│   └── (ResPlan.pkl - external data, not included)
│
├── scripts/                               # Standalone scripts
│   ├── build_numeric_dataset.py           # Build numeric dataset from ResPlan
│   └── build_numeric_subset.py            # Build subset for testing
│
├── tests/                                 # Test suite
│   ├── test_text_to_graph.py              # Text-to-graph pipeline tests
│   ├── test_resplan_graph.py              # Graph object tests
│   └── test_resplan_graph_serialization.py # Serialization tests
│
├── model_notebook_training/               # Model training notebooks
│   ├── model_testing.ipynb                # Training & evaluation pipeline
│   └── rough_layout_to_floorplan.ipynb    # Layout visualization
│
├── docs/                                  # Documentation
│   └── MODEL_TESTING_EXPLANATION.md       # Model testing guide
│
└── trained_models/                        # Directory for model checkpoints
```

## 🔑 Key Workflows

### 1. Text-to-Graph Generation

**Module:** `text_to_graph/workflow.py`

```
User Prompt (text)
    ↓ FloorplanWorkflow.expand_prompt()
Room Expansion (with context)
    ↓ FloorplanWorkflow.generate_graph()
Graph Schema (rooms + edges)
    ↓ (Manual conversion)
FloorplanGraph (canonical)
    ↓ graph_to_canonical_text()
Serialized Graph (text)
```

**Example:**
```python
workflow = FloorplanWorkflow(api_key="...", model="gpt-4o")
expanded = workflow.expand_prompt("compact apartment")
graph_schema = workflow.generate_graph(expanded, "compact apartment")
```

### 2. Graph to Numerical Tensors

**Module:** `graphs/numerical_graph.py`

Converts `FloorplanGraph` objects to PyTorch tensors:
- **Node features** (N × 11): One-hot room types + attributes
- **Edge index** (2 × E): Node pairs defining connections
- **Edge attributes** (E × 10): Relationship type, spatial direction

```python
from graphs.numerical_graph import floorgraph_to_numeric
import torch

numeric_x, numeric_edge_index, numeric_edge_attr = floorgraph_to_numeric(graph, plan_width=1.0)
data = Data(
    x=torch.tensor(numeric_x, dtype=torch.float32),
    edge_index=torch.tensor(numeric_edge_index, dtype=torch.long),
    edge_attr=torch.tensor(numeric_edge_attr, dtype=torch.float32),
)
```

### 3. Layout Prediction with GNN

**Model:** `RoughLayoutGNN` (defined in notebooks)

```python
model = RoughLayoutGNN(node_dim=11, edge_dim=10, hidden=128)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

with torch.no_grad():
    predicted_boxes = model(data).cpu().numpy()
    # Output: (num_rooms, 4) - [cx, cy, width, height] for each room
```

## 🧪 Testing

Run the test suite to verify functionality:

```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_text_to_graph.py

# Run with verbose output
python -m pytest tests/ -v
```

**Test Modules:**
- `test_text_to_graph.py`: Validates prompt expansion and graph generation
- `test_resplan_graph.py`: Tests graph object creation and validation
- `test_resplan_graph_serialization.py`: Verifies canonical serialization round-tripping

## 📚 Documentation

- **[MODEL_TESTING_EXPLANATION.md](docs/MODEL_TESTING_EXPLANATION.md)** - Detailed guide to model training, evaluation, and visualization
- **[complete_workflow.ipynb](complete_workflow.ipynb)** - Interactive demonstration of the full pipeline
- **[model_testing.ipynb](model_notebook_training/model_testing.ipynb)** - Training and inference walkthrough

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root with:

```bash
# LLM API Configuration
OPENAI_API_KEY=your-api-key-here
# OR
NVIDIA_API_KEY=your-nvidia-api-key-here

# Model Selection (optional)
FLOORPLAN_LLM_MODEL=mistralai/mistral-small-4-119b-2603
# Or use OpenAI model
FLOORPLAN_LLM_MODEL=gpt-4o

# API Endpoint (optional, for custom endpoints)
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
```

### Supported LLM Models

- **OpenAI**: `gpt-4`, `gpt-4o`, `gpt-3.5-turbo`
- **NVIDIA AI Endpoints**: `mistralai/mistral-small-4-119b-2603`, `meta-llama/llama-3.1-405b-instruct`
- **Other OpenAI-compatible APIs**: Configure `OPENAI_BASE_URL`

## 🔄 Room Types Supported

Current supported room types:
- `living` - Living room
- `bedroom` - Bedroom
- `kitchen` - Kitchen
- `bathroom` - Bathroom
- `balcony` - Balcony/outdoor space
- `front_door` - Entrance

Room adjacency types:
- `adjacency` - Rooms are adjacent/neighboring
- `via_door` - Rooms connected by a door

Spatial directions:
- `left_of`, `right_of`, `above`, `below` - Relative positioning

## 📈 Performance & Constraints

- **Maximum rooms per floorplan**: ~15-20 (GNN tested up to 100+ rooms with ResPlan data)
- **Model inference time**: ~50-200ms per floorplan (CPU)
- **LLM call time**: ~1-3 seconds per step (network-dependent)
- **Memory requirements**: ~2-4GB for training, ~500MB for inference

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- Additional room types and spatial relationships
- Multi-floor floorplan support
- Constraint satisfaction improvements
- Performance optimization
- Dataset expansion
- Alternative model architectures

## ⚠️ Known Issues & Limitations

1. **Topology-only generation**: Current text-to-graph stage produces topology without detailed geometry
2. **Prompt sensitivity**: Results depend on prompt clarity and completeness
3. **Room vocabulary**: Limited to pre-defined room types
4. **Layout realism**: GNN may produce unrealistic layouts that require post-processing
5. **Determinism**: LLM outputs are non-deterministic (can be controlled with `temperature=0`)

## 📋 Requirements

See [requirements.txt](requirements.txt) for complete dependency list. Key packages:

- `torch` & `torch_geometric` - Neural network and GNN framework
- `langchain` & `langchain-openai` - LLM orchestration
- `numpy`, `pandas` - Data processing
- `matplotlib`, `networkx` - Visualization
- `pydantic` - Schema validation
- `jupyter` - Interactive notebooks

## 📄 License

This project is licensed under the **GNU General Public License v3 (GPLv3)** - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ResPlan dataset team for spatial relationship annotations
- PyTorch Geometric team for GNN framework
- LangChain community for LLM orchestration patterns

## 📞 Support

For issues, questions, or feedback:

1. Check existing documentation in `docs/`
2. Review test cases in `tests/`
3. Examine notebook demonstrations in `.ipynb` files
4. Create an issue with detailed reproduction steps

---

**Last Updated:** April 2026  
**Status:** Active Development
