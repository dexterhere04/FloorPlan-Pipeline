from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv

from text_to_graph.schemas import FloorplanEdge, FloorplanGraphSchema


@dataclass
class CompletionConfig:
    threshold: float = 0.45
    max_added_edges: int = 24


class EdgeCompletionGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()

        def mlp(in_dim: int, out_dim_local: int):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim_local),
            )

        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.conv1 = GINEConv(mlp(hidden_dim, hidden_dim), edge_dim=edge_dim)
        self.conv2 = GINEConv(mlp(hidden_dim, hidden_dim), edge_dim=edge_dim)
        self.conv3 = GINEConv(mlp(hidden_dim, hidden_dim), edge_dim=edge_dim)

        pair_in = hidden_dim * 4
        self.head = nn.Sequential(
            nn.Linear(pair_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        h = F.relu(self.node_proj(x))
        if edge_index.numel() > 0:
            h = F.relu(self.conv1(h, edge_index, edge_attr))
            h = F.relu(self.conv2(h, edge_index, edge_attr))
            h = F.relu(self.conv3(h, edge_index, edge_attr))
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        pair_index: torch.Tensor,
    ):
        h = self.encode(x, edge_index, edge_attr)
        src = pair_index[:, 0]
        dst = pair_index[:, 1]

        hs = h[src]
        hd = h[dst]
        pair_feat = torch.cat([hs, hd, torch.abs(hs - hd), hs * hd], dim=-1)
        return self.head(pair_feat)


class EdgeCompletionPredictor:
    ROOM_TYPES = ["living", "kitchen", "bedroom", "bathroom", "balcony", "front_door"]
    REL_TYPES = ["adjacency", "via_door", "direct"]
    DIR_TYPES = ["left_of", "right_of", "above", "below"]

    def __init__(self, checkpoint_path: str | Path, config: CompletionConfig | None = None):
        self.config = config or CompletionConfig()
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Edge completion checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)

        self.edge_type_vocab = ckpt["edge_type_vocab"]
        self.model = EdgeCompletionGNN(
            node_dim=int(ckpt["node_dim"]),
            edge_dim=int(ckpt["edge_dim"]),
            hidden_dim=int(ckpt["hidden_dim"]),
            out_dim=len(self.edge_type_vocab),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def _onehot(self, value: str | None, vocab: list[str]):
        vec = np.zeros(len(vocab), dtype=np.float32)
        if value in vocab:
            vec[vocab.index(value)] = 1.0
        return vec

    def _node_features(self, graph_schema: FloorplanGraphSchema):
        rows = []
        for room in graph_schema.rooms:
            type_vec = self._onehot(room.type, self.ROOM_TYPES)
            rows.append(np.concatenate([type_vec, np.zeros(5, dtype=np.float32)]))
        return torch.tensor(np.asarray(rows, dtype=np.float32), dtype=torch.float32)

    def _edge_features(self, edge: FloorplanEdge):
        rel_vec = self._onehot(edge.relation, self.REL_TYPES)
        dir_vec = self._onehot(edge.direction, self.DIR_TYPES)
        geom = np.zeros(3, dtype=np.float32)
        return np.concatenate([rel_vec, geom, dir_vec]).astype(np.float32)

    def _observed_edges(self, graph_schema: FloorplanGraphSchema, room_idx: dict[str, int]):
        idx_pairs = []
        attrs = []
        for edge in graph_schema.edges:
            if edge.src not in room_idx or edge.dst not in room_idx:
                continue
            i = room_idx[edge.src]
            j = room_idx[edge.dst]
            feat = self._edge_features(edge)
            # The model was trained with bidirectional edge entries.
            idx_pairs.append([i, j])
            idx_pairs.append([j, i])
            attrs.append(feat)
            attrs.append(feat)

        if not idx_pairs:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, 10), dtype=torch.float32),
            )

        edge_index = torch.tensor(np.asarray(idx_pairs).T, dtype=torch.long)
        edge_attr = torch.tensor(np.asarray(attrs, dtype=np.float32), dtype=torch.float32)
        return edge_index, edge_attr

    def _all_pairs(self, n_nodes: int):
        pairs = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                pairs.append((i, j))
        if not pairs:
            return torch.zeros((0, 2), dtype=torch.long)
        return torch.tensor(np.asarray(pairs), dtype=torch.long)

    def _deterministic_rng(self, graph_schema: FloorplanGraphSchema) -> np.random.Generator:
        token = "|".join(sorted(room.id for room in graph_schema.rooms))
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        return np.random.default_rng(seed)

    def _probabilistic_candidate_selection(
        self,
        graph_schema: FloorplanGraphSchema,
        probs: np.ndarray,
        pair_np: np.ndarray,
        observed_pairs: set[tuple[int, int]],
    ) -> list[tuple[float, int, int, str, str | None]]:
        """
        Build candidates by probabilistically selecting high-likelihood room pairs,
        prioritizing pairs that improve graph connectivity before dense fill-in.
        """
        if probs.size == 0 or len(pair_np) == 0:
            return []

        rng = self._deterministic_rng(graph_schema)
        n_rooms = len(graph_schema.rooms)

        pair_records = []
        for row_idx, (i_raw, j_raw) in enumerate(pair_np):
            a = int(i_raw)
            b = int(j_raw)
            if (a, b) in observed_pairs:
                continue

            row = probs[row_idx]
            cls_idx = int(np.argmax(row))
            conf = float(row[cls_idx])

            rel, direction = self.edge_type_vocab[cls_idx].split("|", 1)
            pair_records.append(
                {
                    "a": a,
                    "b": b,
                    "cls_idx": cls_idx,
                    "conf": conf,
                    "relation": rel,
                    "direction": None if direction == "none" else direction,
                }
            )

        if not pair_records:
            return []

        parent = list(range(n_rooms))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int):
            rx = find(x)
            ry = find(y)
            if rx != ry:
                parent[ry] = rx

        for i, j in observed_pairs:
            union(i, j)

        selected: list[dict] = []
        remaining = pair_records.copy()
        budget = max(0, int(self.config.max_added_edges))

        # Stage 1: probabilistically connect disconnected components.
        while budget > 0 and remaining:
            bridge_candidates = [rec for rec in remaining if find(rec["a"]) != find(rec["b"])]
            if not bridge_candidates:
                break

            # Softer threshold for bridge edges to avoid disconnected graphs.
            eligible = [
                rec
                for rec in bridge_candidates
                if rec["conf"] >= max(self.config.threshold * 0.6, 0.15)
            ]
            if not eligible:
                break

            weights = np.array([max(rec["conf"], 1e-6) ** 2.2 for rec in eligible], dtype=np.float64)
            weights /= weights.sum()
            pick_idx = int(rng.choice(len(eligible), p=weights))
            picked = eligible[pick_idx]

            selected.append(picked)
            union(picked["a"], picked["b"])
            budget -= 1
            remaining = [
                rec
                for rec in remaining
                if not (rec["a"] == picked["a"] and rec["b"] == picked["b"])
            ]

        # Stage 2: fill remaining budget with probabilistic high-confidence edges.
        if budget > 0 and remaining:
            fill_pool = [rec for rec in remaining if rec["conf"] >= self.config.threshold]
            while budget > 0 and fill_pool:
                weights = np.array([max(rec["conf"], 1e-6) ** 1.6 for rec in fill_pool], dtype=np.float64)
                weights /= weights.sum()
                pick_idx = int(rng.choice(len(fill_pool), p=weights))
                picked = fill_pool.pop(pick_idx)
                selected.append(picked)
                budget -= 1

        selected.sort(reverse=True, key=lambda rec: rec["conf"])
        return [
            (rec["conf"], rec["a"], rec["b"], rec["relation"], rec["direction"])
            for rec in selected
        ]

    def _ensure_minimal_validity(
        self,
        graph_schema: FloorplanGraphSchema,
        completed_edges: list[FloorplanEdge],
        max_new_edges: int,
    ) -> list[FloorplanEdge]:
        if max_new_edges <= 0 or len(graph_schema.rooms) < 2:
            return completed_edges

        existing_by_pair = set()
        room_types = {room.id: room.type for room in graph_schema.rooms}
        for edge in completed_edges:
            key = tuple(sorted((edge.src, edge.dst)))
            existing_by_pair.add(key)

        adjacency = {room.id: set() for room in graph_schema.rooms}
        for edge in completed_edges:
            adjacency[edge.src].add(edge.dst)
            adjacency[edge.dst].add(edge.src)

        additions = 0

        def add_edge_if_missing(src: str, dst: str, relation: str = "adjacency") -> bool:
            nonlocal additions
            if additions >= max_new_edges:
                return False
            key = tuple(sorted((src, dst)))
            if key in existing_by_pair:
                return False
            completed_edges.append(
                FloorplanEdge(src=src, dst=dst, relation=relation, direction=None)
            )
            existing_by_pair.add(key)
            adjacency[src].add(dst)
            adjacency[dst].add(src)
            additions += 1
            return True

        living_rooms = [room.id for room in graph_schema.rooms if room.type == "living"]
        hub = living_rooms[0] if living_rooms else graph_schema.rooms[0].id

        front_doors = [room.id for room in graph_schema.rooms if room.type == "front_door"]
        for fd in front_doors:
            if fd == hub:
                continue
            if hub not in adjacency[fd]:
                if not add_edge_if_missing(fd, hub, relation="via_door"):
                    break

        # Ensure every non-front-door room has at least one edge.
        for room in graph_schema.rooms:
            if room.type == "front_door":
                continue
            if not adjacency[room.id]:
                if room.id == hub:
                    continue
                if not add_edge_if_missing(room.id, hub, relation="adjacency"):
                    break

        # Ensure overall graph is connected by linking components to the hub.
        if additions < max_new_edges:
            unvisited = set(adjacency.keys())
            while unvisited:
                start = next(iter(unvisited))
                stack = [start]
                component = set()
                while stack:
                    node = stack.pop()
                    if node in component:
                        continue
                    component.add(node)
                    stack.extend(adjacency[node] - component)
                unvisited -= component

                if hub in component:
                    continue

                candidates = sorted(
                    component,
                    key=lambda rid: (
                        room_types.get(rid) == "front_door",
                        room_types.get(rid) == "balcony",
                    ),
                )
                if not candidates:
                    continue
                if not add_edge_if_missing(candidates[0], hub, relation="adjacency"):
                    break

        return completed_edges

    def complete_graph(self, graph_schema: FloorplanGraphSchema) -> FloorplanGraphSchema:
        if len(graph_schema.rooms) < 2:
            return graph_schema

        room_idx = {room.id: i for i, room in enumerate(graph_schema.rooms)}
        idx_room = {i: room_id for room_id, i in room_idx.items()}

        x = self._node_features(graph_schema).to(self.device)
        edge_index, edge_attr = self._observed_edges(graph_schema, room_idx)
        pair_index = self._all_pairs(len(graph_schema.rooms))

        if pair_index.numel() == 0:
            return graph_schema

        pair_index_dev = pair_index.to(self.device)
        with torch.no_grad():
            logits = self.model(
                x,
                edge_index.to(self.device),
                edge_attr.to(self.device),
                pair_index_dev,
            )
            probs = torch.sigmoid(logits).cpu().numpy()

        observed_pairs = set()
        for edge in graph_schema.edges:
            if edge.src not in room_idx or edge.dst not in room_idx:
                continue
            a, b = sorted((room_idx[edge.src], room_idx[edge.dst]))
            observed_pairs.add((a, b))

        pair_np = pair_index.cpu().numpy()
        candidates = self._probabilistic_candidate_selection(
            graph_schema=graph_schema,
            probs=probs,
            pair_np=pair_np,
            observed_pairs=observed_pairs,
        )

        existing_signatures = {
            tuple(sorted((edge.src, edge.dst))) + (edge.relation, edge.direction)
            for edge in graph_schema.edges
        }

        completed_edges = list(graph_schema.edges)
        model_added_count = 0
        for _conf, i, j, relation, direction in candidates:
            src_id = idx_room[i]
            dst_id = idx_room[j]
            signature = tuple(sorted((src_id, dst_id))) + (relation, direction)
            if signature in existing_signatures:
                continue
            existing_signatures.add(signature)
            completed_edges.append(
                FloorplanEdge(
                    src=src_id,
                    dst=dst_id,
                    relation=relation,
                    direction=direction,
                )
            )
            model_added_count += 1

        remaining_budget = max(0, self.config.max_added_edges - model_added_count)
        completed_edges = self._ensure_minimal_validity(
            graph_schema=graph_schema,
            completed_edges=completed_edges,
            max_new_edges=remaining_budget,
        )

        return graph_schema.model_copy(update={"edges": completed_edges})
