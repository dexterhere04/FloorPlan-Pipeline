import argparse
import os
import pickle
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GINEConv


@dataclass
class GraphSample:
    x: torch.Tensor
    kept_edge_index: torch.Tensor
    kept_edge_attr: torch.Tensor
    pair_index: torch.Tensor
    pair_targets: torch.Tensor


class EdgeTypeVocab:
    def __init__(self):
        self.key_to_idx = {}
        self.idx_to_key = []

    def add(self, relation: str, direction: str) -> int:
        key = f"{relation}|{direction}"
        if key not in self.key_to_idx:
            self.key_to_idx[key] = len(self.idx_to_key)
            self.idx_to_key.append(key)
        return self.key_to_idx[key]

    def to_key(self, idx: int) -> str:
        return self.idx_to_key[idx]

    @property
    def size(self) -> int:
        return len(self.idx_to_key)


def to_object_array(items):
    arr = np.empty(len(items), dtype=object)
    arr[:] = list(items)
    return arr


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def decode_edge_type_from_attr(edge_attr_row: np.ndarray):
    rel_types = ["adjacency", "via_door", "direct"]
    dir_types = ["left_of", "right_of", "above", "below"]

    relation = rel_types[int(np.argmax(edge_attr_row[:3]))]
    dir_slice = edge_attr_row[6:10]
    if np.max(dir_slice) <= 0.0:
        direction = "none"
    else:
        direction = dir_types[int(np.argmax(dir_slice))]
    return relation, direction


def build_edge_type_vocab(edge_attrs: np.ndarray) -> EdgeTypeVocab:
    vocab = EdgeTypeVocab()
    for ea in edge_attrs:
        arr = np.asarray(ea)
        if arr.size == 0:
            continue
        for row in arr:
            relation, direction = decode_edge_type_from_attr(row)
            vocab.add(relation, direction)
    return vocab


def build_pair_labels(edge_index: np.ndarray, edge_attr: np.ndarray, n_nodes: int, vocab: EdgeTypeVocab):
    pair_to_types = {}
    m = edge_index.shape[1]

    for k in range(m):
        i = int(edge_index[0, k])
        j = int(edge_index[1, k])
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        relation, direction = decode_edge_type_from_attr(edge_attr[k])
        t_idx = vocab.key_to_idx[f"{relation}|{direction}"]
        pair_to_types.setdefault((a, b), set()).add(t_idx)

    pairs = []
    targets = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            pairs.append((i, j))
            label = np.zeros(vocab.size, dtype=np.float32)
            if (i, j) in pair_to_types:
                for t_idx in pair_to_types[(i, j)]:
                    label[t_idx] = 1.0
            targets.append(label)

    pair_index = torch.tensor(pairs, dtype=torch.long)
    pair_targets = torch.tensor(np.asarray(targets), dtype=torch.float32)
    return pair_index, pair_targets, pair_to_types


def make_observed_subgraph(edge_index: np.ndarray, edge_attr: np.ndarray, pair_to_types: dict, keep_ratio: float):
    kept_pairs = []
    all_pairs = list(pair_to_types.keys())
    for p in all_pairs:
        if random.random() < keep_ratio:
            kept_pairs.append(p)

    if len(kept_pairs) == 0 and len(all_pairs) > 0:
        kept_pairs.append(random.choice(all_pairs))

    kept_set = set(kept_pairs)
    keep_cols = []
    for k in range(edge_index.shape[1]):
        i = int(edge_index[0, k])
        j = int(edge_index[1, k])
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in kept_set:
            keep_cols.append(k)

    if len(keep_cols) == 0:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, edge_attr.shape[1]), dtype=torch.float32),
        )

    kept_edge_index = torch.tensor(edge_index[:, keep_cols], dtype=torch.long)
    kept_edge_attr = torch.tensor(edge_attr[keep_cols], dtype=torch.float32)
    return kept_edge_index, kept_edge_attr


class EdgeCompletionDataset(Dataset):
    def __init__(self, nodes, edge_index, edge_attr, vocab: EdgeTypeVocab, keep_range=(0.3, 0.8)):
        self.nodes = nodes
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.vocab = vocab
        self.keep_range = keep_range

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        x_np = np.asarray(self.nodes[idx], dtype=np.float32)
        ei_np = np.asarray(self.edge_index[idx], dtype=np.int64)
        ea_np = np.asarray(self.edge_attr[idx], dtype=np.float32)

        n_nodes = x_np.shape[0]
        pair_index, pair_targets, pair_to_types = build_pair_labels(ei_np, ea_np, n_nodes, self.vocab)

        keep_ratio = random.uniform(self.keep_range[0], self.keep_range[1])
        kept_edge_index, kept_edge_attr = make_observed_subgraph(ei_np, ea_np, pair_to_types, keep_ratio)

        return GraphSample(
            x=torch.tensor(x_np, dtype=torch.float32),
            kept_edge_index=kept_edge_index,
            kept_edge_attr=kept_edge_attr,
            pair_index=pair_index,
            pair_targets=pair_targets,
        )


def collate_single(batch):
    if len(batch) != 1:
        raise ValueError("Use batch_size=1 for variable-size graph samples")
    return batch[0]


class EdgeCompletionGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()

        def mlp(in_dim, out_dim_local):
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

    def encode(self, x, edge_index, edge_attr):
        h = F.relu(self.node_proj(x))
        if edge_index.numel() > 0:
            h = F.relu(self.conv1(h, edge_index, edge_attr))
            h = F.relu(self.conv2(h, edge_index, edge_attr))
            h = F.relu(self.conv3(h, edge_index, edge_attr))
        return h

    def forward(self, x, edge_index, edge_attr, pair_index):
        h = self.encode(x, edge_index, edge_attr)
        src = pair_index[:, 0]
        dst = pair_index[:, 1]

        hs = h[src]
        hd = h[dst]
        pair_feat = torch.cat([hs, hd, torch.abs(hs - hd), hs * hd], dim=-1)
        logits = self.head(pair_feat)
        return logits


def multilabel_f1(logits, targets, threshold=0.5, eps=1e-8):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    f1_vals = []

    with torch.no_grad():
        for sample in loader:
            x = sample.x.to(device)
            edge_index = sample.kept_edge_index.to(device)
            edge_attr = sample.kept_edge_attr.to(device)
            pair_index = sample.pair_index.to(device)
            targets = sample.pair_targets.to(device)

            logits = model(x, edge_index, edge_attr, pair_index)
            loss = F.binary_cross_entropy_with_logits(logits, targets)

            _, _, f1 = multilabel_f1(logits, targets)
            f1_vals.append(f1)
            total_loss += loss.item()
            n += 1

    return total_loss / max(n, 1), float(np.mean(f1_vals) if f1_vals else 0.0)


def decode_predictions(pair_index, logits, vocab: EdgeTypeVocab, threshold=0.5):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    pairs = pair_index.detach().cpu().numpy()

    predicted = []
    for row_idx, (i, j) in enumerate(pairs):
        active = np.where(probs[row_idx] >= threshold)[0]
        for t_idx in active:
            rel, direction = vocab.to_key(int(t_idx)).split("|", 1)
            direction = None if direction == "none" else direction
            predicted.append(
                {
                    "src": int(i),
                    "dst": int(j),
                    "relation": rel,
                    "direction": direction,
                    "confidence": float(probs[row_idx, t_idx]),
                }
            )
    return predicted


def predict_missing_edges(model, sample: GraphSample, vocab: EdgeTypeVocab, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits = model(
            sample.x.to(device),
            sample.kept_edge_index.to(device),
            sample.kept_edge_attr.to(device),
            sample.pair_index.to(device),
        )

    all_pred = decode_predictions(sample.pair_index, logits, vocab, threshold=threshold)

    observed_pairs = set()
    ei = sample.kept_edge_index.detach().cpu().numpy()
    for k in range(ei.shape[1]):
        a = int(ei[0, k])
        b = int(ei[1, k])
        observed_pairs.add((min(a, b), max(a, b)))

    missing_pred = [
        p
        for p in all_pred
        if (min(p["src"], p["dst"]), max(p["src"], p["dst"])) not in observed_pairs
    ]
    return missing_pred


def train(args):
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset.endswith(".npz"):
        data = np.load(args.dataset, allow_pickle=True)
        nodes = data["nodes"]
        edge_index = data["edge_index"]
        edge_attr = data["edge_attr"]
    elif args.dataset.endswith(".pkl"):
        with open(args.dataset, "rb") as f:
            data = pickle.load(f)
        nodes = to_object_array(data["nodes"])
        edge_index = to_object_array(data["edge_index"])
        edge_attr = to_object_array(data["edge_attr"])
    else:
        raise ValueError("Unsupported dataset format. Use .npz or .pkl")

    vocab = build_edge_type_vocab(edge_attr)
    print(f"Loaded {len(nodes)} graphs")
    print(f"Discovered {vocab.size} edge types: {vocab.idx_to_key}")

    indices = list(range(len(nodes)))
    random.shuffle(indices)

    n_total = len(indices)
    n_train = int(n_total * args.train_split)
    n_val = int(n_total * args.val_split)

    n_train = max(1, min(n_train, n_total - 2))
    n_val = max(1, min(n_val, n_total - n_train - 1))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_ds = EdgeCompletionDataset(nodes[train_idx], edge_index[train_idx], edge_attr[train_idx], vocab)
    val_ds = EdgeCompletionDataset(nodes[val_idx], edge_index[val_idx], edge_attr[val_idx], vocab)
    test_ds = EdgeCompletionDataset(nodes[test_idx], edge_index[test_idx], edge_attr[test_idx], vocab)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_single)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_single)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_single)

    node_dim = int(np.asarray(nodes[0]).shape[1])
    edge_dim = int(np.asarray(edge_attr[0]).shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = EdgeCompletionGNN(node_dim, edge_dim, args.hidden_dim, vocab.size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "best_edge_completion.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for sample in train_loader:
            x = sample.x.to(device)
            ei = sample.kept_edge_index.to(device)
            ea = sample.kept_edge_attr.to(device)
            pi = sample.pair_index.to(device)
            y = sample.pair_targets.to(device)

            logits = model(x, ei, ea, pi)
            loss = F.binary_cross_entropy_with_logits(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        train_loss = float(np.mean(losses) if losses else 0.0)
        val_loss, val_f1 = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_f1={val_f1:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "node_dim": node_dim,
                    "edge_dim": edge_dim,
                    "hidden_dim": args.hidden_dim,
                    "edge_type_vocab": vocab.idx_to_key,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_f1 = evaluate(model, test_loader, device)
    print(f"Test | loss={test_loss:.4f} | f1={test_f1:.4f}")
    print(f"Saved best checkpoint to: {best_path}")

    if args.demo_index >= 0:
        demo_i = min(args.demo_index, len(test_ds) - 1)
        sample = test_ds[demo_i]
        pred_missing = predict_missing_edges(
            model, sample, vocab, device, threshold=args.pred_threshold
        )
        print(
            f"Demo graph {demo_i}: predicted {len(pred_missing)} missing typed edges "
            f"(threshold={args.pred_threshold:.2f})"
        )
        for item in pred_missing[:30]:
            print(
                f"  ({item['src']} -> {item['dst']}) "
                f"{item['relation']} {item['direction']} conf={item['confidence']:.3f}"
            )


def parse_args():
    p = argparse.ArgumentParser(description="Train a GNN for floorplan edge completion")
    p.add_argument("--dataset", type=str, default="dataset/resplan_numeric.npz")
    p.add_argument("--out-dir", type=str, default="trained_models")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--demo-index", type=int, default=0)
    p.add_argument("--pred-threshold", type=float, default=0.5)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
