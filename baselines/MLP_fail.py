#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import hashlib
import random
import sys
import json
import atexit
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import special_ortho_group
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

_METRICS_FH = None

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)
    def flush(self):
        for s in self.streams:
            s.flush()
    def isatty(self):
        for s in self.streams:
            if hasattr(s, "isatty") and s.isatty():
                return True
        return False

def _metrics_write(obj):
    global _METRICS_FH
    if _METRICS_FH is None:
        return
    _METRICS_FH.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _METRICS_FH.flush()

def _find_repo_root(start):
    p = start
    while True:
        if (p / "data").exists():
            return p
        if p.parent == p:
            return start
        p = p.parent

def _setup_run(model, seed, dataset, num_points, variant, *, lr, epochs):
    here = Path(__file__).resolve().parent
    repo = _find_repo_root(here)
    run_id = f"{model}_{seed}_{dataset}_{num_points}_{variant}"
    stdout_path = repo / f"{run_id}.stdout.log"
    config_path = repo / f"{run_id}.config.json"
    metrics_path = repo / f"{run_id}.metrics.jsonl"

    orig_out, orig_err = sys.stdout, sys.stderr
    fh = open(stdout_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(orig_out, fh)
    sys.stderr = _Tee(orig_err, fh)

    global _METRICS_FH
    _METRICS_FH = open(metrics_path, "w", encoding="utf-8", buffering=1)

    def _cleanup():
        try:
            sys.stdout.flush(); sys.stderr.flush()
        except Exception:
            pass
        try:
            fh.close()
        except Exception:
            pass
        try:
            _METRICS_FH.close()
        except Exception:
            pass

    atexit.register(_cleanup)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": str(model),
                "seed": int(seed),
                "dataset": str(dataset),
                "variant": str(variant),
                "num_points": int(num_points),
                "epochs": int(epochs),
                "lr": float(lr),
                "timestamp": datetime.datetime.now().isoformat(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    _metrics_write({"event": "start", "run_id": run_id, "timestamp": datetime.datetime.now().isoformat()})
    return run_id

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

def make_subseed(base_seed: int, *keys) -> int:
    h = hashlib.sha256(str((base_seed,) + tuple(keys)).encode()).hexdigest()
    return int(h[:8], 16)

def make_rng_pack(base_seed: int, points: int, dataset_tag: str):
    subseed = make_subseed(base_seed, points, dataset_tag)
    np_rs = np.random.RandomState(subseed)
    torch_gen_cpu = torch.Generator(device="cpu").manual_seed(subseed)
    torch_gen_cuda = None
    if torch.cuda.is_available():
        torch_gen_cuda = torch.Generator(device="cuda").manual_seed(subseed)
    return dict(subseed=subseed, np_rs=np_rs, torch_gen_cpu=torch_gen_cpu, torch_gen_cuda=torch_gen_cuda)

def calculate_final_metrics(y_true, y_pred, num_classes_):
    y_true_np = np.asarray(y_true).reshape(-1)
    y_pred_np = np.asarray(y_pred).reshape(-1)
    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(num_classes_)))
    cls_acc = []
    for i in range(num_classes_):
        denom = float(np.sum(cm[i, :]))
        cls_acc.append(float(cm[i, i] / denom) if denom > 0 else 0.0)
    overall = float(np.trace(cm) / np.sum(cm)) if np.sum(cm) > 0 else 0.0
    return cm, cls_acc, overall

class MLPNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_hidden, out_dim):
        super().__init__()
        self.hidden = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
        for _ in range(num_hidden - 1):
            self.hidden.append(nn.Linear(hidden_dim, hidden_dim))
        self.relu = nn.relu()
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = x.view(x.size(0), -1)
        for layer in self.hidden:
            h = self.relu(layer(h))
        return self.fc_out(h)

def run_one_block(base_seed: int, points: int, dataset_tag: str, variant: str, *, epochs: int, lr: float):
    coords = 3
    dim_per_part = points * coords
    batch_size = 35

    if variant == "light":
        hidden_dim = 22
        num_hidden = 3
    else:
        hidden_dim = 35
        num_hidden = 6

    if dataset_tag == "SUO":
        num_classes = 3
        sigma = 0.01
    else:
        num_classes = 5
        sigma = 0.02

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = make_rng_pack(base_seed, points, dataset_tag)
    np_rs = rng["np_rs"]
    tg_cpu = rng["torch_gen_cpu"]
    tg_cuda = rng["torch_gen_cuda"]
    tg_loader = tg_cpu

    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(base_seed)
        torch.cuda.manual_seed_all(base_seed)

    def random_3d_rotation():
        return special_ortho_group.rvs(3, random_state=np_rs)

    def apply_3d_rotation(pts: torch.Tensor):
        R = torch.tensor(random_3d_rotation(), device=pts.device, dtype=pts.dtype)
        return torch.matmul(pts, R.T)

    def add_jitter(pts: torch.Tensor, sigma_: float):
        gen = tg_cuda if (pts.is_cuda and tg_cuda is not None) else tg_cpu
        noise = torch.randn(pts.shape, device=pts.device, dtype=pts.dtype, generator=gen) * sigma_
        return pts + noise

    def apply_permutation(pts: torch.Tensor):
        n = pts.shape[0]
        perm_indices = np_rs.permutation(n).astype(np.int64)
        perm_t = torch.as_tensor(perm_indices, device=pts.device, dtype=torch.long)
        return pts.index_select(0, perm_t)

    def apply_data_augmentation(pts: torch.Tensor, is_training=True):
        if not is_training:
            return pts
        out = add_jitter(pts, sigma)
        out = apply_3d_rotation(out)
        out = apply_permutation(out)
        return out

    def augment_batch(batch_points: torch.Tensor, is_training=True):
        if not is_training:
            return batch_points
        aug_list = []
        for i in range(batch_points.shape[0]):
            aug_list.append(apply_data_augmentation(batch_points[i], is_training))
        return torch.stack(aug_list, dim=0)

    HERE = Path(__file__).resolve().parent
    REPO = HERE.parent

    if dataset_tag == "modelnet":
        npz_name = f"modelnet40_5classes_{points}_1_fps_train700_val100_test200_new.npz"
        dataset_path = REPO / "data" / "ModelNet" / npz_name
    elif dataset_tag == "shapenet":
        npz_name = f"shapenet_5classes_{points}_1_fps_train700_val100_test200_new.npz"
        dataset_path = REPO / "data" / "ShapeNet" / npz_name
    elif dataset_tag == "SUO":
        npz_name = f"SUO_3classes_{points}_1_fps_train700_val100_test200_new.npz"
        dataset_path = REPO / "data" / "Sydney_Urban_Objects" / npz_name
    else:
        raise ValueError("dataset_tag must be 'modelnet', 'shapenet', or 'SUO'")

    data = np.load(dataset_path)
    train_x = data["train_dataset_x"].reshape(-1, points, coords).astype(np.float32)
    train_y = data["train_dataset_y"].astype(np.int64)
    val_x = data["val_dataset_x"].reshape(-1, points, coords).astype(np.float32)
    val_y = data["val_dataset_y"].astype(np.int64)
    test_x = data["test_dataset_x"].reshape(-1, points, coords).astype(np.float32)
    test_y = data["test_dataset_y"].astype(np.int64)

    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=tg_loader, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    torch.manual_seed(rng["subseed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rng["subseed"])
        torch.cuda.manual_seed_all(rng["subseed"])

    model = MLPNet(dim_per_part, hidden_dim, num_hidden, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    @torch.no_grad()
    def evaluate_loss(model_, loader):
        model_.eval()
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model_(xb)
            loss = criterion(logits, yb)
            total += float(loss.item()) * int(yb.numel())
            n += int(yb.numel())
        return total / max(n, 1)

    @torch.no_grad()
    def evaluate_accuracy(model_, loader):
        model_.eval()
        correct, total = 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model_(xb)
            pred = logits.argmax(1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
        return correct / total if total > 0 else 0.0

    best_val = 0.0
    best_epoch = -1
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        sum_loss = 0.0
        n = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = augment_batch(xb, is_training=True)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            sum_loss += float(loss.item()) * int(yb.numel())
            n += int(yb.numel())

        train_loss = sum_loss / max(n, 1)
        val_loss = evaluate_loss(model, val_loader)
        val_acc = evaluate_accuracy(model, val_loader)

        if val_acc >= best_val:
            best_val = float(val_acc)
            best_epoch = int(epoch - 1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"epoch {epoch-1}/{epochs-1} | train loss : {train_loss:.4f} | val loss : {val_loss:.4f} | val accuracy : {val_acc:.4f}")
        _metrics_write({"epoch": int(epoch-1), "train_loss": float(train_loss), "val_loss": float(val_loss), "val_acc": float(val_acc)})

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pred = logits.argmax(1)
            y_true.append(yb.detach().cpu().numpy())
            y_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    _, cls_accs, overall = calculate_final_metrics(y_true, y_pred, num_classes)

    print("\n=== Results ===")
    print(f"Test Accuracy: {float(overall):.4f}")
    print("Class-wise Accuracy:")
    for i, acc in enumerate(cls_accs):
        print(f"  Class {i}: {float(acc):.4f}")

    _metrics_write(
        {
            "final": True,
            "best_epoch": int(best_epoch),
            "best_val_acc": float(best_val),
            "test_acc": float(overall),
            "class_acc": [float(a) for a in cls_accs],
        }
    )

    return float(overall)

def normalize_variant(variant: str) -> str:
    if variant is None:
        raise ValueError("variant is required")
    v = variant.lower()
    if v == "light":
        return "light"
    if v == "mid":
        return "mid"
    raise ValueError("variant must be one of: light, mid")

def resolve_dataset(dataset: str, num_points: int):
    if dataset is None:
        raise ValueError("dataset is required")
    d = dataset.lower()
    if d == "modelnet":
        return "modelnet", f"modelnet40_5classes_{num_points}_1_fps_train700_val100_test200_new.npz"
    if d == "shapenet":
        return "shapenet", f"shapenet_5classes_{num_points}_1_fps_train700_val100_test200_new.npz"
    if d == "suo":
        return "SUO", f"SUO_3classes_{num_points}_1_fps_train700_val100_test200_new.npz"
    raise ValueError("dataset must be one of: modelnet, shapenet, suo")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("--dataset", type=str, choices=["modelnet", "shapenet", "suo"], required=True)
    parser.add_argument("--num_points", type=int, required=True)
    parser.add_argument("--variant", type=str, choices=["light", "mid"], required=True)
    args = parser.parse_args()

    base_seed = int(args.seed)
    num_points = int(args.num_points)
    variant = normalize_variant(args.variant)
    dataset_tag, _ = resolve_dataset(args.dataset, num_points)

    lr = 1e-3
    epochs = 1000

    _setup_run(Path(__file__).stem, base_seed, args.dataset, num_points, variant, lr=lr, epochs=epochs)

    print(f"seed={base_seed}, dataset={args.dataset}, variant={variant}, num_points={num_points}, epochs={epochs}, lr={lr}")

    acc = run_one_block(
        base_seed,
        num_points,
        dataset_tag,
        variant,
        epochs=epochs,
        lr=lr,
    )

    print(float(acc))

if __name__ == "__main__":
    main()
