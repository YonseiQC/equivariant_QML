#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import atexit
import datetime

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

def _setup_run(model, seed, dataset, num_points, variant, *, lr, epochs, k=None):
    here = Path(__file__).resolve().parent
    repo = _find_repo_root(here)
    run_id = f"{model}_{seed}_{dataset}_{num_points}_{variant}"
    if k is not None:
        run_id = f"{run_id}_{k}"
    stdout_path = repo / f"{run_id}.stdout.log"
    config_path = repo / f"{run_id}.config.json"
    metrics_path = repo / f"{run_id}.metrics.jsonl"

    orig_out, orig_err = sys.stdout, sys.stderr
    fh = open(stdout_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(orig_out, fh)
    sys.stderr = _Tee(orig_err, fh)

    global _METRICS_FH
    _METRICS_FH = open(metrics_path, "a", encoding="utf-8", buffering=1)

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
                "k": None if k is None else int(k),
                "timestamp": datetime.datetime.now().isoformat(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    _metrics_write({"event": "start", "run_id": run_id, "timestamp": datetime.datetime.now().isoformat()})
    return run_id
import argparse
from pathlib import Path
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import special_ortho_group
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


# ====================== Determinism & RNG pack ======================

def make_rng_pack(seed: int):
    py_rng = random.Random(seed)

    np_gen = np.random.default_rng(seed)
    np_rs = np.random.RandomState(seed)

    torch_gen_cpu = torch.Generator(device="cpu").manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch_gen_cuda = torch.Generator(device="cuda").manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        device = torch.device("cpu")
        torch_gen_cuda = None

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    return dict(
        seed=seed,
        py_rng=py_rng,
        np_gen=np_gen,
        np_rs=np_rs,
        torch_gen_cpu=torch_gen_cpu,
        torch_gen_cuda=torch_gen_cuda,
        device=device,
    )


def fixed_loader(dataset, batch_size, shuffle, torch_gen):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=torch_gen,
        num_workers=0,
        pin_memory=True,
    )


# ============================ Utils ============================

def normalize_unit_sphere(points):
    pts = points.copy()
    centroid = np.mean(pts, axis=0)
    pts -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(pts**2, axis=1)))
    if furthest_distance > 0:
        pts /= furthest_distance
    return pts


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k=k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :].view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return feature


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---- Augmentation: Jitter -> Rotation -> Permutation ----

def _random_3d_rotation(np_rs):
    return special_ortho_group.rvs(3, random_state=np_rs)


def _add_jitter(points, sigma, np_gen):
    noise = np_gen.normal(0.0, sigma, size=points.shape)
    return points + noise


def _apply_3d_rotation(points, np_rs):
    R = _random_3d_rotation(np_rs)
    return points.dot(R.T)


def _apply_permutation(points, np_rs):
    return points[np_rs.permutation(points.shape[0])]


def apply_data_augmentation(points, sigma, np_gen, np_rs, is_training=True):
    if not is_training:
        return points
    pts = _add_jitter(points, sigma, np_gen)
    pts = _apply_3d_rotation(pts, np_rs)
    pts = _apply_permutation(pts, np_rs)
    return pts


# ============================== Tiny 3x3 STN ==============================

class TNet3x3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 16, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 9, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(9)
        self.fc = nn.Linear(9, 9)

        nn.init.zeros_(self.fc.weight)
        with torch.no_grad():
            self.fc.bias.zero_()
            self.fc.bias += torch.eye(3).reshape(-1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x = self.fc(x).view(-1, 3, 3)
        return x


# ============================ Compact DGCNN ============================

class CompactDGCNN(nn.Module):
    def __init__(self, num_classes, k, dropout, c1, c2, c3, f1, m):
        super().__init__()
        self.k = k
        self.stn = TNet3x3()

        self.bn1 = nn.BatchNorm2d(c1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.bn3 = nn.BatchNorm2d(c3)
        self.bn_fc = nn.BatchNorm1d(f1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, c1, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * c1, c2, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * c2, c3, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(0.2),
        )

        gdim = c1 + c2 + c3

        self.fc1 = nn.Sequential(
            nn.Linear(gdim, f1, bias=False),
            self.bn_fc,
            nn.LeakyReLU(0.2),
            nn.Dropout(p=dropout),
        )
        self.fc2 = nn.Linear(f1, m)
        self.fc3 = nn.Linear(m, num_classes)

    def forward(self, x):
        B = x.size(0)

        trans = self.stn(x)
        x = torch.bmm(trans, x)

        x = get_graph_feature(x, k=self.k)
        x1 = self.conv1(x).max(dim=-1)[0]

        x = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x).max(dim=-1)[0]

        x = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x).max(dim=-1)[0]

        x = torch.cat((x1, x2, x3), dim=1)
        x = F.adaptive_max_pool1d(x, 1).view(B, -1)

        x = self.fc1(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.fc3(x)
        return x


# =============================== Dataset ===============================

class PointCloudDataset(Dataset):
    def __init__(self, data_path, num_points, split="train", sigma=0.02, np_gen=None, np_rs=None):
        with np.load(data_path) as data:
            if split == "train":
                self.points = data["train_dataset_x"]
                self.labels = data["train_dataset_y"]
            elif split == "val":
                self.points = data["val_dataset_x"]
                self.labels = data["val_dataset_y"]
            elif split == "test":
                self.points = data["test_dataset_x"]
                self.labels = data["test_dataset_y"]
            else:
                raise ValueError("split must be one of {'train','val','test'}")
        self.num_points = num_points
        self.split = split
        self.sigma = sigma
        self.np_gen = np_gen
        self.np_rs = np_rs

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        pointcloud = self.points[idx].copy()

        if len(pointcloud) < self.num_points:
            indices = self.np_gen.choice(len(pointcloud), self.num_points, replace=True)
        else:
            indices = self.np_gen.choice(len(pointcloud), self.num_points, replace=False)
        pointcloud = pointcloud[indices]

        is_training = self.split == "train"
        pointcloud = apply_data_augmentation(
            pointcloud, self.sigma, self.np_gen, self.np_rs, is_training=is_training
        )
        pointcloud = normalize_unit_sphere(pointcloud)

        return torch.FloatTensor(pointcloud.T), torch.LongTensor([self.labels[idx]])


# ============================ Train / Eval ============================

def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for data, label in train_loader:
        data, label = data.to(device), label.squeeze().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


@torch.no_grad()
@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    for data, label in loader:
        data = data.to(device)
        label = label.squeeze().to(device)
        out = model(data)
        loss = F.cross_entropy(out, label, reduction="sum")
        total += float(loss.item())
        n += int(label.numel())
    return total / max(n, 1)

def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for data, label in loader:
        data, label = data.to(device), label.squeeze().to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.numel()
    return correct / total if total > 0 else 0.0


# =============================== Runner ===============================

def run_experiment(
    dataset_file,
    num_points,
    k,
    model_cfg,
    *,
    num_classes=5,
    batch_size=35,
    epochs,
        lr,
    dropout=0.0,
    sigma=0.02,
    seed=831,
):
    rng = make_rng_pack(seed)
    device = rng["device"]
    torch_gen = rng["torch_gen_cpu"]

    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"'{dataset_file}' does not exist.")

    train_dataset = PointCloudDataset(
        dataset_file, num_points=num_points, split="train", sigma=sigma, np_gen=rng["np_gen"], np_rs=rng["np_rs"]
    )
    val_dataset = PointCloudDataset(
        dataset_file, num_points=num_points, split="val", sigma=sigma, np_gen=rng["np_gen"], np_rs=rng["np_rs"]
    )
    test_dataset = PointCloudDataset(
        dataset_file, num_points=num_points, split="test", sigma=sigma, np_gen=rng["np_gen"], np_rs=rng["np_rs"]
    )

    train_loader = fixed_loader(train_dataset, batch_size=batch_size, shuffle=True, torch_gen=torch_gen)
    val_loader = fixed_loader(val_dataset, batch_size=batch_size, shuffle=False, torch_gen=torch_gen)
    test_loader = fixed_loader(test_dataset, batch_size=batch_size, shuffle=False, torch_gen=torch_gen)

    if k >= num_points:
        k = max(1, num_points)
        print(f"{k} changes to {num_points}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model = CompactDGCNN(num_classes=num_classes, k=k, dropout=dropout, **model_cfg).to(device)
    total_params = count_parameters(model)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0)

    best_val = 0.0
    best_epoch = -1
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        train_acc = evaluate_accuracy(model, train_loader, device)
        val_acc = evaluate_accuracy(model, val_loader, device)
        val_loss = evaluate_loss(model, val_loader, device)

        if val_acc >= best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state_dict = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
        # torch.save(...) disabled (no extra files)
        print(f"epoch {epoch-1}/{epochs-1} | train loss : {train_loss:.4f} | val loss : {val_loss:.4f} | val accuracy : {val_acc:.4f}")
        _metrics_write({"epoch": int(epoch-1), "train_loss": float(train_loss), "val_loss": float(val_loss), "val_acc": float(val_acc)})

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    test_acc = evaluate_accuracy(model, test_loader, device)

    print("\n=== Results ===")
    print(f"Test Accuracy: {float(test_acc):.4f}")
    print("Class-wise Accuracy:")
    for i, acc in enumerate(cls_accs):
        print(f"  Class {i}: {acc:.4f}")

    return test_acc


def normalize_variant(variant: str) -> str:
    if variant is None:
        raise ValueError("variant is required")
    v = variant.lower()
    if v == "light":
        return "light"
    if v == "mid":
        return "mid"
    raise ValueError("variant must be one of: light, mid")


def variant_config(variant: str):
    if variant == "light":
        return dict(c1=7, c2=15, c3=8, f1=13, m=9)
    return dict(c1=8, c2=32, c3=72, f1=16, m=9)


def resolve_dataset(dataset: str, num_points: int):
    if dataset is None:
        raise ValueError("dataset is required")
    d = dataset.lower()
    if d == "modelnet":
        return "modelnet", f"modelnet40_5classes_{num_points}_1_fps_train700_val100_test200_new.npz", 5, 0.02
    if d == "shapenet":
        return "shapenet", f"shapenet_5classes_{num_points}_1_fps_train700_val100_test200_new.npz", 5, 0.02
    if d == "suo":
        return "SUO", f"SUO_3classes_{num_points}_1_fps_train700_val100_test200_new.npz", 3, 0.01
    raise ValueError("dataset must be one of: modelnet, shapenet, suo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("--dataset", type=str, choices=["modelnet", "shapenet", "suo"], required=True)
    parser.add_argument("--num_points", type=int, required=True)
    parser.add_argument("--variant", type=str, choices=["light", "mid"], required=True)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()

    base_seed = args.seed
    num_points = args.num_points
    variant = normalize_variant(args.variant)
    model_cfg = variant_config(variant)
    dataset_tag, dataset_file, num_classes, sigma = resolve_dataset(args.dataset, num_points)
    HERE = Path(__file__).resolve().parent
    REPO = HERE.parent

    tag = str(dataset_tag).lower()
    if tag == "modelnet":
        dataset_file = str(REPO / "data" / "ModelNet" / dataset_file)
    elif tag == "shapenet":
        dataset_file = str(REPO / "data" / "ShapeNet" / dataset_file)
    else:
        dataset_file = str(REPO / "data" / "Sydney_Urban_Objects" / dataset_file)
    k = args.k

    epochs = 1000
    lr = 0.01
    _setup_run(Path(__file__).stem, base_seed, args.dataset, num_points, variant, lr=lr, epochs=epochs, k=k)
    print(f"seed={base_seed}, dataset={args.dataset}, variant={variant}, num_points={num_points}, epochs={epochs}, lr={lr}")

    test_acc = run_experiment(
        dataset_file,
        num_points,
        k,
        model_cfg,
        num_classes=num_classes,
        batch_size=35,
        epochs=3,
        lr=0.01,
        dropout=0.0,
        sigma=sigma,
        seed=base_seed,
    )

    print(float(test_acc))


if __name__ == "__main__":
    main()
