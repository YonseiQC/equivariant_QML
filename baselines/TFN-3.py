#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import atexit
import datetime
import argparse
from pathlib import Path
import os
import random
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import special_ortho_group
from torch.utils.data import DataLoader, Dataset
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


# ====================== Determinism & RNG pack ======================

def make_subseed(base_seed: int, *keys) -> int:
    h = hashlib.sha256(str((base_seed,) + tuple(keys)).encode()).hexdigest()
    return int(h[:8], 16)

def make_rng_pack(base_seed: int, num_points: int, dataset_tag: str):
    py_rng = random.Random(base_seed)
    np_gen = np.random.default_rng(base_seed)
    np_rs = np.random.RandomState(base_seed)

    torch_gen_cpu = torch.Generator(device="cpu").manual_seed(base_seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch_gen_cuda = torch.Generator(device="cuda").manual_seed(base_seed)
        torch.cuda.manual_seed(base_seed)
        torch.cuda.manual_seed_all(base_seed)
    else:
        device = torch.device("cpu")
        torch_gen_cuda = None

    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    return dict(
        seed=base_seed,
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

EPSILON = 1e-8

def normalize_unit_sphere(points):
    pts = points.copy()
    centroid = np.mean(pts, axis=0)
    pts -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(pts**2, axis=1)))
    if furthest_distance > 0:
        pts /= furthest_distance
    return pts


def calculate_final_metrics(y_true, y_pred, num_classes_):
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()
    cm = confusion_matrix(y_true_np, y_pred_np, labels=range(num_classes_))
    class_accuracies = []
    for i in range(num_classes_):
        denom = np.sum(cm[i, :])
        if denom > 0:
            class_accuracies.append(cm[i, i] / denom)
        else:
            class_accuracies.append(0.0)
    overall_accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0
    return cm, class_accuracies, overall_accuracy


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


# ============================ TFN Model ============================

def github_gaussian_rbf(d, low, high, count):
    centers = torch.linspace(float(low), float(high), int(count), device=d.device, dtype=d.dtype)
    spacing = (float(high) - float(low)) / max(1, int(count))
    gamma = 1.0 / max(spacing, 1e-12)
    x = d.unsqueeze(-1) - centers.view(1, 1, 1, -1)
    return torch.exp(-gamma * x * x)


class RadialFunction(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int | None = None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else int(input_dim)

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, rbf):
        x = F.relu(self.fc1(rbf))
        x = self.fc2(x)
        return x


class ScalarSelfInteraction(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.weight = nn.Parameter(torch.empty(self.output_dim, self.input_dim))
        self.bias = nn.Parameter(torch.zeros(self.output_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        out = torch.einsum("bnc,oc->bno", x, self.weight)
        if self.bias is not None:
            out = out + self.bias.view(1, 1, -1)
        return out


class VectorSelfInteraction(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.weight = nn.Parameter(torch.empty(self.output_dim, self.input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return torch.einsum("bncm,oc->bnom", x, self.weight)


class RotationEquivariantNonlinearity(nn.Module):
    def __init__(self, channels: int, order: int):
        super().__init__()
        self.channels = int(channels)
        self.order = int(order)
        self.bias = nn.Parameter(torch.zeros(self.channels)) if self.order == 1 else None

    def forward(self, x):
        if self.order == 0:
            return F.elu(x)
        if self.order == 1:
            norm = torch.sqrt(torch.sum(x * x, dim=-1) + EPSILON)
            nonlin_out = F.elu(norm + self.bias.view(1, 1, -1))
            factor = nonlin_out / norm
            return x * factor.unsqueeze(-1)
        raise ValueError("Only l=0 and l=1 are supported.")


class ScalarEmbedding(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.si = ScalarSelfInteraction(input_dim=1, output_dim=int(output_dim), bias=False)

    def forward(self, batch_size: int, num_points: int, device, dtype):
        ones = torch.ones((batch_size, num_points, 1), device=device, dtype=dtype)
        return self.si(ones)


class TFNModuleL01(nn.Module):
    def __init__(
        self,
        scalar_in: int,
        vector_in: int,
        scalar_out: int,
        vector_out: int,
        rbf_count: int,
        rbf_low: float,
        rbf_high: float,
        radial_hidden: int | None = None,
    ):
        super().__init__()
        self.scalar_in = int(scalar_in)
        self.vector_in = int(vector_in)
        self.scalar_out = int(scalar_out)
        self.vector_out = int(vector_out)
        self.rbf_count = int(rbf_count)
        self.rbf_low = float(rbf_low)
        self.rbf_high = float(rbf_high)
        self.radial_hidden = int(radial_hidden) if radial_hidden is not None else int(rbf_count)

        self.scalar_to_scalar_f0 = RadialFunction(self.rbf_count, self.scalar_in, self.radial_hidden)
        self.scalar_to_vector_f1 = RadialFunction(self.rbf_count, self.scalar_in, self.radial_hidden)

        if self.vector_in > 0:
            self.vector_to_vector_f0 = RadialFunction(self.rbf_count, self.vector_in, self.radial_hidden)
            self.vector_to_scalar_f1 = RadialFunction(self.rbf_count, self.vector_in, self.radial_hidden)
            self.vector_to_vector_f1 = RadialFunction(self.rbf_count, self.vector_in, self.radial_hidden)
        else:
            self.vector_to_vector_f0 = None
            self.vector_to_scalar_f1 = None
            self.vector_to_vector_f1 = None

        scalar_cat_dim = self.scalar_in + self.vector_in
        vector_cat_dim = self.scalar_in + 2 * self.vector_in

        self.scalar_si = ScalarSelfInteraction(input_dim=scalar_cat_dim, output_dim=self.scalar_out, bias=True)
        self.vector_si = VectorSelfInteraction(input_dim=vector_cat_dim, output_dim=self.vector_out)
        self.scalar_nl = RotationEquivariantNonlinearity(channels=self.scalar_out, order=0)
        self.vector_nl = RotationEquivariantNonlinearity(channels=self.vector_out, order=1)

    @staticmethod
    def _geometry(x):
        xi = x.unsqueeze(2)
        xj = x.unsqueeze(1)
        rij = xi - xj
        dij = torch.sqrt(torch.sum(rij * rij, dim=-1) + EPSILON)
        unit = rij / dij.unsqueeze(-1)
        return rij, dij, unit

    @staticmethod
    def _mask_radial(radial, dij):
        return torch.where(dij.unsqueeze(-1) < EPSILON, torch.zeros_like(radial), radial)

    def forward(self, x, scalar, vector=None):
        _, dij, unit = self._geometry(x)
        rbf = github_gaussian_rbf(dij, low=self.rbf_low, high=self.rbf_high, count=self.rbf_count)

        scalar_outputs = []
        vector_outputs = []

        f0_scalar = self.scalar_to_scalar_f0(rbf)
        scalar_outputs.append(torch.einsum("bijc,bjc->bic", f0_scalar, scalar))

        f1_scalar = self._mask_radial(self.scalar_to_vector_f1(rbf), dij)
        filt_scalar_to_vector = f1_scalar.unsqueeze(-1) * unit.unsqueeze(-2)
        vector_outputs.append(torch.einsum("bijcm,bjc->bicm", filt_scalar_to_vector, scalar))

        if vector is not None:
            f0_vector = self.vector_to_vector_f0(rbf)
            vector_outputs.append(torch.einsum("bijc,bjcm->bicm", f0_vector, vector))

            f1_vector_to_scalar = self._mask_radial(self.vector_to_scalar_f1(rbf), dij)
            filt_vector_to_scalar = f1_vector_to_scalar.unsqueeze(-1) * unit.unsqueeze(-2)
            scalar_outputs.append(torch.einsum("bijcm,bjcm->bic", filt_vector_to_scalar, vector))

            f1_vector_to_vector = self._mask_radial(self.vector_to_vector_f1(rbf), dij)
            filt_vector_to_vector = f1_vector_to_vector.unsqueeze(-1) * unit.unsqueeze(-2)
            cross = torch.cross(filt_vector_to_vector, vector.unsqueeze(1), dim=-1).sum(dim=2)
            vector_outputs.append(cross)

        scalar_cat = torch.cat(scalar_outputs, dim=-1)
        vector_cat = torch.cat(vector_outputs, dim=-2)

        scalar_out = self.scalar_nl(self.scalar_si(scalar_cat))
        vector_out = self.vector_nl(self.vector_si(vector_cat))
        return scalar_out, vector_out


class TFNClassifier(nn.Module):
    def __init__(self, num_classes: int, cfg: dict):
        super().__init__()
        layer_dims = list(cfg["layer_dims"])
        if len(layer_dims) != 4 or int(layer_dims[0]) != 1:
            raise ValueError("TFN shape-classification layout must use layer_dims=[1, w1, w2, w3].")

        self.layer_dims = [int(v) for v in layer_dims]
        self.rbf_low = float(cfg["rbf_low"])
        self.rbf_high = float(cfg["rbf_high"])
        self.rbf_count = int(cfg["rbf_count"])
        self.radial_hidden = int(cfg.get("radial_hidden", self.rbf_count))

        self.embed = ScalarEmbedding(output_dim=self.layer_dims[0])

        modules = []
        scalar_in = self.layer_dims[0]
        vector_in = 0
        for width in self.layer_dims[1:]:
            modules.append(
                TFNModuleL01(
                    scalar_in=scalar_in,
                    vector_in=vector_in,
                    scalar_out=int(width),
                    vector_out=int(width),
                    rbf_count=self.rbf_count,
                    rbf_low=self.rbf_low,
                    rbf_high=self.rbf_high,
                    radial_hidden=self.radial_hidden,
                )
            )
            scalar_in = int(width)
            vector_in = int(width)
        self.modules_l01 = nn.ModuleList(modules)

        self.classifier = nn.Linear(int(self.layer_dims[-1]), int(num_classes), bias=True)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        B, N, _ = x.shape
        scalar = self.embed(B, N, x.device, x.dtype)
        vector = None
        for layer in self.modules_l01:
            scalar, vector = layer(x, scalar, vector)
        pooled = scalar.mean(dim=1)
        out = self.classifier(pooled)
        return out


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
        self.np_gen = np_gen if np_gen is not None else np.random.default_rng(0)
        self.np_rs = np_rs if np_rs is not None else np.random.RandomState(0)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        pointcloud = self.points[idx].copy()
        is_training = self.split == "train"
        pointcloud = apply_data_augmentation(
            pointcloud, self.sigma, self.np_gen, self.np_rs, is_training=is_training
        )
        pointcloud = normalize_unit_sphere(pointcloud)
        # TFN expects [N, 3] input
        return torch.FloatTensor(pointcloud), torch.LongTensor([self.labels[idx]])


# ============================ Train / Eval ============================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


# ============================ Config ============================

def fixed_cfg(variant: str):
    variant = str(variant).lower()
    if variant == "light":
        return dict(
            layer_dims=[1, 4, 8, 16],
            rbf_low=0.0,
            rbf_high=3.5,
            rbf_count=4,
            radial_hidden=4,
        )
    if variant == "mid":
        return dict(
            layer_dims=[1, 8, 12, 14],
            rbf_low=0.0,
            rbf_high=3.5,
            rbf_count=4,
            radial_hidden=39,
        )
    raise ValueError("variant must be one of: light, mid")


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
        return "modelnet", f"modelnet40_5classes_{num_points}_1_fps_train700_val100_test200_new.npz", 5, 0.02
    if d == "shapenet":
        return "shapenet", f"shapenet_5classes_{num_points}_1_fps_train700_val100_test200_new.npz", 5, 0.02
    if d == "suo":
        return "SUO", f"SUO_3classes_{num_points}_1_fps_train700_val100_test200_new.npz", 3, 0.01
    raise ValueError("dataset must be one of: modelnet, shapenet, suo")


# =============================== Runner ===============================

def run_experiment(
    dataset_file,
    num_points,
    variant,
    *,
    num_classes,
    batch_size,
    epochs,
    lr,
    sigma,
    seed,
):
    rng = make_rng_pack(seed, num_points, variant)
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

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    cfg = fixed_cfg(variant)
    model = TFNClassifier(num_classes=num_classes, cfg=cfg).to(device)
    pcount = count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = 0.0
    best_epoch = -1
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate_accuracy(model, val_loader, device)
        val_loss = evaluate_loss(model, val_loader, device)

        if val_acc >= best_val:
            best_val = val_acc
            best_epoch = epoch - 1
            best_state_dict = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}

        print(f"epoch {epoch-1}/{epochs-1} | train loss : {train_loss:.4f} | val loss : {val_loss:.4f} | val accuracy : {val_acc:.4f}")
        _metrics_write({"epoch": int(epoch-1), "train_loss": float(train_loss), "val_loss": float(val_loss), "val_acc": float(val_acc)})

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    y_true = []
    y_pred = []

    model.eval()
    for data, label in test_loader:
        data = data.to(device)
        label = label.squeeze().to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        y_true.append(label.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    cm, cls_accs, overall = calculate_final_metrics(y_true, y_pred, num_classes)

    print("\n=== Results ===")
    print(f"Test Accuracy: {float(overall):.4f}")
    print("Class-wise Accuracy:")
    for i, acc in enumerate(cls_accs):
        print(f" Class {i}: {float(acc):.4f}")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("--dataset", type=str, choices=["modelnet", "shapenet", "suo"], required=True)
    parser.add_argument("--num_points", type=int, required=True)
    parser.add_argument("--variant", type=str, choices=["light", "mid"], required=True)
    args = parser.parse_args()

    base_seed = args.seed
    num_points = args.num_points
    variant = normalize_variant(args.variant)
    dataset_tag, dataset_file, num_classes, sigma = resolve_dataset(args.dataset, num_points)

    epochs = 1000
    lr = 1e-3

    _setup_run(Path(__file__).stem, base_seed, args.dataset, num_points, variant, lr=lr, epochs=epochs)
    print(f"seed={base_seed}, dataset={args.dataset}, variant={variant}, num_points={num_points}, epochs={epochs}, lr={lr}")

    HERE = Path(__file__).resolve().parent
    REPO = HERE.parent

    tag = str(dataset_tag).lower()
    if tag == "modelnet":
        dataset_file = str(REPO / "data" / "ModelNet" / dataset_file)
    elif tag == "shapenet":
        dataset_file = str(REPO / "data" / "ShapeNet" / dataset_file)
    else:
        dataset_file = str(REPO / "data" / "Sydney_Urban_Objects" / dataset_file)

    test_acc = run_experiment(
        dataset_file,
        num_points,
        variant,
        num_classes=num_classes,
        batch_size=35,
        epochs=epochs,
        lr=lr,
        sigma=sigma,
        seed=base_seed,
    )

    print(float(test_acc))


if __name__ == "__main__":
    main()
