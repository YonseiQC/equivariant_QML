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
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import special_ortho_group
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix

# ============================================================
# Baseline-compatible VN-PointNet classifier
# ------------------------------------------------------------
# What is kept identical to the official VNN repo:
# - VNLinear / VNLinearLeakyReLU / VNBatchNorm / VNMaxPool / VNStdFeature
# - get_graph_feature_cross
# - PointNetEncoder flow:
#     graph feature -> conv_pos -> local pool -> conv1
#     -> optional feature transform -> conv2 -> conv3+bn3
#     -> concat global mean -> std_feature -> flatten -> global max
# - Classification head structure:
#     fc1 -> bn1 -> relu -> fc2 -> bn2 -> relu -> fc3
#
# What is changed:
# - Only channel widths / MLP widths are reduced to hit parameter budgets.
# - Pooling is set back to mean, following the paper's training setting and
#   the repo's default training argument.
# - The model returns raw logits and the training loop uses F.cross_entropy,
#   matching the baseline loss pairing.
# - CUDA-hardcoded device lines in the original graph utility are made
#   device-safe so the script runs on CPU deterministically.
# ============================================================

_METRICS_FH = None
EPSILON = 1e-8
VN_EPS = 1e-6

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
                "k": None if k is None else int(k),
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
    subseed = make_subseed(base_seed, num_points, dataset_tag)

    py_rng = random.Random(subseed)
    np_gen = np.random.default_rng(subseed)
    np_rs = np.random.RandomState(subseed)

    torch_gen_cpu = torch.Generator(device="cpu").manual_seed(subseed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch_gen_cuda = torch.Generator(device="cuda").manual_seed(subseed)
        torch.cuda.manual_seed(subseed)
        torch.cuda.manual_seed_all(subseed)
    else:
        device = torch.device("cpu")
        torch_gen_cuda = None

    random.seed(subseed)
    np.random.seed(subseed)
    torch.manual_seed(subseed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    return dict(
        seed=subseed,
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


# ============================================================
# Official VN layers (structure kept)
# source-aligned with models/vn_layers.py
# ============================================================

class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        return self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super().__init__()
        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = float(negative_slope)

    def forward(self, x):
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1.0 - self.negative_slope) * (
            mask * x + (1.0 - mask) * (x - (dotprod / (d_norm_sq + VN_EPS)) * d)
        )
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super().__init__()
        self.dim = int(dim)
        if self.dim in (3, 4):
            self.bn = nn.BatchNorm1d(num_features)
        elif self.dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
        else:
            raise ValueError(f"Unsupported VNBatchNorm dim={dim}")

    def forward(self, x):
        norm = torch.norm(x, dim=2) + VN_EPS
        norm_bn = self.bn(norm)
        return x / norm.unsqueeze(2) * norm_bn.unsqueeze(2)


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2):
        super().__init__()
        self.dim = int(dim)
        self.negative_slope = float(negative_slope)

        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)

        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        p = self.batchnorm(p)

        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (p * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * p + (1.0 - self.negative_slope) * (
            mask * p + (1.0 - mask) * (p - (dotprod / (d_norm_sq + VN_EPS)) * d)
        )
        return x_out


class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]

        mesh = torch.meshgrid(
            *[torch.arange(s, device=x.device) for s in x.size()[:-1]],
            indexing="ij",
        )
        x_max = x[(*mesh, idx)]
        return x_max


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2):
        super().__init__()
        self.dim = int(dim)
        self.normalize_frame = bool(normalize_frame)

        self.vn1 = VNLinearLeakyReLU(
            in_channels,
            in_channels // 2,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
        )
        self.vn2 = VNLinearLeakyReLU(
            in_channels // 2,
            in_channels // 4,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            negative_slope=negative_slope,
        )
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels // 4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels // 4, 3, bias=False)

    def forward(self, x):
        z0 = self.vn1(x)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)

        if self.normalize_frame:
            v1 = z0[:, 0, :]
            v1_norm = torch.sqrt((v1 * v1).sum(1, keepdim=True))
            u1 = v1 / (v1_norm + VN_EPS)

            v2 = z0[:, 1, :]
            v2 = v2 - (v2 * u1).sum(1, keepdim=True) * u1
            v2_norm = torch.sqrt((v2 * v2).sum(1, keepdim=True))
            u2 = v2 / (v2_norm + VN_EPS)

            u3 = torch.cross(u1, u2, dim=1)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)

        if self.dim == 4:
            x_std = torch.einsum("bijm,bjkm->bikm", x, z0)
        elif self.dim == 3:
            x_std = torch.einsum("bij,bjk->bik", x, z0)
        elif self.dim == 5:
            x_std = torch.einsum("bijmn,bjkmn->bikmn", x, z0)
        else:
            raise ValueError(f"Unsupported VNStdFeature dim={self.dim}")

        return x_std, z0


# ============================================================
# Official graph feature utility (structure kept)
# source-aligned with models/utils/vn_dgcnn_util.py
# Only change: device-safe instead of forced CUDA.
# ============================================================

def knn(x, k):
    inner = -2.0 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature_cross(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k=k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x_center = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x_center, dim=-1)

    feature = torch.cat((feature - x_center, x_center, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()
    return feature


# ============================================================
# Official VN-PointNet encoder/classifier (structure kept)
# source-aligned with:
# - models/vn_pointnet.py
# - models/vn_pointnet_cls.py
# Only width arguments are reduced.
# Changed: returns raw logits (no log_softmax) for F.cross_entropy.
# ============================================================

class STNkd(nn.Module):
    def __init__(self, args, d):
        super().__init__()
        self.args = args
        self.conv1 = VNLinearLeakyReLU(d, args.stn_c1, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(args.stn_c1, args.stn_c2, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(args.stn_c2, args.stn_c3, dim=4, negative_slope=0.0)
        self.fc1 = VNLinearLeakyReLU(args.stn_c3, args.stn_fc1, dim=3, negative_slope=0.0)
        self.fc2 = VNLinearLeakyReLU(args.stn_fc1, args.stn_fc2, dim=3, negative_slope=0.0)
        self.fc3 = VNLinear(args.stn_fc2, d)

        if args.pooling == "max":
            self.pool = VNMaxPool(args.stn_c3)
        elif args.pooling == "mean":
            self.pool = mean_pool
        else:
            raise ValueError(f"Unsupported pooling={args.pooling}")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x) if self.pool is not mean_pool else self.pool(x, dim=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, args, global_feat=True, feature_transform=True, channel=3):
        super().__init__()
        self.args = args
        self.n_knn = int(args.n_knn)

        self.conv_pos = VNLinearLeakyReLU(3, args.enc_cpos, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(args.enc_cpos, args.enc_c1, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(args.enc_c1 * 2, args.enc_c2, dim=4, negative_slope=0.0)
        self.conv3 = VNLinear(args.enc_c2, args.enc_c3)
        self.bn3 = VNBatchNorm(args.enc_c3, dim=4)
        self.std_feature = VNStdFeature(args.enc_c3 * 2, dim=4, normalize_frame=False, negative_slope=0.0)

        if args.pooling == "max":
            self.pool = VNMaxPool(args.enc_c1)
        elif args.pooling == "mean":
            self.pool = mean_pool
        else:
            raise ValueError(f"Unsupported pooling={args.pooling}")

        self.global_feat = bool(global_feat)
        self.feature_transform = bool(feature_transform)
        if self.feature_transform:
            self.fstn = STNkd(args, d=args.enc_c1)

    def forward(self, x):
        B, D, N = x.size()

        x = x.unsqueeze(1)
        feat = get_graph_feature_cross(x, k=self.n_knn)
        x = self.conv_pos(feat)
        x = self.pool(x) if self.pool is not mean_pool else self.pool(x, dim=-1)
        x = self.conv1(x)

        if self.feature_transform:
            x_global = self.fstn(x).unsqueeze(-1).repeat(1, 1, 1, N)
            x = torch.cat((x, x_global), 1)

        pointfeat = x
        x = self.conv2(x)
        x = self.bn3(self.conv3(x))

        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        x = x.view(B, -1, N)
        x = torch.max(x, -1, keepdim=False)[0]

        trans_feat = None
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(B, -1, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class VNPointNetCls(nn.Module):
    def __init__(self, args, num_class=40, normal_channel=False):
        super().__init__()
        self.args = args
        channel = 6 if normal_channel else 3

        self.feat = PointNetEncoder(args, global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(args.enc_c3 * 6, args.cls_fc1)
        self.fc2 = nn.Linear(args.cls_fc1, args.cls_fc2)
        self.fc3 = nn.Linear(args.cls_fc2, num_class)
        self.bn1 = nn.BatchNorm1d(args.cls_fc1)
        self.bn2 = nn.BatchNorm1d(args.cls_fc2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # baseline-compatible input: [B, N, 3]
        x = x.transpose(1, 2).contiguous()  # -> [B, 3, N]
        x, trans, trans_feat = self.feat(x)

        # Official classifier head order kept.
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        # Return raw logits (no log_softmax) for F.cross_entropy
        return x


# ============================ Config ============================

def build_args(variant: str, n_knn: int):
    variant = variant.lower()

    if variant == "light":
        cfg = dict(
            n_knn=int(n_knn),
            pooling="mean",
            enc_cpos=2,
            enc_c1=2,
            enc_c2=4,
            enc_c3=11,
            stn_c1=2,
            stn_c2=4,
            stn_c3=10,
            stn_fc1=5,
            stn_fc2=2,
            cls_fc1=7,
            cls_fc2=5,
        )
        return SimpleNamespace(**cfg), cfg

    if variant == "mid":
        cfg = dict(
            n_knn=int(n_knn),
            pooling="mean",
            enc_cpos=4,
            enc_c1=4,
            enc_c2=8,
            enc_c3=25,
            stn_c1=4,
            stn_c2=8,
            stn_c3=24,
            stn_fc1=12,
            stn_fc2=6,
            cls_fc1=16,
            cls_fc2=14,
        )
        return SimpleNamespace(**cfg), cfg

    raise ValueError("variant must be one of: light, mid")


def make_model(num_classes: int, variant: str, n_knn: int):
    args, cfg = build_args(variant, n_knn)
    model = VNPointNetCls(args, num_class=num_classes, normal_channel=False)
    return model, cfg


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
        # VN-PointNet expects [B, N, 3] input (transposes internally)
        return torch.FloatTensor(pointcloud), torch.LongTensor([self.labels[idx]])


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


# ============================ Helpers ============================

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


def invariance_sanity_check(variant: str, n_knn: int, num_classes: int = 5, num_points: int = 6, seed: int = 0):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    model, cfg = make_model(num_classes=num_classes, variant=variant, n_knn=n_knn)
    model.eval()

    pts = rng.randn(num_points, 3).astype(np.float32)
    pts = normalize_unit_sphere(pts)

    perm = rng.permutation(num_points)
    R = special_ortho_group.rvs(3, random_state=rng).astype(np.float32)

    x = torch.from_numpy(pts[None, ...]).float()
    x_perm = torch.from_numpy(pts[perm][None, ...]).float()
    x_rot = torch.from_numpy((pts @ R.T)[None, ...]).float()
    x_both = torch.from_numpy(((pts[perm]) @ R.T)[None, ...]).float()

    with torch.no_grad():
        y = model(x)
        y_perm = model(x_perm)
        y_rot = model(x_rot)
        y_both = model(x_both)

    return {
        "variant": variant,
        "param_count": count_parameters(model),
        "max_abs_perm_diff": float((y - y_perm).abs().max().item()),
        "max_abs_rot_diff": float((y - y_rot).abs().max().item()),
        "max_abs_perm_rot_diff": float((y - y_both).abs().max().item()),
    }


# =============================== Runner ===============================

def run_experiment(
    dataset_file,
    num_points,
    variant,
    n_knn,
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

    model, cfg = make_model(num_classes=num_classes, variant=variant, n_knn=n_knn)
    model = model.to(device)
    pcount = count_parameters(model)
    print(f"param_count: {pcount}")

    sanity = invariance_sanity_check(variant=variant, n_knn=n_knn, num_classes=num_classes, num_points=max(6, num_points), seed=seed)
    print(f"invariance_sanity_check: {sanity}")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("--dataset", type=str, choices=["modelnet", "shapenet", "suo"], required=True)
    parser.add_argument("--num_points", type=int, required=True)
    parser.add_argument("--variant", type=str, choices=["light", "mid"], required=True)
    parser.add_argument("--n_knn", type=int, required=True)
    parser.add_argument("--sanity_only", action="store_true")
    args = parser.parse_args()

    base_seed = args.seed
    num_points = args.num_points
    variant = normalize_variant(args.variant)
    dataset_tag, dataset_file, num_classes, sigma = resolve_dataset(args.dataset, num_points)

    if args.sanity_only:
        print(invariance_sanity_check(variant, n_knn=args.n_knn, num_classes=num_classes, num_points=max(6, num_points), seed=base_seed))
        return

    epochs = 1000
    lr = 1e-2

    _setup_run(Path(__file__).stem, base_seed, args.dataset, num_points, variant, lr=lr, epochs=epochs, k=args.n_knn)
    print(f"seed={base_seed}, dataset={args.dataset}, variant={variant}, num_points={num_points}, n_knn={args.n_knn}, epochs={epochs}, lr={lr}")

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
        args.n_knn,
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
