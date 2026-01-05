#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_param_report(model, title="Model"):
    total = count_parameters(model)
    print(f"{title} trainable params: {total/1e6:.3f}M ({total:,})")
    for name, module in model.named_children():
        p = count_parameters(module)
        print(f"  - {name:12s}: {p/1e6:.3f}M ({p:,})")


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


def index_points(feats, idx):
    B, N, C = feats.shape
    K = idx.size(-1)
    idx_expand = idx.unsqueeze(-1).expand(B, N, K, C)
    feats_expand = feats.unsqueeze(1).expand(B, N, N, C)
    gathered = torch.gather(feats_expand, 2, idx_expand)
    return gathered


# ============================== Core layers ==============================

class Linear1d(nn.Module):
    def __init__(self, in_c, out_c, bias=False):
        super().__init__()
        self.fc = nn.Linear(in_c, out_c, bias=bias)

    def forward(self, x):
        B, N, C = x.shape
        y = self.fc(x.view(B * N, C)).view(B, N, -1)
        return y


class BNAct1d(nn.Module):
    def __init__(self, C, act="relu"):
        super().__init__()
        self.bn = nn.BatchNorm1d(C)
        self.act = nn.ReLU(inplace=True) if act == "relu" else nn.GELU()

    def forward(self, x):
        B, N, C = x.shape
        y = self.bn(x.reshape(B * N, C)).reshape(B, N, C)
        return self.act(y)


class ResidualMLPBlock(nn.Module):
    def __init__(self, in_c, out_c, ratio=2, kind="expand", act="relu"):
        super().__init__()
        self.in_c, self.out_c = in_c, out_c
        self.kind = kind
        self.ratio = max(1, ratio)

        if kind == "expand":
            hidden = max(1, in_c * self.ratio)
        elif kind == "bottleneck":
            hidden = max(1, out_c // self.ratio)
        else:
            raise ValueError("kind must be 'expand' or 'bottleneck'")

        self.fc1 = Linear1d(in_c, hidden, bias=False)
        self.bn1 = BNAct1d(hidden, act=act)
        self.fc2 = Linear1d(hidden, out_c, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)

        self.proj = None
        if in_c != out_c:
            self.proj = Linear1d(in_c, out_c, bias=False)

        self.act = nn.ReLU(inplace=True) if act == "relu" else nn.GELU()

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.fc2(out)

        B, N, C = out.shape
        out = self.bn2(out.reshape(B * N, C)).reshape(B, N, C)
        out = out + identity
        out = self.act(out)
        return out


class GAPooling(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x, idx):
        gathered = index_points(x, idx)
        pooled = gathered.mean(dim=2)
        return pooled


class GAEncode(nn.Module):
    def __init__(self, in_c, k):
        super().__init__()
        self.k = k
        self.fc = Linear1d(in_c, in_c, bias=False)

    def forward(self, xyz_B3N, feats_BNC):
        idx = knn(xyz_B3N, k=self.k)
        pooled = GAPooling(self.k)(feats_BNC, idx)
        out = self.fc(pooled)
        return out


class ContextFusion(nn.Module):
    def __init__(self, in_c, act="relu", eps=1e-5):
        super().__init__()
        self.in_c = in_c
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, 1, 2 * in_c))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 2 * in_c))
        self.align = Linear1d(2 * in_c, in_c, bias=False)
        self.act = nn.ReLU(inplace=True) if act == "relu" else nn.GELU()

    def forward(self, x):
        B, N, C = x.shape
        if N <= 1:
            return x

        F_center = x.unsqueeze(2).expand(-1, -1, N, -1)
        F_neighbors = x.unsqueeze(1).expand(-1, N, -1, -1)
        diff = F_neighbors - F_center
        var = diff.var(dim=2, unbiased=False, keepdim=True)
        F_tilde = diff / torch.sqrt(var + self.eps)

        Fc_K = torch.cat([F_tilde, F_center], dim=-1)
        Fc_K = Fc_K * self.gamma + self.beta
        weights = torch.softmax(Fc_K, dim=2)
        Fc_C = (weights * Fc_K).sum(dim=2)

        tokens = self.align(Fc_C)
        return self.act(tokens)


class PointMLPStageSimpleLight(nn.Module):
    def __init__(self, in_c, out_c, ratio_expand=2, ratio_bottleneck=2, act="relu"):
        super().__init__()
        self.pre = ResidualMLPBlock(in_c, in_c, ratio=ratio_expand, kind="bottleneck", act=act)
        self.fuse = ContextFusion(in_c, act=act)
        self.post = ResidualMLPBlock(in_c, out_c, ratio=ratio_bottleneck, kind="expand", act=act)

    def forward(self, xyz_B3N, feats_BNC):
        x = self.pre(feats_BNC)
        x = self.fuse(x)
        x = self.post(x)
        return x


class PointMLPStageSimpleMid(nn.Module):
    def __init__(self, in_c, out_c, ratio_expand=4, ratio_bottleneck=2, act="relu"):
        super().__init__()
        self.pre = ResidualMLPBlock(in_c, in_c, ratio=ratio_bottleneck, kind="bottleneck", act=act)
        self.fuse = ContextFusion(in_c, act=act)
        self.post = ResidualMLPBlock(in_c, out_c, ratio=ratio_expand, kind="expand", act=act)

    def forward(self, xyz_B3N, feats_BNC):
        x = self.pre(feats_BNC)
        x = self.fuse(x)
        x = self.post(x)
        return x


# ============================ PointMLP ============================

class CompactPointMLP(nn.Module):
    def __init__(self, num_classes, k, variant):
        super().__init__()
        self.k = k

        if variant == "light":
            dims = (6, 10, 12)
            ratio_expand = 2
            ratio_bottleneck = 2
            Stage = PointMLPStageSimpleLight
            num_stages = 2
            head_out = num_classes
            head_hidden = 8
        else:
            dims = (8, 16, 32)
            ratio_expand = 4
            ratio_bottleneck = 2
            Stage = PointMLPStageSimpleMid
            num_stages = 3
            head_out = num_classes
            head_hidden = 16

        self.stem_fc = nn.Linear(3, dims[0], bias=False)
        self.stem_bn = nn.BatchNorm1d(dims[0])

        self.ga = GAEncode(dims[0], k)

        if num_stages == 2:
            self.stage1 = Stage(dims[0], dims[1], ratio_expand, ratio_bottleneck, "relu")
            self.stage2 = Stage(dims[1], dims[2], ratio_expand, ratio_bottleneck, "relu")
            self.stage3 = None
        else:
            self.stage1 = Stage(dims[0], dims[0], ratio_expand, ratio_bottleneck, "relu")
            self.stage2 = Stage(dims[0], dims[1], ratio_expand, ratio_bottleneck, "relu")
            self.stage3 = Stage(dims[1], dims[2], ratio_expand, ratio_bottleneck, "relu")

        self.head_fc1 = nn.Linear(dims[2], head_hidden, bias=False)
        self.head_bn1 = nn.BatchNorm1d(head_hidden)

        if variant == "light":
            self.head_fc2 = nn.Linear(head_hidden, head_out)
            self.head_bn2 = None
            self.head_fc3 = None
        else:
            self.head_fc2 = nn.Linear(head_hidden, 8)
            self.head_bn2 = nn.BatchNorm1d(8)
            self.head_fc3 = nn.Linear(8, head_out)

    def forward(self, x):
        B, C, N = x.shape

        stem = self.stem_fc(x.transpose(2, 1)).view(B * N, -1)
        stem = self.stem_bn(stem).view(B, N, -1)
        stem = F.relu(stem, inplace=True)

        feats = self.ga(x, stem)

        feats = self.stage1(x, feats)
        feats = self.stage2(x, feats)
        if self.stage3 is not None:
            feats = self.stage3(x, feats)

        feats = feats.max(dim=1).values

        y = self.head_fc1(feats)
        y = self.head_bn1(y)
        y = F.relu(y, inplace=True)
        y = self.head_fc2(y)
        if self.head_fc3 is not None:
            y = self.head_bn2(y)
            y = F.relu(y, inplace=True)
            y = self.head_fc3(y)
        return y


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

        if len(pointcloud) < self.num_points:
            indices = self.np_gen.choice(len(pointcloud), self.num_points, replace=True)
        else:
            indices = self.np_gen.choice(len(pointcloud), self.num_points, replace=False)
        pointcloud = pointcloud[indices]

        is_training = self.split == "train"
        pointcloud = apply_data_augmentation(pointcloud, self.sigma, self.np_gen, self.np_rs, is_training=is_training)
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
    *,
    num_classes,
    batch_size,
    epochs,
    lr,
    dropout,
    sigma,
    seed,
    variant,
):
    rng = make_rng_pack(seed)
    device = rng["device"]
    torch_gen = rng["torch_gen_cuda"] or rng["torch_gen_cpu"]

    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"데이터셋 파일 '{dataset_file}'이 존재하지 않습니다.")

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
        k = max(1, num_points - 1)
        print(f"k값을 {k}로 조정했습니다 (num_points={num_points})")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model = CompactPointMLP(num_classes=num_classes, k=k, variant=variant).to(device)
    print_param_report(model, title="CompactPointMLP")

    optimizer = Adam(model.parameters(), lr=lr)

    best_val = 0.0
    best_epoch = -1
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        train_acc = evaluate_accuracy(model, train_loader, device)
        val_acc = evaluate_accuracy(model, val_loader, device)

        if val_acc >= best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state_dict = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
            torch.save(best_state_dict, "best_compact_pointmlp_by_val.pth")

        print(
            f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
            f"Best Val: {best_val:.4f} @ {best_epoch}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    test_acc = evaluate_accuracy(model, test_loader, device)

    print("\n==============================")
    print(f"[BEST by Val] epoch={best_epoch}, val_acc={best_val:.4f}")
    print(f"Test Acc (evaluated ONCE with best-val params) = {test_acc:.4f}")
    print("==============================\n")

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
    parser.add_argument("--num_qubit", type=int, required=True)
    parser.add_argument("--variant", type=str, choices=["light", "mid"], required=True)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()

    base_seed = args.seed
    num_qubit = args.num_qubit
    if num_qubit % 2 != 0:
        raise ValueError("num_qubit must be even")

    num_points = num_qubit // 2
    variant = normalize_variant(args.variant)
    dataset_tag, dataset_file, num_classes, sigma = resolve_dataset(args.dataset, num_points)
    k = args.k

    print(f"Using seed={base_seed}")
    print(
        f"dataset={dataset_tag}, variant={variant}, num_qubit={num_qubit}, num_points={num_points}, k={k}"
    )

    test_acc = run_experiment(
        dataset_file,
        num_points=num_points,
        k=k,
        num_classes=num_classes,
        batch_size=35,
        epochs=1000,
        lr=0.001,
        dropout=0.0,
        sigma=sigma,
        seed=base_seed,
        variant=variant,
    )

    print(float(test_acc))


if __name__ == "__main__":
    main()
