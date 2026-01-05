#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import special_ortho_group
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix


# --------------------------- Determinism & RNG pack ---------------------------

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
    )


# --------------------------- Utils ---------------------------

def random_sample_points(points, num_sample, np_gen):
    N = points.shape[0]
    if N == num_sample:
        return points
    indices = np_gen.choice(N, num_sample, replace=False)
    return points[indices]


def normalize_unit_sphere(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    m = np.max(np.sqrt(np.sum(points**2, axis=1))) + 1e-12
    return points / m


def random_3d_rotation(np_rs):
    return special_ortho_group.rvs(3, random_state=np_rs)


def apply_3d_rotation(points, np_rs):
    R = random_3d_rotation(np_rs)
    return points @ R.T


def add_jitter(points, np_gen, sigma):
    noise = np_gen.normal(0.0, sigma, size=points.shape)
    return points + noise


def apply_permutation(points, np_rs):
    perm_indices = np_rs.permutation(points.shape[0])
    return points[perm_indices]


def apply_data_augmentation(points, is_training, np_gen, np_rs, sigma):
    if not is_training:
        return points
    pts = add_jitter(points, np_gen, sigma=sigma)
    pts = apply_3d_rotation(pts, np_rs)
    pts = apply_permutation(pts, np_rs)
    return pts


# --------------------------- Dataset ---------------------------

class PointNetDataset(Dataset):
    def __init__(self, points, labels, num_points, is_training=False, normalize=False, sigma=0.02, np_gen=None, np_rs=None):
        self.points = points
        self.labels = labels
        self.num_points = num_points
        self.is_training = is_training
        self.normalize = normalize
        self.sigma = sigma
        self.np_gen = np_gen
        self.np_rs = np_rs

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        pts = self.points[idx].astype(np.float32)
        lbl = self.labels[idx].astype(np.int64)

        if self.normalize:
            pts = normalize_unit_sphere(pts)

        if pts.shape[0] != self.num_points:
            pts = random_sample_points(pts, self.num_points, self.np_gen)

        pts = apply_data_augmentation(pts, is_training=self.is_training, np_gen=self.np_gen, np_rs=self.np_rs, sigma=self.sigma)
        return torch.from_numpy(pts).float(), torch.tensor(lbl).long()


# --------------------------- T-Net orthogonality regularizer ---------------------------

def tnet_ortho_reg(A: torch.Tensor) -> torch.Tensor:
    B, k, _ = A.shape
    I = torch.eye(k, device=A.device).unsqueeze(0).expand(B, -1, -1)
    diff = I - A.bmm(A.transpose(2, 1))
    return torch.mean(torch.sum(diff * diff, dim=(1, 2)))


# --------------------------- Model ---------------------------

class TNet1L(nn.Module):
    def __init__(self, k=3, c=8, fc_hidden=10):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(k, c, 1, bias=True)
        self.bn = nn.BatchNorm1d(c)
        self.fc1 = nn.Linear(c, fc_hidden, bias=True)
        self.bn1 = nn.BatchNorm1d(fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, k * k, bias=True)

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.bn(self.conv(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(B, -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        I = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(B, 1)
        x = x + I
        return x.view(B, self.k, self.k)


class PointNetLight(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.input_transform = TNet1L(k=3, c=8, fc_hidden=10)
        self.conv1 = nn.Conv1d(3, 9, 1, bias=True)
        self.bn1 = nn.BatchNorm1d(9)
        self.feature_transform = TNet1L(k=9, c=9, fc_hidden=10)
        self.fc_mid = nn.Linear(9, 16, bias=True)
        self.fc = nn.Linear(16, num_classes, bias=True)

    def forward(self, x):
        B = x.size(0)
        T_in = self.input_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, T_in).transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        T_feat = self.feature_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, T_feat).transpose(2, 1)
        x = torch.max(x, 2, keepdim=True)[0].view(B, -1)
        x = F.relu(self.fc_mid(x))
        x = self.fc(x)
        return x, T_in, T_feat


class PointNetMid(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.input_transform = TNet1L(k=3, c=8, fc_hidden=10)

        self.conv1 = nn.Conv1d(3, 4, 1, bias=True)
        self.bn1 = nn.BatchNorm1d(4)
        self.conv2 = nn.Conv1d(4, 8, 1, bias=True)
        self.bn2 = nn.BatchNorm1d(8)

        self.feature_transform = TNet1L(k=8, c=16, fc_hidden=20)

        self.conv3 = nn.Conv1d(8, 32, 1, bias=True)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(32, 120, 1, bias=True)
        self.bn4 = nn.BatchNorm1d(120)
        self.conv5 = nn.Conv1d(120, 8, 1, bias=True)
        self.bn5 = nn.BatchNorm1d(8)

        self.fc1 = nn.Linear(8, num_classes, bias=True)

    def forward(self, x):
        B = x.size(0)
        T_in = self.input_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, T_in).transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        T_feat = self.feature_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, T_feat).transpose(2, 1)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = torch.max(x, 2, keepdim=True)[0].view(B, -1)
        x = self.fc1(x)
        return x, T_in, T_feat


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------- Metrics ---------------------------

def calculate_final_metrics(y_true, y_pred, num_classes):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    overall_acc = np.trace(cm) / np.sum(cm)
    class_accuracies = []
    for i in range(num_classes):
        denom = cm[i].sum()
        class_accuracies.append((cm[i, i] / denom) if denom > 0 else 0.0)
    return cm, class_accuracies, overall_acc


# --------------------------- Train ---------------------------

def train_model_with_val(
    model,
    train_loader,
    val_loader,
    test_loader,
    num_classes,
    num_epochs,
    learning_rate,
    weight_decay,
    device,
    lambda_reg: float = 1e-3,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_acc = -1.0
    best_state = None
    best_epoch = -1

    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
    print(f"Training with validation selection for {num_epochs} epochs")

    for epoch in range(num_epochs):
        model.train()
        running_train = 0.0
        for pts, lbl in train_loader:
            pts, lbl = pts.to(device), lbl.to(device)
            pts = pts.transpose(2, 1)
            optimizer.zero_grad()
            pred, T_in, T_feat = model(pts)
            ce = criterion(pred, lbl)
            reg = tnet_ortho_reg(T_in) + tnet_ortho_reg(T_feat)
            loss = ce + lambda_reg * reg
            loss.backward()
            optimizer.step()
            running_train += loss.item()

        train_loss = running_train / max(1, len(train_loader))

        model.eval()
        running_val = 0.0
        all_pred, all_lbl = [], []
        with torch.no_grad():
            for pts, lbl in val_loader:
                pts, lbl = pts.to(device), lbl.to(device)
                pts = pts.transpose(2, 1)
                pred, _, _ = model(pts)
                loss = criterion(pred, lbl)
                running_val += loss.item()
                _, p = torch.max(pred, 1)
                all_pred.extend(p.cpu())
                all_lbl.extend(lbl.cpu())
        val_loss = running_val / max(1, len(val_loader))
        all_pred = torch.tensor(all_pred)
        all_lbl = torch.tensor(all_lbl)
        val_acc = (all_pred == all_lbl).float().mean().item()

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[Epoch {epoch:3d}] TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
            f"ValAcc {val_acc:.4f} (Best {best_val_acc:.4f} @ {best_epoch})"
        )

    print("\n=== Evaluate on TEST with Best-Validation Checkpoint ===")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    all_pred, all_lbl = [], []
    with torch.no_grad():
        for pts, lbl in test_loader:
            pts, lbl = pts.to(device), lbl.to(device)
            pts = pts.transpose(2, 1)
            pred, _, _ = model(pts)
            _, p = torch.max(pred, 1)
            all_pred.extend(p.cpu())
            all_lbl.extend(lbl.cpu())
    all_pred = torch.tensor(all_pred)
    all_lbl = torch.tensor(all_lbl)

    cm, class_acc, overall_acc = calculate_final_metrics(all_lbl, all_pred, num_classes)
    print(f"Test Overall Accuracy: {overall_acc:.4f}")
    print("Class-wise Accuracies:")
    for i, a in enumerate(class_acc):
        print(f"  Class {i}: {a:.4f}")
    print("\nConfusion Matrix:\n", cm)

    return best_val_acc, overall_acc


def main_with_val(data_filename, num_points, num_epochs, learning_rate, batch_size, weight_decay, seed, variant, sigma):
    rng = make_rng_pack(seed)
    device = rng["device"]
    torch_gen = rng["torch_gen_cuda"] or rng["torch_gen_cpu"]

    print(f"Loading dataset: {data_filename}")
    try:
        data = np.load(data_filename)
        train_points = data["train_dataset_x"].astype(np.float32)
        train_labels = data["train_dataset_y"].astype(np.int64)
        val_points = data["val_dataset_x"].astype(np.float32)
        val_labels = data["val_dataset_y"].astype(np.int64)
        test_points = data["test_dataset_x"].astype(np.float32)
        test_labels = data["test_dataset_y"].astype(np.int64)
    except Exception as e:
        print("NPZ load failed:", e)
        return None

    train_dataset = PointNetDataset(
        train_points,
        train_labels,
        num_points=num_points,
        is_training=True,
        normalize=False,
        sigma=sigma,
        np_gen=rng["np_gen"],
        np_rs=rng["np_rs"],
    )
    val_dataset = PointNetDataset(
        val_points,
        val_labels,
        num_points=num_points,
        is_training=False,
        normalize=False,
        sigma=sigma,
        np_gen=rng["np_gen"],
        np_rs=rng["np_rs"],
    )
    test_dataset = PointNetDataset(
        test_points,
        test_labels,
        num_points=num_points,
        is_training=False,
        normalize=False,
        sigma=sigma,
        np_gen=rng["np_gen"],
        np_rs=rng["np_rs"],
    )

    print(f"Sizes | Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")

    train_loader = fixed_loader(train_dataset, batch_size=batch_size, shuffle=True, torch_gen=torch_gen)
    val_loader = fixed_loader(val_dataset, batch_size=batch_size, shuffle=False, torch_gen=torch_gen)
    test_loader = fixed_loader(test_dataset, batch_size=batch_size, shuffle=False, torch_gen=torch_gen)

    num_classes = len(np.unique(train_labels))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if variant == "light":
        model = PointNetLight(num_classes=num_classes).to(device)
    else:
        model = PointNetMid(num_classes=num_classes).to(device)

    s_pts, s_lbl = train_dataset[0]
    print(f"Sample point cloud shape: {s_pts.shape} (N={num_points})  |  Label: {s_lbl}")
    print(f"Total parameters (C={num_classes}): {count_parameters(model):,}")

    best_val_acc, test_acc = train_model_with_val(
        model,
        train_loader,
        val_loader,
        test_loader,
        num_classes,
        num_epochs,
        learning_rate,
        weight_decay,
        device,
        lambda_reg=1e-3,
    )
    print("\n===== SUMMARY =====")
    print(f"Best Val Accuracy : {best_val_acc:.4f}")
    print(f"Test Accuracy @BestVal : {test_acc:.4f}")
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
        return "modelnet", f"modelnet40_5classes_{num_points}_1_fps_train700_val100_test200_new.npz", 0.02
    if d == "shapenet":
        return "shapenet", f"shapenet_5classes_{num_points}_1_fps_train700_val100_test200_new.npz", 0.02
    if d == "suo":
        return "SUO", f"SUO_3classes_{num_points}_1_fps_train700_val100_test200_new.npz", 0.01
    raise ValueError("dataset must be one of: modelnet, shapenet, suo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("--dataset", type=str, choices=["modelnet", "shapenet", "suo"], required=True)
    parser.add_argument("--num_qubit", type=int, required=True)
    parser.add_argument("--variant", type=str, choices=["light", "mid"], required=True)
    args = parser.parse_args()

    base_seed = args.seed
    num_qubit = args.num_qubit
    if num_qubit % 2 != 0:
        raise ValueError("num_qubit must be even")

    num_points = num_qubit // 2
    variant = normalize_variant(args.variant)
    dataset_tag, data_filename, sigma = resolve_dataset(args.dataset, num_points)

    print(f"Using seed={base_seed}")
    print(
        f"dataset={dataset_tag}, variant={variant}, num_qubit={num_qubit}, num_points={num_points}, sigma={sigma}"
    )

    acc = main_with_val(
        data_filename,
        num_points,
        num_epochs=1000,
        learning_rate=0.001,
        batch_size=35,
        weight_decay=0.0,
        seed=base_seed,
        variant=variant,
        sigma=sigma,
    )

    print(float(acc))


if __name__ == "__main__":
    main()
