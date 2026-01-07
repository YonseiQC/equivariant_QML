#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from scipy.stats import special_ortho_group

def make_rng_pack(seed: int):
    py_rng = random.Random(seed)
    np_gen = np.random.default_rng(seed)
    np_rs  = np.random.RandomState(seed)
    torch_gen_cpu = torch.Generator(device='cpu').manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch_gen_cuda = torch.Generator(device='cuda').manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        device = torch.device('cpu')
        torch_gen_cuda = None
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return dict(seed=seed, py_rng=py_rng, np_gen=np_gen, np_rs=np_rs,
                torch_gen_cpu=torch_gen_cpu, torch_gen_cuda=torch_gen_cuda,
                device=device)

def fixed_loader(dataset, batch_size, shuffle, torch_gen):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=False, generator=torch_gen, num_workers=0, pin_memory=True)

def normalize_unit_sphere(points):
    pts = points.copy()
    centroid = np.mean(pts, axis=0)
    pts -= centroid
    r = np.max(np.sqrt(np.sum(pts**2, axis=1)))
    if r > 0: pts /= r
    return pts

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    if not is_training: return points
    pts = _add_jitter(points, sigma, np_gen)
    pts = _apply_3d_rotation(pts, np_rs)
    pts = _apply_permutation(pts, np_rs)
    return pts

def knn_indices(x, k):
    B, _, N = x.size()
    xt = x.transpose(2, 1)
    inner = -2 * torch.matmul(xt, xt.transpose(2, 1))
    xx = torch.sum(xt**2, dim=2, keepdim=True)
    dist = -xx - inner - xx.transpose(2, 1)
    eye = torch.eye(N, device=x.device).unsqueeze(0)
    dist = dist - 1e9 * eye
    idx = dist.topk(k=k, dim=-1)[1]
    return idx

class ChannelLN(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.ln = nn.LayerNorm(C)
    def forward(self, x):
        return self.ln(x.transpose(2,1)).transpose(2,1)

class PTBlock(nn.Module):
    def __init__(self, Cin, Cout, k):
        super().__init__()
        self.k = k
        self.in_proj = nn.Conv1d(Cin, Cout, 1, bias=True)
        self.q = nn.Conv1d(Cout, Cout, 1, bias=False)
        self.kv_k = nn.Conv1d(Cout, Cout, 1, bias=False)
        self.kv_v = nn.Conv1d(Cout, Cout, 1, bias=False)
        self.delta = nn.Sequential(nn.Conv2d(3, Cout, 1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(Cout, Cout, 1, bias=True))
        self.attn_logits = nn.Conv2d(Cout, Cout, 1, bias=True)
        self.ln1 = ChannelLN(Cout)
        self.ln2 = ChannelLN(Cout)
        self.ffn = nn.Sequential(nn.Conv1d(Cout, 4*Cout, 1, bias=True),
                                 nn.ReLU(inplace=True),
                                 nn.Conv1d(4*Cout, Cout, 1, bias=True))

    def forward(self, x, p):
        h = self.in_proj(x)
        B, C, N = h.size()
        k = min(self.k, max(1, N-1))
        h_n = self.ln1(h)

        idx = knn_indices(p, k=k)
        base = torch.arange(B, device=h.device)[:,None,None]*N
        flat = (idx+base).reshape(-1)

        h_t = h_n.transpose(2,1).contiguous()
        k_all = self.kv_k(h_n).transpose(2,1)
        v_all = self.kv_v(h_n).transpose(2,1)
        k_j = k_all.reshape(B*N, C)[flat,:].reshape(B,N,k,C).permute(0,3,1,2).contiguous()
        v_j = v_all.reshape(B*N, C)[flat,:].reshape(B,N,k,C).permute(0,3,1,2).contiguous()

        p_t = p.transpose(2,1).contiguous()
        p_j = p_t.reshape(B*N,3)[flat,:].reshape(B,N,k,3).permute(0,3,1,2).contiguous()
        p_i = p.unsqueeze(-1).expand(-1,-1,-1,k)
        d = self.delta(p_i - p_j)

        q_i = self.q(h_n).unsqueeze(-1).expand(-1,-1,-1,k)
        logits = self.attn_logits(q_i - k_j + d)
        a = torch.softmax(logits, dim=-1)
        y = torch.sum(a * (v_j + d), dim=-1)

        h = h + y
        h2 = self.ln2(h)
        h = h + self.ffn(h2)
        return h

class PointTransformerClassifier(nn.Module):
    def __init__(self, num_classes, widths, k, dropout):
        super().__init__()
        self.k = k
        self.stem = nn.Conv1d(3, widths[0], 1, bias=True)
        blocks = []
        for i in range(len(widths)-1):
            blocks.append(PTBlock(widths[i], widths[i+1], k))
        self.blocks = nn.ModuleList(blocks)
        C = widths[-1]
        self.head = nn.Sequential(nn.Conv1d(C, C, 1, bias=True),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(p=dropout),
                                  nn.Conv1d(C, num_classes, 1, bias=True))
    def forward(self, x):
        p = x
        f = self.stem(x)
        for blk in self.blocks:
            f = blk(f, p)
        logits = self.head(f).mean(dim=-1)
        return logits

class PointCloudDataset(Dataset):
    def __init__(self, data_path, split='train', sigma=0.02, np_gen=None, np_rs=None):
        with np.load(data_path) as data:
            if split == 'train':
                self.points = data['train_dataset_x']
                self.labels = data['train_dataset_y']
            elif split == 'val':
                self.points = data['val_dataset_x']
                self.labels = data['val_dataset_y']
            elif split == 'test':
                self.points = data['test_dataset_x']
                self.labels = data['test_dataset_y']
            else:
                raise ValueError("split must be one of {'train','val','test'}")
        self.split = split
        self.sigma = sigma
        self.np_gen = np_gen
        self.np_rs  = np_rs

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        pc = self.points[idx].copy()
        is_tr = (self.split == 'train')
        pc = apply_data_augmentation(pc, self.sigma, self.np_gen, self.np_rs, is_training=is_tr)
        pc = normalize_unit_sphere(pc)
        return torch.FloatTensor(pc.T), torch.LongTensor([self.labels[idx]])

def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    tot = 0.0
    for data, label in train_loader:
        data, label = data.to(device), label.squeeze().to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, label)
        loss.backward()
        optimizer.step()
        tot += loss.item()
    return tot / len(train_loader)

@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for data, label in loader:
        data, label = data.to(device), label.squeeze().to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.numel()
    return correct / total if total > 0 else 0.0

def run_experiment(dataset_file, widths, k, *, num_classes, batch_size,
                   epochs, lr, dropout, sigma, seed):
    rng = make_rng_pack(seed)
    device = rng['device']
    torch_gen = rng['torch_gen_cuda'] or rng['torch_gen_cpu']
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"데이터셋 파일 '{dataset_file}'이 존재하지 않습니다.")
    train_dataset = PointCloudDataset(dataset_file, split='train', sigma=sigma,
                                      np_gen=rng['np_gen'], np_rs=rng['np_rs'])
    val_dataset   = PointCloudDataset(dataset_file, split='val',   sigma=sigma,
                                      np_gen=rng['np_gen'], np_rs=rng['np_rs'])
    test_dataset  = PointCloudDataset(dataset_file, split='test',  sigma=sigma,
                                      np_gen=rng['np_gen'], np_rs=rng['np_rs'])
    train_loader = fixed_loader(train_dataset, batch_size=batch_size, shuffle=True,  torch_gen=torch_gen)
    val_loader   = fixed_loader(val_dataset,   batch_size=batch_size, shuffle=False, torch_gen=torch_gen)
    test_loader  = fixed_loader(test_dataset,  batch_size=batch_size, shuffle=False, torch_gen=torch_gen)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = PointTransformerClassifier(num_classes=num_classes, widths=widths, k=k, dropout=dropout).to(device)
    print(f"params: {count_parameters(model):,}")
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0)
    best_val, best_epoch, best_state = 0.0, -1, None
    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        train_acc  = evaluate_accuracy(model, train_loader, device)
        val_acc    = evaluate_accuracy(model, val_loader, device)
        if val_acc >= best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
            torch.save(best_state, 'best_point_transformer_no_fps_by_val.pth')
        print(f'Epoch {epoch:03d} | Loss {train_loss:.4f} | Train {train_acc:.4f} | Val {val_acc:.4f} | Best {best_val:.4f} @ {best_epoch}')
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc = evaluate_accuracy(model, test_loader, device)
    print("\n==============================")
    print(f"[BEST by Val] epoch={best_epoch}, val_acc={best_val:.4f}")
    print(f"Test Acc = {test_acc:.4f}")
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


def variant_widths(variant: str):
    if variant == "light":
        return [5, 7, 8]
    return [8, 14, 18]


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
    num_qubit = args.num_points
    variant = normalize_variant(args.variant)
    widths = variant_widths(variant)
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

    print(f"Using seed={base_seed}")
    print(
        f"dataset={dataset_tag}, variant={variant}, num_points={num_points}, k={k}"
    )

    test_acc = run_experiment(
        dataset_file,
        widths,
        k,
        num_classes=num_classes,
        batch_size=35,
        epochs=1000,
        lr=0.01,
        dropout=0.0,
        sigma=sigma,
        seed=base_seed,
    )

    print(float(test_acc))


if __name__ == "__main__":
    main()




