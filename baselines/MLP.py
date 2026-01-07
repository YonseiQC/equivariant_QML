import argparse
from pathlib import Path
import hashlib
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import special_ortho_group
from torch.utils.data import DataLoader, TensorDataset


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
    return dict(subseed=subseed, np_rs=np_rs, torch_gen=torch_gen_cuda or torch_gen_cpu)


class MLPNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_hidden, out_dim):
        super().__init__()
        self.hidden = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
        for _ in range(num_hidden - 1):
            self.hidden.append(nn.Linear(hidden_dim, hidden_dim))
        self.tanh = nn.Tanh()
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = x.view(x.size(0), -1)
        for layer in self.hidden:
            h = self.tanh(layer(h))
        return self.fc_out(h)


def run_one_block(base_seed: int, points: int, dataset_tag: str, variant: str):
    coords = 3
    dim_per_part = points * coords
    batch_size = 35
    lr = 1e-3
    num_epochs = 1000

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
    tg = rng["torch_gen"]

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
        noise = torch.randn(pts.shape, device=pts.device, dtype=pts.dtype, generator=tg) * sigma_
        return pts + noise

    def apply_permutation(pts: torch.Tensor):
        num_points = pts.shape[0]
        perm_indices = np_rs.permutation(num_points).astype(np.int64)
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
    train_y = data["train_dataset_y"]
    val_x = data["val_dataset_x"].reshape(-1, points, coords).astype(np.float32)
    val_y = data["val_dataset_y"]
    test_x = data["test_dataset_x"].reshape(-1, points, coords).astype(np.float32)
    test_y = data["test_dataset_y"]

    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=tg, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    torch.manual_seed(rng["subseed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rng["subseed"])
        torch.cuda.manual_seed_all(rng["subseed"])

    model = MLPNet(dim_per_part, hidden_dim, num_hidden, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(
        f"[seed={base_seed}|sub={rng['subseed']}] points={points}, dataset={dataset_tag} | "
        f"params={sum(p.numel() for p in model.parameters())}"
    )

    @torch.no_grad()
    def evaluate_accuracy(model_, loader):
        model_.eval()
        correct, total = 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model_(xb)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
        return correct / total if total > 0 else 0.0

    best_val = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = augment_batch(xb, is_training=True)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * xb.size(0)

        avg_train_loss = running_train_loss / len(train_loader.dataset)

        train_acc = evaluate_accuracy(model, train_loader)
        val_acc = evaluate_accuracy(model, val_loader)

        if val_acc >= best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[{dataset_tag}] Epoch {epoch:4d} | Loss {avg_train_loss:.4f} | "
                f"Train {train_acc:.4f} | Val {val_acc:.4f} | Best {best_val:.4f}@{best_epoch}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    final_predictions, final_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(1)
            final_predictions.extend(pred.cpu().numpy())
            final_targets.extend(yb.cpu().numpy())

    final_overall_acc = float((np.array(final_predictions) == np.array(final_targets)).mean())
    print(f"[{dataset_tag}] BEST@{best_epoch} | Val {best_val:.4f} | Test {final_overall_acc:.4f}\n")
    return final_overall_acc


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

    base_seed = args.seed
    num_points = args.num_points
    variant = normalize_variant(args.variant)
    dataset_tag, _ = resolve_dataset(args.dataset, num_points)

    print(f"Using seed={base_seed}")
    print(
        f"dataset={dataset_tag}, variant={variant}, num_points={num_points}"
    )

    acc = run_one_block(base_seed, num_points, dataset_tag, variant)
    print(float(acc))


if __name__ == "__main__":
    main()
