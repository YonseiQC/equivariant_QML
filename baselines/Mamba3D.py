import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import special_ortho_group


def make_subseed(base_seed: int, *keys) -> int:
    import hashlib
    h = hashlib.sha256(str((base_seed,) + tuple(keys)).encode()).hexdigest()
    return int(h[:8], 16)


def make_rng_pack(base_seed: int, num_point: int, dataset_tag: str):
    subseed = make_subseed(base_seed, num_point, dataset_tag)
    scipy_rs = np.random.RandomState(subseed)
    np.random.seed(subseed)
    torch.manual_seed(subseed)
    return dict(subseed=subseed, scipy_rs=scipy_rs, base_seed=base_seed)


def _random_3d_rotation(np_rs):
    return special_ortho_group.rvs(3, random_state=np_rs)


def _add_jitter(points, sigma, np_gen):
    noise = np_gen.normal(loc=0.0, scale=sigma, size=points.shape)
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


class PointTokenizer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.point_embed = nn.Linear(3, embed_dim)

    def forward(self, x):
        """
        x: [B, N, 3]
        return: [B, N, C]
        """
        return self.point_embed(x)


class LNPBlock(nn.Module):
    """
    Implements LNP without k-NN search (global neighbor set) as in Eq. (7)–(8).
    """
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(1, 1, 1, 2 * embed_dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 2 * embed_dim))
        self.align = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, x):
        """
        x: [B, L, C] with CLS at index 0
        return: [B, L, C]
        """
        B, L, _ = x.shape
        C = self.embed_dim

        if L <= 1:
            return x

        cls_token = x[:, :1, :]
        F = x[:, 1:, :]  # [B, N, C]
        N = F.shape[1]

        if N == 1:
            Fc_C = torch.cat([F, F], dim=-1)
            tokens = self.align(Fc_C)
            return torch.cat([cls_token, tokens], dim=1)

        # Use all points as the neighbor set (no k-NN search)
        F_center = F.unsqueeze(2).expand(-1, -1, N, -1)         # [B, N, N, C]
        F_neighbors = F.unsqueeze(1).expand(-1, N, -1, -1)      # [B, N, N, C]

        diff = F_neighbors - F_center                           # [B, N, N, C]
        var = diff.var(dim=2, unbiased=False, keepdim=True)     # [B, N, 1, C]
        F_tilde = diff / torch.sqrt(var + self.eps)             # [B, N, N, C]

        Fc_K = torch.cat([F_tilde, F_center], dim=-1)           # [B, N, N, 2C]
        Fc_K = Fc_K * self.gamma + self.beta                    # affine in channel

        weights = torch.softmax(Fc_K, dim=2)                    # softmax over neighbors
        Fc_C = (weights * Fc_K).sum(dim=2)                      # [B, N, 2C]

        tokens = self.align(Fc_C)                               # [B, N, C]
        return torch.cat([cls_token, tokens], dim=1)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv = d_conv

        self.dt_rank = max(1, (d_model + 15) // 16)

        self.norm = nn.LayerNorm(d_model)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.conv = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self._init_dt(dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, dt_scale=1.0, dt_init="random")

    @staticmethod
    def _inv_softplus(y: torch.Tensor):
        return y + torch.log(-torch.expm1(-y))

    def _init_dt(self, dt_min, dt_max, dt_init_floor, dt_scale, dt_init):
        dt_init_std = (self.dt_rank ** -0.5) * dt_scale
        with torch.no_grad():
            if dt_init == "constant":
                self.dt_proj.weight.fill_(dt_init_std)
            elif dt_init == "random":
                self.dt_proj.weight.uniform_(-dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            log_dt_min = torch.log(torch.tensor(dt_min, dtype=torch.float32))
            log_dt_max = torch.log(torch.tensor(dt_max, dtype=torch.float32))
            dt = torch.exp(torch.rand(self.d_inner, dtype=torch.float32) * (log_dt_max - log_dt_min) + log_dt_min)
            dt = dt.clamp(min=dt_init_floor)
            self.dt_proj.bias.copy_(self._inv_softplus(dt))

    def forward(self, x):
        B, L, _ = x.shape
        x_norm = self.norm(x)

        u, z = self.in_proj(x_norm).chunk(2, dim=-1)

        u_conv = self.conv(u.transpose(1, 2))[:, :, :L].transpose(1, 2)
        u_conv = F.silu(u_conv)

        x_dbl = self.x_proj(u_conv.reshape(B * L, self.d_inner))
        dt, Bv, Cv = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        delta = F.linear(dt, self.dt_proj.weight).reshape(B, L, self.d_inner)
        delta = F.softplus(delta + self.dt_proj.bias.reshape(1, 1, self.d_inner))

        B_t = Bv.reshape(B, L, self.d_state)
        C_t = Cv.reshape(B, L, self.d_state)

        A = (-torch.exp(self.A_log)).to(dtype=u_conv.dtype, device=u_conv.device)
        D = self.D.to(dtype=u_conv.dtype, device=u_conv.device)

        h = torch.zeros((B, self.d_inner, self.d_state), device=u_conv.device, dtype=u_conv.dtype)
        ys = []

        z_gate = F.silu(z)

        for t in range(L):
            dt_t = delta[:, t, :]
            u_t = u_conv[:, t, :]
            Bt = B_t[:, t, :]
            Ct = C_t[:, t, :]

            exp_A = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            h = exp_A * h + (dt_t * u_t).unsqueeze(-1) * Bt.unsqueeze(1)

            y_t = (h * Ct.unsqueeze(1)).sum(dim=-1) + D.unsqueeze(0) * u_t
            y_t = y_t * z_gate[:, t, :]

            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        y = self.out_proj(y)
        return x + y


class BiSSM(nn.Module):
    """
    bi-SSM(F) = L+SSM(F_L+) + C-SSM(F_C-) (no residual add here).
    """
    def __init__(self, num_points, embed_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.L_ssm = MambaBlock(
            d_model=embed_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.C_ssm = MambaBlock(
            d_model=num_points,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        """
        x: [B, L, C]  (L = num_points, C = embed_dim)
        """
        F_L_plus = x
        out_L = self.L_ssm(F_L_plus)                 # [B, L, C]
        delta_L = out_L - F_L_plus                   # residual-less update

        F_C_minus = torch.flip(x, dims=[2])          # vertical flip over channels
        F_C_minus_t = F_C_minus.transpose(1, 2)      # [B, C, L]
        out_C_t = self.C_ssm(F_C_minus_t)            # [B, C, L]
        delta_C = (out_C_t - F_C_minus_t).transpose(1, 2)  # [B, L, C]

        return delta_L + delta_C


class Mamba3DBlock(nn.Module):
    """
    Token mixer (LNP) + channel mixer (bi-SSM), both with pre-norm and residual.
    """
    def __init__(self, num_points, embed_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.lnp = LNPBlock(embed_dim)
        self.bi_ssm = BiSSM(
            num_points=num_points,
            embed_dim=embed_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x, pos):
        """
        x: [B, L, C]  (L = num_points, C = embed_dim)
        pos: [B, L, C]
        """
        x = x + pos
        x = x + self.lnp(self.norm1(x))
        x = x + self.bi_ssm(self.norm2(x))
        return x


class CompactMamba3D(nn.Module):
    def __init__(
        self,
        num_points,
        num_classes,
        embed_dim=16,
        num_blocks=2,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()

        # Point tokenizer: per-point tokens (길이 N)
        self.tokenizer = PointTokenizer(embed_dim=embed_dim)

        # [CLS] token (길이 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # bi-SSM blocks: 시퀀스 길이가 N+1 ([CLS] 포함)이므로 num_points+1 사용
        self.blocks = nn.ModuleList(
            [
                Mamba3DBlock(
                    num_points=num_points + 1,
                    embed_dim=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_blocks)
            ]
        )

        self.head_norm = nn.LayerNorm(embed_dim)
        self.head_fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x: [B, N, 3]
        """
        # Point-wise token embedding
        tokens = self.tokenizer(x)          # [B, N, C]
        B, N, C = tokens.shape

        # [CLS] 토큰 붙이기
        cls = self.cls_token.expand(B, 1, C)  # [B, 1, C]
        h = torch.cat([cls, tokens], dim=1)   # [B, N+1, C]

        # Positional encoding for each block (CLS is learnable)
        pos = self.pos_embed(x)               # [B, N, C]
        cls_pos = self.cls_pos.expand(B, 1, C)
        pos = torch.cat([cls_pos, pos], dim=1)

        # LNP + bi-SSM stack
        for blk in self.blocks:
            h = blk(h, pos)

        # head: CLS 하나만 사용
        h = self.head_norm(h)
        cls_out = h[:, 0, :]                 # [B, C]
        logits = self.head_fc(cls_out)       # [B, K]
        return logits


def train_compact_mamba3d(
    dataset,
    batch_size,
    learning_rate,
    epochs,
    l2,
    rng_pack,
    use_augmentation=True,
    sigma=0.02,
    embed_dim=16,
    num_blocks=2,
    d_state=16,
    d_conv=4,
    expand=2,
):
    train_x = dataset["train_dataset_x"]
    train_y = dataset["train_dataset_y"]
    val_x = dataset["val_dataset_x"]
    val_y = dataset["val_dataset_y"]
    test_x = dataset["test_dataset_x"]
    test_y = dataset["test_dataset_y"]

    train_x = train_x.astype(np.float32)
    val_x = val_x.astype(np.float32)
    test_x = test_x.astype(np.float32)
    train_y = train_y.astype(np.int64)
    val_y = val_y.astype(np.int64)
    test_y = test_y.astype(np.int64)

    num_points = train_x.shape[1]
    num_classes = len(np.unique(train_y))

    model = CompactMamba3D(
        num_points=num_points,
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"총 파라미터 개수: {param_count:,}")

    best_val_acc = -1.0
    best_epoch = -1
    best_state = None

    subseed = rng_pack["subseed"]

    for epoch in range(epochs):
        shuffle_seed = make_subseed(subseed, "shuffle", epoch)
        perm = np.random.RandomState(shuffle_seed).permutation(train_x.shape[0])
        xs = train_x[perm]
        ys = train_y[perm]

        if use_augmentation:
            aug_seed = make_subseed(subseed, "aug", epoch)
            np_rs = np.random.RandomState(aug_seed)
            np_gen = np.random.RandomState(aug_seed + 1)
            xs_aug = np.stack(
                [
                    apply_data_augmentation(pt, sigma, np_gen, np_rs, is_training=True)
                    for pt in xs
                ]
            )
        else:
            xs_aug = xs

        xs_aug = xs_aug.astype(np.float32)

        model.train()
        num_samples = xs_aug.shape[0]
        assert num_samples % batch_size == 0
        num_batches = num_samples // batch_size
        epoch_loss = 0.0

        for i in range(num_batches):
            batch_x = torch.from_numpy(
                xs_aug[i * batch_size : (i + 1) * batch_size]
            ).to(device)
            batch_y = torch.from_numpy(
                ys[i * batch_size : (i + 1) * batch_size]
            ).to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            if l2 > 0:
                l2_loss = 0.0
                for p in model.parameters():
                    if p.requires_grad:
                        l2_loss += torch.sum(p ** 2)
                loss = loss + l2 * l2_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches

        model.eval()
        with torch.no_grad():
            train_logits = model(torch.from_numpy(train_x).to(device))
            train_preds = train_logits.argmax(dim=1).cpu().numpy()
            train_acc = np.mean(train_preds == train_y)

            val_logits = model(torch.from_numpy(val_x).to(device))
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = np.mean(val_preds == val_y)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:3d}: Loss={avg_loss:.4f} | Train={train_acc:.3f} | "
            f"Val={val_acc:.3f} | Best={best_val_acc:.3f}@{best_epoch}"
        )

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        test_logits = model(torch.from_numpy(test_x).to(device))
        test_preds = test_logits.argmax(dim=1).cpu().numpy()
        test_acc = np.mean(test_preds == test_y)

    print(f"Test Acc (evaluated ONCE with best-val params) = {test_acc:.4f}")
    print("=" * 60 + "\n")
    return model, best_epoch, best_val_acc, test_acc


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
        return dict(embed_dim=5, num_blocks=2, d_state=6, d_conv=2, expand=2)
    return dict(embed_dim=5, num_blocks=8, d_state=6, d_conv=2, expand=2)


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
    parser.add_argument("--num_points", type=int, required=True)
    parser.add_argument("--variant", type=str, choices=["light", "mid"], required=True)
    args = parser.parse_args()

    base_seed = args.seed
    num_points = args.num_points
    variant = normalize_variant(args.variant)
    dataset_tag, dataset_file, sigma = resolve_dataset(args.dataset, num_points)
    HERE = Path(__file__).resolve().parent
    REPO = HERE.parent

    tag = str(dataset_tag).lower()
    if tag == "modelnet":
        dataset_file = str(REPO / "data" / "ModelNet" / dataset_file)
    elif tag == "shapenet":
        dataset_file = str(REPO / "data" / "ShapeNet" / dataset_file)
    else:
        dataset_file = str(REPO / "data" / "Sydney_Urban_Objects" / dataset_file)
    cfg = variant_config(variant)

    np.random.seed(base_seed)

    dataset = np.load(dataset_file)
    rng_pack = make_rng_pack(base_seed, num_points, dataset_tag)

    print(f"Using seed={base_seed}")
    print(
        f"dataset={dataset_tag}, variant={variant}, num_points={num_points}, sigma={sigma}"
    )

    _, best_epoch, best_val, test_acc = train_compact_mamba3d(
        dataset,
        batch_size=35,
        learning_rate=0.01,
        epochs=1000,
        l2=0.0,
        rng_pack=rng_pack,
        use_augmentation=True,
        sigma=sigma,
        embed_dim=cfg["embed_dim"],
        num_blocks=cfg["num_blocks"],
        d_state=cfg["d_state"],
        d_conv=cfg["d_conv"],
        expand=cfg["expand"],
    )

    print(
        f"[{dataset_tag}] num_point={num_points} | BEST epoch={best_epoch}, "
        f"Val={best_val:.4f}, Test={test_acc:.4f}"
    )
    print(float(test_acc))


if __name__ == "__main__":
    main()
