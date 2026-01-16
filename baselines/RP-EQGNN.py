#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pennylane as qml
from flax import linen as nn
from sklearn.metrics import confusion_matrix

import sys
import json
import atexit
import datetime

jax.config.update("jax_enable_x64", True)

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


def make_subseed(base_seed: int, *keys) -> int:
    h = hashlib.sha256(str((base_seed,) + tuple(keys)).encode()).hexdigest()
    return int(h[:8], 16)


def make_rng_pack(base_seed: int, num_point: int, dataset_tag: str):
    subseed = make_subseed(base_seed, num_point, dataset_tag)
    base_key = jax.random.PRNGKey(subseed)
    return dict(subseed=subseed, base_key=base_key)


def ensure_points_dim(x: np.ndarray):
    if x.ndim != 3:
        raise ValueError(f"Expected (M, N, 3), got {x.shape}")
    return x


def epoch_shuffle_numpy(x: np.ndarray, y: np.ndarray, seed: int):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(x.shape[0])
    return x[idx], y[idx]


def build_edge_pairs(num_points: int):
    pairs = []
    for i in range(num_points):
        for j in range(i + 1, num_points):
            pairs.append((i, j))
    return pairs


def make_edge_feature_fn(num_points: int):
    pairs = build_edge_pairs(num_points)

    def edge_features(points: jnp.ndarray) -> jnp.ndarray:
        feats = []
        eps = 1e-8
        for (u, w) in pairs:
            pu = points[u]
            pw = points[w]
            diff = pu - pw
            dist = jnp.linalg.norm(diff)
            denom = jnp.linalg.norm(pu) * jnp.linalg.norm(pw) + eps
            cosang = jnp.dot(pu, pw) / denom
            cosang = jnp.clip(cosang, -1.0, 1.0)
            ang = jnp.arccos(cosang)
            feats.append(dist)
            feats.append(ang)
        return jnp.stack(feats)

    return edge_features, pairs, len(pairs)


def random_rotation_matrix(key):
    A = jax.random.normal(key, (3, 3))
    Q, R = jnp.linalg.qr(A)
    d = jnp.sign(jnp.diag(R))
    Q = Q * d
    det = jnp.linalg.det(Q)
    Q = jnp.where(det < 0, Q.at[:, 0].set(-Q[:, 0]), Q)
    return Q


def add_jitter(points, key, sigma):
    noise = jax.random.normal(key, points.shape) * sigma
    return points + noise


def augment_one(points, key, sigma):
    k1, k2, k3 = jax.random.split(key, 3)
    pts = add_jitter(points, k1, sigma)
    R = random_rotation_matrix(k2)
    pts = pts @ R.T
    idx = jax.random.permutation(k3, pts.shape[0])
    pts = pts[idx]
    return pts


def augment_batch(batch_points, key, sigma, is_training: bool):
    if not is_training:
        return batch_points
    B = batch_points.shape[0]
    keys = jax.random.split(key, B)
    return jax.vmap(augment_one, in_axes=(0, 0, None))(batch_points, keys, sigma)


def make_global_z_obs(num_qubits: int):
    obs = qml.PauliZ(0)
    for i in range(1, num_qubits):
        obs = obs @ qml.PauliZ(i)
    return obs


class RPHead(nn.Module):
    num_classes: int
    variant: str

    @nn.compact
    def __call__(self, q_scalar: jnp.ndarray):
        x = nn.Dense(32)(q_scalar)
        x = nn.silu(x)
        if self.variant == "mid":
            x = nn.Dense(64)(x)
            x = nn.silu(x)
            x = nn.Dense(64)(x)
            x = nn.silu(x)
        x = nn.Dense(32)(x)
        x = nn.silu(x)
        x = nn.Dense(self.num_classes)(x)
        return x


def build_rpeqgnn_qnode(num_points: int, depth: int = 2):
    num_qubits = num_points
    dev = qml.device("default.qubit", wires=num_qubits)

    edge_feat_fn, edge_pairs, E = make_edge_feature_fn(num_points)

    L_enc = 3 * num_points
    L_node = num_points
    L_edge = 2 * E
    L_vec = L_enc + L_node + L_edge

    dim_w_total = 4 * depth

    @qml.qnode(dev, interface="jax")
    def qnode(weights: jnp.ndarray, vec_input: jnp.ndarray):
        enc_params = vec_input[:L_enc]
        node_params = vec_input[L_enc : L_enc + L_node]
        edge_feats = vec_input[L_enc + L_node :]

        coords = enc_params.reshape((num_points, 3))
        node_feats = node_params.reshape((num_points,))
        edge_feats = edge_feats.reshape((E, 2))

        w_c = weights[: 2 * depth].reshape((depth, 2))
        w_e = weights[2 * depth :].reshape((depth, 2))

        for i in range(num_qubits):
            qml.RY(coords[i, 0], wires=i)
            qml.RX(coords[i, 1], wires=i)
            qml.RY(coords[i, 2], wires=i)

        for l in range(depth):
            w_c_l = w_c[l]
            w_e_l = w_e[l]

            for i in range(num_qubits):
                f = node_feats[i]
                qml.RY(f * w_c_l[0], wires=i)
                qml.RX(f * w_c_l[1], wires=i)

            for e_idx, (a, b) in enumerate(edge_pairs):
                dist = edge_feats[e_idx, 0]
                ang = edge_feats[e_idx, 1]
                theta = w_e_l[0] * dist + w_e_l[1] * ang
                qml.IsingXX(theta, wires=[a, b])

        z_obs = make_global_z_obs(num_qubits)
        return qml.expval(z_obs)

    qnode_batched = jax.vmap(qnode, in_axes=(None, 0))
    qnode_batched = jax.jit(qnode_batched)

    return qnode_batched, edge_feat_fn, L_vec, dim_w_total


def calculate_final_metrics(y_true, y_pred, num_classes):
    y_true_np = np.asarray(y_true).flatten()
    y_pred_np = np.asarray(y_pred).flatten()
    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(num_classes)))
    class_accs = []
    for i in range(num_classes):
        denom = cm[i].sum()
        class_accs.append(cm[i, i] / denom if denom > 0 else 0.0)
    overall_acc = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0
    return cm, class_accs, float(overall_acc)


def build_model(num_points: int, num_classes: int, depth: int = 2, variant: str = "light"):
    qnode_batched, edge_feat_fn, _, dim_w_total = build_rpeqgnn_qnode(num_points, depth=depth)
    head = RPHead(num_classes=num_classes, variant=variant)

    def build_input_vec(points: jnp.ndarray) -> jnp.ndarray:
        x_flat = points.reshape(-1)
        norms = jnp.linalg.norm(points, axis=1)
        node_feats = norms
        edge_feats = edge_feat_fn(points)
        return jnp.concatenate([x_flat, node_feats.reshape(-1), edge_feats], axis=0)

    build_input_vec_batched = jax.vmap(build_input_vec, in_axes=0)

    def init_params(rng):
        rng_q, rng_h = jax.random.split(rng)
        w_init = 0.02 * jax.random.normal(rng_q, (dim_w_total,))
        dummy_q = jnp.zeros((1, 1))
        head_params = head.init(rng_h, dummy_q)["params"]
        return {"quantum": w_init, "head": head_params}

    def forward(params, pts_batch: jnp.ndarray):
        vec_batch = build_input_vec_batched(pts_batch)
        q_scalar = qnode_batched(params["quantum"], vec_batch).reshape(-1, 1)
        logits = head.apply({"params": params["head"]}, q_scalar)
        return logits

    return init_params, jax.jit(forward)


def train_rpeqgnn(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    num_points: int,
    num_classes: int,
    base_seed: int,
    dataset_tag: str,
    num_epochs: int = 100,
    batch_size: int = 35,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    use_augmentation: bool = True,
    depth: int = 2,
    variant: str = "light",
    sigma: float = 0.02,
):
    train_x = ensure_points_dim(train_x)
    val_x = ensure_points_dim(val_x)
    test_x = ensure_points_dim(test_x)
    assert train_x.shape[1] == num_points

    rng_pack = make_rng_pack(base_seed, num_points, dataset_tag)
    subseed = rng_pack["subseed"]
    base_key = rng_pack["base_key"]

    init_params, forward = build_model(num_points, num_classes, depth=depth, variant=variant)
    params = init_params(jax.random.fold_in(base_key, 0))

    def loss_fn(params_, x_batch, y_batch):
        if y_batch.ndim > 1:
            y_batch = jnp.squeeze(y_batch)
        logits = forward(params_, x_batch)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y_batch))
        return loss

    @jax.jit
    def loss_and_grad(params_, x_batch, y_batch):
        return jax.value_and_grad(loss_fn)(params_, x_batch, y_batch)

    @jax.jit
    def accuracy(params_, x_all, y_all):
        logits = forward(params_, x_all)
        preds = jnp.argmax(logits, axis=-1)
        return jnp.mean((preds == y_all).astype(jnp.float32))

    @jax.jit
    def eval_loss(params_, x_all, y_all):
        logits = forward(params_, x_all)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y_all))

    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay) if weight_decay > 0 else optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    num_train = train_x.shape[0]
    assert num_train % batch_size == 0
    num_batches = num_train // batch_size

    best_val_acc = 0.0
    best_epoch = -1
    best_params = params

    val_x_j = jnp.array(val_x)
    val_y_j = jnp.array(val_y.astype(np.int32))
    test_x_j = jnp.array(test_x)
    test_y_j = jnp.array(test_y.astype(np.int32))

    for epoch in range(1, num_epochs + 1):
        shuffle_seed = make_subseed(subseed, "shuffle", epoch)
        train_x_shuf, train_y_shuf = epoch_shuffle_numpy(train_x, train_y, shuffle_seed)

        epoch_key = jax.random.fold_in(base_key, epoch)
        train_x_aug = augment_batch(jnp.array(train_x_shuf), epoch_key, sigma, is_training=use_augmentation)

        epoch_loss = 0.0
        for b in range(num_batches):
            start = b * batch_size
            end = start + batch_size
            x_batch = train_x_aug[start:end]
            y_batch = jnp.array(train_y_shuf[start:end].astype(np.int32))

            loss_val, grads = loss_and_grad(params, x_batch, y_batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            epoch_loss += float(loss_val) / num_batches

        val_loss = float(eval_loss(params, val_x_j, val_y_j))
        val_acc = float(accuracy(params, val_x_j, val_y_j))

        e0 = epoch - 1
        print(f"epoch {e0}/{num_epochs-1} | train loss : {epoch_loss:.4f} | val loss : {val_loss:.4f} | val accuracy : {val_acc:.4f}")
        _metrics_write({"epoch": int(e0), "train_loss": float(epoch_loss), "val_loss": float(val_loss), "val_acc": float(val_acc)})

        if val_acc >= best_val_acc:
            best_val_acc = float(val_acc)
            best_epoch = int(e0)
            best_params = params

    test_logits = forward(best_params, test_x_j)
    test_preds = jnp.argmax(test_logits, axis=-1)
    _, class_acc, test_acc = calculate_final_metrics(np.array(test_y_j), np.array(test_preds), num_classes)

    print("\n=== Results ===")
    print(f"Test Accuracy: {float(test_acc):.4f}")
    print("Class-wise Accuracy:")
    for i, a in enumerate(class_acc):
        print(f"  Class {i}: {float(a):.4f}")

    _metrics_write(
        {
            "final": True,
            "best_epoch": int(best_epoch),
            "best_val_acc": float(best_val_acc),
            "test_acc": float(test_acc),
            "class_acc": [float(a) for a in class_acc],
        }
    )

    return float(best_val_acc), float(test_acc)


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
    parser.add_argument("--num_points", type=int, required=True)
    parser.add_argument("--variant", type=str, choices=["light", "mid"], required=True)
    args = parser.parse_args()

    base_seed = int(args.seed)
    num_points = int(args.num_points)
    variant = normalize_variant(args.variant)
    dataset_tag, dataset_file, num_classes, sigma = resolve_dataset(args.dataset, num_points)

    epochs = 3
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

    data = np.load(dataset_file)
    train_x = data["train_dataset_x"].astype(np.float64)
    train_y = data["train_dataset_y"].astype(np.int32)
    val_x = data["val_dataset_x"].astype(np.float64)
    val_y = data["val_dataset_y"].astype(np.int32)
    test_x = data["test_dataset_x"].astype(np.float64)
    test_y = data["test_dataset_y"].astype(np.int32)

    test_acc = train_rpeqgnn(
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        num_points=num_points,
        num_classes=num_classes,
        base_seed=base_seed,
        dataset_tag=dataset_tag,
        num_epochs=epochs,
        batch_size=35,
        learning_rate=lr,
        weight_decay=0.0,
        use_augmentation=True,
        depth=50,
        variant=variant,
        sigma=sigma,
    )[1]

    print(float(test_acc))


if __name__ == "__main__":
    main()
