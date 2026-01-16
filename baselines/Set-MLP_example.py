#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import sys
import json
import atexit
import datetime

import jax
jax.config.update("jax_enable_x64", True)

import hashlib

import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from scipy.stats import special_ortho_group
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

    cfg = {
        "model": str(model),
        "seed": int(seed),
        "dataset": str(dataset),
        "variant": str(variant),
        "num_points": int(num_points),
        "epochs": int(epochs),
        "lr": float(lr),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if k is not None:
        cfg["k"] = int(k)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    _metrics_write({"event": "start", "run_id": run_id, "timestamp": datetime.datetime.now().isoformat()})
    return run_id


# --------------------- RNG helpers ---------------------

def make_subseed(base_seed: int, *keys) -> int:
    h = hashlib.sha256(str((base_seed,) + tuple(keys)).encode()).hexdigest()
    return int(h[:8], 16)


def make_rng_pack(base_seed: int, num_point: int, dataset_tag: str):
    subseed = make_subseed(base_seed, num_point, dataset_tag)
    scipy_rs = np.random.RandomState(subseed)
    base_key = jax.random.PRNGKey(subseed)
    return dict(subseed=subseed, scipy_rs=scipy_rs, base_key=base_key)


# --------------------- Augmentation ---------------------

def random_3d_rotation(scipy_rs):
    return special_ortho_group.rvs(3, random_state=scipy_rs)


def apply_3d_rotation(points, scipy_rs):
    rotation_matrix = random_3d_rotation(scipy_rs)
    rotation_matrix_jax = jnp.array(rotation_matrix)
    return jnp.dot(points, rotation_matrix_jax.T)


def apply_permutation(points, scipy_rs):
    num_points = points.shape[0]
    perm_indices = scipy_rs.permutation(num_points)
    return points[perm_indices]


def add_jitter(points, key, sigma):
    noise = jax.random.normal(key, points.shape) * sigma
    return points + noise


def apply_data_augmentation_3d(points, key, scipy_rs, sigma=0.02, is_training=True):
    if not is_training:
        return points
    _, key2 = jax.random.split(key)
    out = add_jitter(points, key2, sigma=sigma)
    out = apply_3d_rotation(out, scipy_rs)
    out = apply_permutation(out, scipy_rs)
    return out


def augment_batch_3d(batch_points, key, scipy_rs, sigma=0.02, is_training=True):
    if not is_training:
        return batch_points
    batch_size = batch_points.shape[0]
    keys = jax.random.split(key, batch_size)
    augment_fn = jax.vmap(lambda points, k: apply_data_augmentation_3d(points, k, scipy_rs, sigma, is_training))
    return augment_fn(batch_points, keys)


# --------------------- Model ---------------------

class SimpleNN(nn.Module):
    variant: str
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        x = jnp.expand_dims(x, axis=-1)

        if self.variant == "mid":
            x = nn.Dense(features=8)(x)
            x = nn.tanh(x)
            x = nn.Dense(features=16)(x)
            x = nn.tanh(x)
            x = nn.Dense(features=32)(x)
            x = nn.tanh(x)
        else:
            x = nn.Dense(features=4)(x)
            x = nn.tanh(x)
            x = nn.Dense(features=4)(x)
            x = nn.tanh(x)

        mean_pool = jnp.mean(x, axis=1)
        max_pool = jnp.max(x, axis=1)
        min_pool = jnp.min(x, axis=1)
        std_pool = jnp.std(x, axis=1)
        sum_pool = jnp.sum(x, axis=1)
        var_pool = jnp.var(x, axis=1)

        x = jnp.concatenate([mean_pool, max_pool, min_pool, std_pool, sum_pool, var_pool], axis=-1)

        if self.variant == "mid":
            x = nn.Dense(features=32)(x)
            x = nn.tanh(x)
            x = nn.Dense(features=16)(x)
            x = nn.tanh(x)
            x = nn.Dense(features=8)(x)
            x = nn.tanh(x)
        else:
            x = nn.Dense(features=24)(x)
            x = nn.tanh(x)
            x = nn.Dense(features=24)(x)
            x = nn.tanh(x)

        x = nn.Dense(features=self.num_classes)(x)
        return x


# --------------------- Loss / Metrics ---------------------

def loss_fn(params, batch_x, batch_y, model, l2):
    logits = model.apply(params, batch_x)
    loss = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(logits, batch_y))
    l2_loss = l2 * sum(jnp.sum(jnp.square(param)) for param in jax.tree_util.tree_leaves(params))
    return loss + l2_loss


def calculate_final_metrics(y_true, y_pred, num_classes):
    y_true_np = np.asarray(y_true).reshape(-1)
    y_pred_np = np.asarray(y_pred).reshape(-1)
    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(num_classes)))
    class_accs = []
    for i in range(num_classes):
        denom = cm[i].sum()
        class_accs.append(float(cm[i, i] / denom) if denom > 0 else 0.0)
    overall_acc = float(np.trace(cm) / np.sum(cm)) if np.sum(cm) > 0 else 0.0
    return cm, class_accs, overall_acc


# --------------------- Train (Val-best -> Test once) ---------------------

def train_attention_deepsets(
    dataset,
    batch_size,
    learning_rate,
    epochs,
    l2,
    rng_pack,
    model_variant,
    num_classes,
    use_augmentation=True,
    sigma=0.02,
):
    train_x_3d = dataset["train_dataset_x"].astype(np.float64)
    train_y = dataset["train_dataset_y"].astype(np.int32)
    val_x_3d = dataset["val_dataset_x"].astype(np.float64)
    val_y = dataset["val_dataset_y"].astype(np.int32)
    test_x_3d = dataset["test_dataset_x"].astype(np.float64)
    test_y = dataset["test_dataset_y"].astype(np.int32)

    model = SimpleNN(variant=model_variant, num_classes=num_classes)

    params = model.init(rng_pack["base_key"], train_x_3d[:1])
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"총 파라미터 개수: {param_count:,}")

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch_x, batch_y):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch_x, batch_y, model, l2)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_loss(params, x_all, y_all):
        logits = model.apply(params, x_all)
        return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(logits, y_all))

    @jax.jit
    def eval_acc(params, x_all, y_all):
        logits = model.apply(params, x_all)
        preds = jnp.argmax(logits, axis=-1)
        return jnp.mean((preds == y_all).astype(jnp.float32))

    best_val_acc = 0.0
    best_epoch = -1
    best_params = params

    train_x_all = jnp.array(train_x_3d)
    train_y_all = jnp.array(train_y)
    val_x_all = jnp.array(val_x_3d)
    val_y_all = jnp.array(val_y)
    test_x_all = jnp.array(test_x_3d)
    test_y_all = jnp.array(test_y)

    N = int(train_x_3d.shape[0])
    assert N % batch_size == 0

    num_batches = N // batch_size

    for epoch in range(1, epochs + 1):
        shuffle_key = jax.random.PRNGKey(make_subseed(rng_pack["subseed"], "shuffle", epoch))
        perm = jax.random.permutation(shuffle_key, N)
        xs = train_x_all[perm]
        ys = train_y_all[perm]

        if use_augmentation:
            epoch_key = jax.random.fold_in(rng_pack["base_key"], epoch)
            xs = augment_batch_3d(xs, epoch_key, rng_pack["scipy_rs"], sigma=sigma, is_training=True)

        epoch_loss = 0.0
        for b in range(num_batches):
            start = b * batch_size
            end = start + batch_size
            params, opt_state, loss = train_step(params, opt_state, xs[start:end], ys[start:end])
            epoch_loss += float(loss) / num_batches

        train_loss = float(epoch_loss)
        val_loss = float(eval_loss(params, val_x_all, val_y_all))
        val_acc = float(eval_acc(params, val_x_all, val_y_all))

        if val_acc >= best_val_acc:
            best_val_acc = float(val_acc)
            best_epoch = int(epoch - 1)
            best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)

        print(f"epoch {epoch-1}/{epochs-1} | train loss : {train_loss:.4f} | val loss : {val_loss:.4f} | val accuracy : {val_acc:.4f}")
        _metrics_write({"epoch": int(epoch-1), "train_loss": float(train_loss), "val_loss": float(val_loss), "val_acc": float(val_acc)})

    test_logits = model.apply(best_params, test_x_all)
    test_pred = jnp.argmax(test_logits, axis=-1)

    cm, class_accs, test_acc = calculate_final_metrics(np.array(test_y_all), np.array(test_pred), num_classes)

    print("\n==============================")
    print(f"[BEST by Val] epoch={best_epoch}, val_acc={best_val_acc:.4f}")
    print(f"Test Acc (evaluated ONCE with best-val params) = {test_acc:.4f}")
    print("Class-wise Accuracy:")
    for i, a in enumerate(class_accs):
        print(f"  Class {i}: {float(a):.4f}")
    print("==============================\n")

    _metrics_write(
        {
            "final": True,
            "best_epoch": int(best_epoch),
            "best_val_acc": float(best_val_acc),
            "test_acc": float(test_acc),
            "class_acc": [float(a) for a in class_accs],
        }
    )

    return best_params, model, best_epoch, best_val_acc, test_acc


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
    model_variant = normalize_variant(args.variant)
    dataset_tag, npz_name, num_classes, sigma = resolve_dataset(args.dataset, num_points)

    np.random.seed(base_seed)
    HERE = Path(__file__).resolve().parent
    REPO = HERE.parent

    tag = str(dataset_tag).lower()
    if tag == "modelnet":
        dataset_path = REPO / "data" / "ModelNet" / npz_name
    elif tag == "shapenet":
        dataset_path = REPO / "data" / "ShapeNet" / npz_name
    else:
        dataset_path = REPO / "data" / "Sydney_Urban_Objects" / npz_name

    lr = 0.0001
    epochs = 3
    _setup_run(Path(__file__).stem, base_seed, args.dataset, num_points, model_variant, lr=lr, epochs=epochs)

    dataset = np.load(dataset_path)

    print(f"seed={base_seed}, dataset={args.dataset}, variant={model_variant}, num_points={num_points}, epochs={epochs}, lr={lr}")
    print("jax_enable_x64=True")

    rng_pack = make_rng_pack(base_seed, num_points, dataset_tag)

    _, _, best_epoch, best_val, test_acc_once = train_attention_deepsets(
        dataset,
        batch_size=35,
        learning_rate=lr,
        epochs=epochs,
        l2=0.0,
        rng_pack=rng_pack,
        model_variant=model_variant,
        num_classes=num_classes,
        use_augmentation=True,
        sigma=sigma,
    )

    print(f"[{dataset_tag}] num_point={num_points} | BEST epoch={best_epoch}, Val={best_val:.4f}, Test={test_acc_once:.4f}")
    print(float(test_acc_once))


if __name__ == "__main__":
    main()
