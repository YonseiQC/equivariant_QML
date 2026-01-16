#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import json
import atexit
import datetime
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from scipy.stats import special_ortho_group

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
            sys.stdout.flush()
            sys.stderr.flush()
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
    scipy_rs = np.random.RandomState(subseed)
    base_key = jax.random.PRNGKey(subseed)
    return dict(subseed=subseed, scipy_rs=scipy_rs, base_key=base_key)


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


def loss_only(params, batch_x, batch_y, model, l2):
    logits = model.apply({"params": params}, batch_x)
    loss = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(logits, batch_y))
    l2_loss = l2 * sum(jnp.sum(jnp.square(param)) for param in jax.tree_util.tree_leaves(params))
    return loss + l2_loss


def ce_loss(params, x_all, y_all, model):
    logits = model.apply({"params": params}, x_all)
    losses = optax.losses.softmax_cross_entropy_with_integer_labels(logits, y_all)
    return jnp.mean(losses)


def accuracy_fn(params, batch_x, batch_y, model):
    logits = model.apply({"params": params}, batch_x)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == batch_y)


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
    train_x_3d = dataset["train_dataset_x"]
    train_y = dataset["train_dataset_y"]
    val_x_3d = dataset["val_dataset_x"]
    val_y = dataset["val_dataset_y"]
    test_x_3d = dataset["test_dataset_x"]
    test_y = dataset["test_dataset_y"]

    model = SimpleNN(variant=model_variant, num_classes=num_classes)

    variables = model.init(rng_pack["base_key"], jnp.array(train_x_3d[:1]))
    params = variables["params"]

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch_x, batch_y):
        loss, grads = jax.value_and_grad(loss_only)(params, batch_x, batch_y, model, l2)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    train_x_j = jnp.array(train_x_3d)
    train_y_j = jnp.array(train_y.astype(np.int32))
    val_x_j = jnp.array(val_x_3d)
    val_y_j = jnp.array(val_y.astype(np.int32))
    test_x_j = jnp.array(test_x_3d)
    test_y_j = jnp.array(test_y.astype(np.int32))

    best_val_acc = -1.0
    best_epoch = -1
    best_params = params

    for epoch in range(epochs):
        shuffle_key = jax.random.PRNGKey(make_subseed(rng_pack["subseed"], "shuffle", epoch))
        perm = jax.random.permutation(shuffle_key, train_x_3d.shape[0])
        xs = jnp.array(train_x_3d)[perm]
        ys = jnp.array(train_y.astype(np.int32))[perm]

        if use_augmentation:
            epoch_key = jax.random.fold_in(rng_pack["base_key"], epoch)
            current_train_x_3d = augment_batch_3d(
                xs,
                epoch_key,
                rng_pack["scipy_rs"],
                sigma=sigma,
                is_training=True,
            )
        else:
            current_train_x_3d = xs

        batch_size_total = len(current_train_x_3d)
        assert batch_size_total % batch_size == 0

        num_batches = batch_size_total // batch_size
        current_train_x_batched = current_train_x_3d.reshape(num_batches, batch_size, -1, 3)
        train_y_batched = ys.reshape(num_batches, batch_size)

        epoch_loss = 0.0
        for i in range(num_batches):
            params, opt_state, loss = train_step(
                params,
                opt_state,
                current_train_x_batched[i],
                train_y_batched[i],
            )
            epoch_loss += loss
        avg_train_loss = float(epoch_loss) / num_batches

        val_loss = float(ce_loss(params, val_x_j, val_y_j, model))
        val_acc = float(accuracy_fn(params, val_x_j, val_y_j, model))

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)

        print(
            f"epoch {epoch}/{epochs-1} | train loss : {avg_train_loss:.4f} | val loss : {val_loss:.4f} | val accuracy : {val_acc:.4f}"
        )
        _metrics_write(
            {
                "epoch": int(epoch),
                "train_loss": float(avg_train_loss),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }
        )

    final_test_acc = float(accuracy_fn(best_params, test_x_j, test_y_j, model))

    print("\n==============================")
    print(f"[BEST by Val] epoch={best_epoch}, val_acc={best_val_acc:.4f}")
    print(f"Test Acc (evaluated ONCE with best-val params) = {final_test_acc:.4f}")
    print("==============================\n")

    return best_params, model, best_epoch, best_val_acc, final_test_acc


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
    if d in {"suo", "scan"}:
        return "SUO", f"SUO_3classes_{num_points}_1_fps_train700_val100_test200_new.npz", 3, 0.01
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
    model_variant = normalize_variant(args.variant)

    epochs = 1000
    lr = 0.0001

    _setup_run(Path(__file__).stem, base_seed, args.dataset, num_points, model_variant, lr=lr, epochs=epochs)
    print(f"seed={base_seed}, dataset={args.dataset}, variant={model_variant}, num_points={num_points}, epochs={epochs}, lr={lr}")

    dataset_tag, npz_name, num_classes, sigma = resolve_dataset(args.dataset, num_points)

    HERE = Path(__file__).resolve().parent
    REPO = HERE.parent

    tag = str(dataset_tag).lower()
    if tag == "modelnet":
        dataset_path = REPO / "data" / "ModelNet" / npz_name
    elif tag == "shapenet":
        dataset_path = REPO / "data" / "ShapeNet" / npz_name
    else:
        dataset_path = REPO / "data" / "Sydney_Urban_Objects" / npz_name

    dataset = np.load(dataset_path)

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

    print(float(test_acc_once))


if __name__ == "__main__":
    main()
