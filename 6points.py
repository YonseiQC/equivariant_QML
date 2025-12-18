#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import jax
import jax.numpy as jnp
import optax
import math
from flax import linen as nn
from hyqurp_cpp import QRPCircuit
import hashlib
from sklearn.metrics import confusion_matrix

jax.config.update("jax_enable_x64", True)

def make_subseed(base_seed: int, *keys) -> int:
    h = hashlib.sha256(str((base_seed,) + tuple(keys)).encode()).hexdigest()
    return int(h[:8], 16)

def make_rng_pack(base_seed: int, num_point: int, dataset_tag: str):
    subseed = make_subseed(base_seed, num_point, dataset_tag)
    scipy_rs = np.random.RandomState(subseed)
    base_key = jax.random.PRNGKey(subseed)
    return dict(subseed=subseed, scipy_rs=scipy_rs, base_key=base_key)

scipy_rng = None
key = None
_global_subseed = None

def get_Theta(point):
    point = point["train_dataset_x"]
    point = point - jnp.mean(point, axis=0)
    point_sqr = jnp.power(point, 2)
    norms = jnp.sqrt(jnp.sum(point_sqr, axis=-1))
    Theta = jnp.max(norms) * 1.2
    return Theta

def add_jitter_numpy(points, rng, sigma):
    return points + rng.normal(size=points.shape) * sigma

def apply_data_augmentation_numpy(points, rng, sigma, is_training=True):
    if not is_training:
        return points
    return add_jitter_numpy(points, rng, sigma)

def augment_batch_numpy(batch_points, seed, sigma, is_training=True):
    if not is_training:
        return batch_points
    rng = np.random.RandomState(seed)
    return apply_data_augmentation_numpy(batch_points, rng, sigma, is_training=True)

def epoch_shuffle_numpy(x, y, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(x.shape[0])
    return x[idx], y[idx]

class MyNN(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = jnp.expand_dims(x, axis=-1)
        x = nn.Dense(features=8)(x)
        x = jnp.tanh(x)
        x = nn.Dense(features=16)(x)
        x = jnp.tanh(x)
        x = nn.Dense(features=32)(x)
        x = jnp.tanh(x)

        mean_pool = jnp.mean(x, axis=1)
        max_pool = jnp.max(x, axis=1)
        min_pool = jnp.min(x, axis=1)
        std_pool = jnp.std(x, axis=1)
        sum_pool = jnp.sum(x, axis=1)
        var_pool = jnp.var(x, axis=1)

        x = jnp.concatenate([mean_pool, max_pool, min_pool, sum_pool, std_pool, var_pool], axis=-1)

        x = nn.Dense(features=32)(x)
        x = jnp.tanh(x)
        x = nn.Dense(features=16)(x)
        x = jnp.tanh(x)
        x = nn.Dense(features=8)(x)
        x = jnp.tanh(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x

def calculate_final_metrics(y_true, y_pred, num_classes):
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()
    cm = confusion_matrix(y_true_np, y_pred_np, labels=range(num_classes))
    class_accuracies = []
    for i in range(num_classes):
        denom = np.sum(cm[i, :])
        class_accuracies.append((cm[i, i] / denom) if denom > 0 else 0.0)
    overall_accuracy = np.trace(cm) / np.sum(cm)
    return cm, class_accuracies, overall_accuracy

def ensure_r1(x):
    if x.ndim == 3:
        return x
    if x.ndim == 4:
        return x[:, 0, :, :]
    raise ValueError(f"Unexpected shape: {x.shape}")

def adam_init_numpy(param_np):
    return {"m": np.zeros_like(param_np), "v": np.zeros_like(param_np), "t": 0}

def adam_update_numpy(param_np, grad_np, state, lr, b1, b2, eps):
    t = state["t"] + 1
    m = b1 * state["m"] + (1.0 - b1) * grad_np
    v = b2 * state["v"] + (1.0 - b2) * (grad_np * grad_np)
    mhat = m / (1.0 - (b1 ** t))
    vhat = v / (1.0 - (b2 ** t))
    param_np = param_np - lr * mhat / (np.sqrt(vhat) + eps)
    state["m"] = m
    state["v"] = v
    state["t"] = t
    return param_np, state

def train(minibatch_size, Theta, epochs, num_blocks_reupload, num_points, use_augmentation, sigma, num_classes, l2, learning_rate, b1=0.9, b2=0.999, eps=1e-8):
    global train_dataset_x, train_dataset_y, val_dataset_x, val_dataset_y, test_dataset_x, test_dataset_y
    assert len(train_dataset_x) == len(train_dataset_y)
    assert len(train_dataset_x) % minibatch_size == 0

    repeat = num_blocks_reupload
    H = num_points * (num_points - 1) // 2
    feat_dim = 2 * H

    circ = QRPCircuit(num_points)

    init_u2 = (0.02 * math.pi) / (2 * (num_points - 1) * repeat)
    qkey = jax.random.PRNGKey(make_subseed(_global_subseed, "init_q"))
    params_q_jax = init_u2 * jax.random.uniform(qkey, (repeat, num_points - 1, 2), dtype=jnp.float64)
    q_np = np.asarray(np.array(params_q_jax), dtype=np.float64, order="C")
    q_state = adam_init_numpy(q_np)

    dummy_input = jnp.ones((1, feat_dim), dtype=jnp.float64)
    model = MyNN(num_classes=num_classes)
    params_c = model.init(key, dummy_input)

    solver_c = optax.adam(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps)
    opt_state_c = solver_c.init(params_c)

    def logits_from_feats(params_c_, feats_flat_):
        return model.apply(params_c_, feats_flat_)

    def loss_from_feats(params_c_, feats_flat_, y_batch_):
        logits = logits_from_feats(params_c_, feats_flat_)
        return jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(logits, y_batch_))

    vg_loss = jax.value_and_grad(loss_from_feats, argnums=(0, 1))

    @jax.jit
    def step_c(params_c_, opt_state_c_, values_, y_batch_):
        feats_flat = values_.reshape(minibatch_size, -1)
        loss_val, (grad_c, grad_feats_flat) = vg_loss(params_c_, feats_flat, y_batch_)
        grad_feats = grad_feats_flat.reshape(minibatch_size, 2, H)

        if l2 != 0:
            grad_c = jax.tree.map(lambda g, p: g + 2.0 * l2 * p, grad_c, params_c_)
            c_sq = jnp.sum(jnp.array([jnp.sum(jnp.square(p)) for p in jax.tree.leaves(params_c_)]))
            loss_val = loss_val + l2 * c_sq

        updates_c, opt_state_c_ = solver_c.update(grad_c, opt_state_c_, params_c_)
        params_c_ = optax.apply_updates(params_c_, updates_c)
        return params_c_, opt_state_c_, loss_val, grad_feats

    best_val_acc = -1.0
    best_epoch = -1
    best_q_np = q_np.copy()
    best_params_c = params_c

    batch_size = len(train_dataset_x)
    num_batches = batch_size // minibatch_size

    for epoch in range(epochs):
        shuffle_seed = make_subseed(_global_subseed, "shuffle", epoch)
        xs, ys = epoch_shuffle_numpy(train_dataset_x, train_dataset_y, shuffle_seed)

        if use_augmentation:
            aug_seed = make_subseed(_global_subseed, "aug", epoch)
            xs = augment_batch_numpy(xs, aug_seed, sigma, is_training=True)

        epoch_loss = 0.0

        for i in range(num_batches):
            x_batch = xs[i * minibatch_size : (i + 1) * minibatch_size]
            y_batch = ys[i * minibatch_size : (i + 1) * minibatch_size]

            x_scaled = np.asarray(x_batch / float(Theta), dtype=np.float64, order="C")

            values, grads = circ.value_and_grad(x_scaled, q_np)
            values_j = jnp.asarray(values)
            yb_j = jnp.asarray(y_batch)

            params_c, opt_state_c, loss_val_j, grad_feats_j = step_c(params_c, opt_state_c, values_j, yb_j)

            grad_feats_np = np.asarray(grad_feats_j, dtype=np.float64, order="C")
            grads_np = np.asarray(grads, dtype=np.float64, order="C")
            grad_q_np = np.tensordot(grad_feats_np, grads_np, axes=([0, 1, 2], [0, 1, 2]))

            if l2 != 0:
                grad_q_np = grad_q_np + 2.0 * l2 * q_np
                loss_val_j = loss_val_j + l2 * float(np.sum(q_np * q_np))

            q_np, q_state = adam_update_numpy(q_np, grad_q_np, q_state, learning_rate, b1, b2, eps)

            epoch_loss += float(loss_val_j) / num_batches

        val_x_scaled = np.asarray(val_dataset_x / float(Theta), dtype=np.float64, order="C")
        val_values = jnp.asarray(circ.forward(val_x_scaled, q_np))
        val_feats_flat = val_values.reshape(val_dataset_x.shape[0], -1)

        val_logits = logits_from_feats(params_c, val_feats_flat)
        val_preds = np.argmax(np.array(val_logits), axis=-1)
        val_acc = float(np.mean((val_preds == np.array(val_dataset_y).squeeze()).astype(np.float32)))

        print(f"epoch {epoch} | train_loss {epoch_loss:.6f} | val_acc {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_q_np = q_np.copy()
            best_params_c = jax.tree.map(lambda x: x.copy(), params_c)

    test_x_scaled = np.asarray(test_dataset_x / float(Theta), dtype=np.float64, order="C")
    test_values = jnp.asarray(circ.forward(test_x_scaled, best_q_np))
    test_feats_flat = test_values.reshape(test_dataset_x.shape[0], -1)

    test_logits = logits_from_feats(best_params_c, test_feats_flat)
    test_preds = np.argmax(np.array(test_logits), axis=-1)

    cm, cls_accs, overall = calculate_final_metrics(test_dataset_y, test_preds, num_classes)

    print("\n=== Best on Validation ===")
    print(f"Best Val Accuracy: {best_val_acc:.4f} @ epoch {best_epoch}")
    print("\n=== Test @ Best-Validation Checkpoint ===")
    print(f"Test Overall Accuracy: {overall:.4f}")
    print("Class-wise Accuracy:")
    for i, acc in enumerate(cls_accs):
        print(f"  Class {i}: {acc:.4f}")

    return overall

results = []
num_qubit = 12
num_points = num_qubit // 2
test_learning_rate = 0.001
num_blocks_reupload = 12
sigma = 0.02
Theta = 1.7
use_augmentation = True
num_classes = 5
epochs = 1000
l2 = 0

for _base_seed in [1557,831, 121, 2023, 2024, 2025, 2026]:
    print(jax.default_backend())
    print(jax.devices())

    dataset_tag = "modelnet"
    dataset = np.load(f"modelnet40_5classes_{num_points}_1_fps_train700_val100_test200_new.npz")

    def result_one_dataset(dataset_tag_, dataset_):
        global scipy_rng, key, _global_subseed
        rng_pack = make_rng_pack(_base_seed, num_points, dataset_tag_)
        scipy_rng = rng_pack["scipy_rs"]
        key = rng_pack["base_key"]
        _global_subseed = rng_pack["subseed"]

        global train_dataset_x, train_dataset_y, val_dataset_x, val_dataset_y, test_dataset_x, test_dataset_y
        train_dataset_x = ensure_r1(dataset_["train_dataset_x"])
        train_dataset_y = dataset_["train_dataset_y"]
        val_dataset_x = ensure_r1(dataset_["val_dataset_x"])
        val_dataset_y = dataset_["val_dataset_y"]
        test_dataset_x = ensure_r1(dataset_["test_dataset_x"])
        test_dataset_y = dataset_["test_dataset_y"]

        return train(
            minibatch_size=35,
            Theta=Theta,
            epochs=epochs,
            num_blocks_reupload=num_blocks_reupload,
            num_points=num_points,
            use_augmentation=use_augmentation,
            sigma=sigma,
            num_classes=num_classes,
            l2=l2,
            learning_rate=test_learning_rate,
        )

    print(float(get_Theta(dataset)))
    print(_base_seed)
    a = result_one_dataset(dataset_tag, dataset)
    results.append(a)

    dataset_tag = "shapenet"
    dataset = np.load(f"shapenet_5classes_{num_points}_1_fps_train700_val100_test200_new.npz")
    print(float(get_Theta(dataset)))
    print(_base_seed)
    a = result_one_dataset(dataset_tag, dataset)
    results.append(a)

print(results)
