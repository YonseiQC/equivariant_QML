#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, math, hashlib, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.stats import special_ortho_group

import pennylane as qml
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from gates_fast import Spin_twirling, create_singlet

# --------------------------- Seed & JAX opts ---------------------------
jax.config.update("jax_enable_x64", True)

def make_subseed(base_seed: int, *keys) -> int:
    h = hashlib.sha256(str((base_seed,) + tuple(keys)).encode()).hexdigest()
    return int(h[:8], 16)

def make_rng_pack(base_seed: int, num_point: int, dataset_tag: str):
    subseed = make_subseed(base_seed, num_point, dataset_tag)
    scipy_rs = np.random.RandomState(subseed)     # for SO(3) sampling
    base_key = jax.random.PRNGKey(subseed)        # for JAX
    return dict(subseed=subseed, scipy_rs=scipy_rs, base_key=base_key)

# globals (kept for minimal code touch)
scipy_rng = None
key = None
_global_subseed = None

# --------------------------- Utils ---------------------------
def get_Theta(npz):
    """데이터 전체에서 (centroid→0) 후 최댓반경 × 1.2."""
    pts = np.array(npz['train_dataset_x'])
    pts = pts - pts.mean(axis=1, keepdims=True)
    norms = np.sqrt((pts**2).sum(axis=-1))
    return float(norms.max() * 1.2)

def random_3d_rotation():
    return special_ortho_group.rvs(3, random_state=scipy_rng)

def apply_3d_rotation(points):
    R = random_3d_rotation()
    return jnp.asarray(points) @ jnp.asarray(R.T)

def add_jitter(points, key, sigma):
    return points + jax.random.normal(key, points.shape) * sigma

def apply_permutation(points):
    idx = scipy_rng.permutation(points.shape[0])
    return points[idx]

def apply_data_augmentation(points, key, is_training=True):
    if not is_training:
        return points
    # 필요 시 아래 라인 풀어서 사용
    key1, key2 = jax.random.split(key)
    points = add_jitter(points, key2, sigma)
    # points = apply_3d_rotation(points)
    # points = apply_permutation(points)
    return points

def augment_batch(batch_points, key, is_training=True):
    if not is_training:
        return batch_points
    B = batch_points.shape[0]
    keys = jax.random.split(key, B)
    fn = jax.vmap(lambda pts, k: apply_data_augmentation(pts, k, is_training))
    return fn(batch_points, keys)

def epoch_shuffle_numpy(x, y):
    idx = np.random.permutation(x.shape[0])
    return x[idx], y[idx]

# --------------------------- Model ---------------------------
class MyNN(nn.Module):
    @nn.compact
    def __call__(self, x):           # x: (B, H)
        x = jnp.expand_dims(x, -1)   # (B, H, 1)
        x = nn.Dense(4)(x); x = nn.tanh(x)
        x = nn.Dense(4)(x); x = nn.tanh(x)

        mean_pool = jnp.mean(x, axis=1)
        max_pool  = jnp.max(x, axis=1)
        min_pool  = jnp.min(x, axis=1)
        std_pool  = jnp.std(x, axis=1)
        sum_pool  = jnp.sum(x, axis=1)
        var_pool  = jnp.var(x, axis=1)
        x = jnp.concatenate([mean_pool, max_pool, min_pool, sum_pool, std_pool, var_pool], axis=-1)

        x = nn.Dense(24)(x); x = nn.tanh(x)
        x = nn.Dense(24)(x); x = nn.tanh(x)
        x = nn.Dense(3)(x)           # C=3 고정
        return x

def NN_circuit(dataset, params):
    return MyNN().apply(params["c"], dataset)

# --------------------------- Metrics (torch 포맷과 동일) ---------------------------
def calculate_final_metrics(y_true, y_pred, num_classes):
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()
    cm = confusion_matrix(y_true_np, y_pred_np, labels=np.arange(num_classes))
    overall = np.trace(cm) / np.sum(cm)
    class_acc = []
    for i in range(num_classes):
        denom = cm[i].sum()
        class_acc.append((cm[i, i] / denom) if denom > 0 else 0.0)
    return cm, class_acc, overall

# --------------------------- Quantum Encoding / Ham ---------------------------
def encode(point, num_qubit):
    # point: (B, N, 3)
    norms = jnp.linalg.norm(point, axis=-1)  # (B, N)
    nx, ny, nz = point[:,:,0]/norms, point[:,:,1]/norms, point[:,:,2]/norms
    nx_T, ny_T, nz_T = nx.T, ny.T, nz.T
    norms_T = norms.T

    for i in range(int(num_qubit/2)):  # pair index
        cos_n = jnp.cos(norms_T[i])
        sin_n = jnp.sin(norms_T[i])
        nx_i, ny_i, nz_i = nx_T[i], ny_T[i], nz_T[i]
        U = jnp.array([
            [cos_n + 1j*sin_n*nz_i, 1j*sin_n*nx_i + sin_n*ny_i],
            [1j*sin_n*nx_i - sin_n*ny_i, cos_n - 1j*sin_n*nz_i]
        ]).transpose(2,0,1)  # (B,2,2)
        qml.QubitUnitary(U, wires=2*i)

def create_Hamiltonian(num_point):
    terms = []
    for i in range(num_point-1):
        for j in range(i+1, num_point):
            terms.append(
                (qml.PauliX(2*i)+qml.PauliX(2*i+1)) @ (qml.PauliX(2*j)+qml.PauliX(2*j+1)) +
                (qml.PauliY(2*i)+qml.PauliY(2*i+1)) @ (qml.PauliY(2*j)+qml.PauliY(2*j+1)) +
                (qml.PauliZ(2*i)+qml.PauliZ(2*i+1)) @ (qml.PauliZ(2*j)+qml.PauliZ(2*j+1))
            )
            terms.append(
                (qml.PauliX(2*i)-qml.PauliX(2*i+1)) @ (qml.PauliX(2*j)-qml.PauliX(2*j+1)) +
                (qml.PauliY(2*i)-qml.PauliY(2*i+1)) @ (qml.PauliY(2*j)-qml.PauliY(2*j+1)) +
                (qml.PauliZ(2*i)-qml.PauliZ(2*i+1)) @ (qml.PauliZ(2*j)-qml.PauliZ(2*j+1))
            )
    return terms

def prepare_init_state(num_qubit):
    for i in range(0, num_qubit, 2):
        create_singlet(i, i+1)

def create_twirling_circuit(num_qubit, num_blocks_reupload, num_reupload, Theta, H):
    def twirling_circuit(params, data_pt):
        # data_pt: (B, R, N, 3)
        prepare_init_state(num_qubit)
        k = 0
        for r in range(num_reupload):
            data = data_pt[:, r, :, :]  # (B, N, 3)
            encode(data/Theta, num_qubit)
            for _ in range(num_blocks_reupload):
                for p in range(2, int(num_qubit/2)+1):
                    Spin_twirling(params["q"][k], params["q"][k+1], p, wires=range(num_qubit))
                    k += 2
        return [qml.expval(h) for h in H]
    return twirling_circuit

# --------------------------- Train (VAL select -> TEST once) ---------------------------
def train(gate_type, minibatch, Theta, epochs, key, init_scale,
          num_blocks_reupload, num_qubit, num_reupload, use_augmentation,
          learning_rate, l2, num_classes, dev):

    global train_dataset_x, train_dataset_y, val_dataset_x, val_dataset_y, test_dataset_x, test_dataset_y
    assert len(train_dataset_x) == len(train_dataset_y)
    assert len(train_dataset_x) % minibatch == 0

    H = create_Hamiltonian(int(num_qubit/2))

    # ----- init params -----
    init_u2 = init_scale * math.pi / (2 * (int(num_qubit/2)-1) * (num_blocks_reupload * num_reupload))
    qkey = jax.random.PRNGKey(make_subseed(_global_subseed, 'init_q'))
    params_q = init_u2 * jax.random.uniform(qkey, (2 * (int(num_qubit/2)-1) * (num_blocks_reupload * num_reupload),))

    dummy_input = jnp.ones((1, 2 * math.comb(int(num_qubit/2), 2)))
    params_c = MyNN().init(key, dummy_input)
    params = {"q": params_q, "c": params_c}

    tqc = qml.QNode(
        create_twirling_circuit(num_qubit, num_blocks_reupload, num_reupload, Theta, H),
        device=dev, interface='jax'
    )
    tqc = jax.jit(tqc)

    def forward_expval(params, x_batch):
        expval = (jnp.array(tqc(params, x_batch))).T    # (B, #H)
        logits = NN_circuit(expval, params)              # (B, C)
        return logits

    def loss_fn(params, x_batch, y_batch):
        logits = forward_expval(params, x_batch)
        ce = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(logits, y_batch))
        l2_penalty = l2 * jnp.sum(jnp.array([jnp.sum(jnp.square(p)) for p in jax.tree.leaves(params)]))
        return ce + l2_penalty

    def accuracy(params, x, y):
        logits = forward_expval(params, x)
        preds = jnp.argmax(logits, axis=-1)
        return jnp.mean((preds == y.squeeze()).astype(jnp.float32))

    opt = optax.adam(learning_rate)
    opt_state = opt.init(params)

    best_val_acc = -1.0
    best_epoch = -1
    params_best = params

    B_total = len(train_dataset_x)
    num_batches = B_total // minibatch

    print(f"\n🚀 Training with validation selection for {epochs} epochs")
    for epoch in range(epochs):
        # shuffle samples (dataset index만)
        np.random.seed(make_subseed(_global_subseed, 'shuffle', epoch))
        xs, ys = epoch_shuffle_numpy(train_dataset_x, train_dataset_y)
        if use_augmentation:
            epoch_key = jax.random.fold_in(key, epoch)
            xs_cur = augment_batch(xs, epoch_key, is_training=True)
        else:
            xs_cur = xs
        xs_cur = xs_cur.reshape(num_batches, minibatch, num_reupload, -1, 3)
        ys_cur = ys.reshape(num_batches, minibatch)

        # ----- one epoch -----
        running_train = 0.0
        for i in range(num_batches):
            loss, grad = jax.value_and_grad(loss_fn)(params, xs_cur[i], ys_cur[i])
            updates, opt_state = opt.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            running_train += float(loss)

        train_loss = running_train / max(1, num_batches)

        # ----- validation -----
        val_loss = float(loss_fn(params, val_dataset_x, val_dataset_y))
        val_acc  = float(accuracy(params, val_dataset_x, val_dataset_y))

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # cheap copy
            params_best = jax.tree.map(lambda x: x.copy() if hasattr(x, 'copy') else x, params)

        print(f"[Epoch {epoch:3d}] TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
              f"ValAcc {val_acc:.4f} (Best {best_val_acc:.4f} @ {best_epoch})")

    # ===== TEST @ Best-Validation =====
    print("\n=== Evaluate on TEST with Best-Validation Checkpoint ===")
    logits = forward_expval(params_best, test_dataset_x)
    preds = jnp.argmax(logits, axis=-1)
    cm, class_acc, overall_acc = calculate_final_metrics(test_dataset_y, preds, num_classes)

    print(f"Test Overall Accuracy: {overall_acc:.4f}")
    print("Class-wise Accuracies:")
    for i, a in enumerate(class_acc):
        print(f"  Class {i}: {a:.4f}")
    print("\nConfusion Matrix:\n", cm)

    return overall_acc

# --------------------------- Config / Dataset ---------------------------
results = []
num_qubit = 12             
num_reupload = 1
gate_type = "u2"
num_blocks_reupload = 12
init_scale = 0.02
sigma = 0.01
Theta = 1.7
use_augmentation = True
num_classes = 3
epochs = 1000
l2 = 0.0
minibatch_size = 35
learning_rate = 0.001

dev = qml.device("default.qubit", wires=num_qubit)

for _base_seed in [831, 121, 2023, 2024, 2025, 2026]:
    dataset_tag = 'SUO'
    npz_name = f'SUO_3classes_{int(num_qubit/2)}_{num_reupload}_fps_train700_val100_test200_new.npz'
    data = np.load(npz_name)

    _Theta = get_Theta(data)
    print("\n" + "=" * 60)
    print(f"📁 Loading dataset: {npz_name}")
    print(f"[DATASET] tag={dataset_tag} | points={int(num_qubit/2)} | reupload={num_reupload}")
    print(f"[SEED   ] {_base_seed}")
    print(f"[Theta  ] {_Theta:.6f}")
    print("=" * 60)

    def ensure_reupload_dim(x):
        if x.ndim == 3:
            return x.reshape(x.shape[0], num_reupload, -1, 3)
        elif x.ndim == 4:
            return x
        else:
            raise ValueError(f"Unexpected shape: {x.shape}")

    train_dataset_x = ensure_reupload_dim(data['train_dataset_x'])
    train_dataset_y = data['train_dataset_y']
    val_dataset_x   = ensure_reupload_dim(data['val_dataset_x'])
    val_dataset_y   = data['val_dataset_y']
    test_dataset_x  = ensure_reupload_dim(data['test_dataset_x'])
    test_dataset_y  = data['test_dataset_y']

    # RNG pack & seeds
    _num_point = int(num_qubit/2)
    _rng = make_rng_pack(_base_seed, _num_point, dataset_tag)
    scipy_rng       = _rng['scipy_rs']
    key             = _rng['base_key']
    _global_subseed = _rng['subseed']

    print('🎯 Hyperparameters (HyQuRP-lite, JAX)')
    print(f'   num_points={_num_point}, epochs={epochs}, minibatch={minibatch_size}, weight_decay(l2)={l2}')

    acc = train(gate_type, minibatch_size, Theta, epochs, key, init_scale,
                num_blocks_reupload, num_qubit, num_reupload, use_augmentation,
                learning_rate, l2=l2, num_classes=num_classes, dev=dev)
    results.append(float(acc))

print("\n" + "#" * 60)
print("[RESULTS]", [round(x, 6) for x in results])
print("#" * 60)
