import os
import argparse
from pathlib import Path
import hyqurp_cpp
hyqurp_cpp.kokkos_initialize({})
from hyqurp_cpp import QRPCircuitC128, QRPCircuitC64

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("seed", type=int)
_pre.add_argument("--dataset")
_pre.add_argument("--num_qubit", type=int, required=True)
_pre.add_argument("--variant", type=str, choices=["light", "mid"])
_pre_args, _ = _pre.parse_known_args()

_num_points_pre = _pre_args.num_qubit // 2
_enable_x64 = True if _num_points_pre <= 5 else False

if _enable_x64:
    QRPCircuit = QRPCircuitC128
else:
    QRPCircuit = QRPCircuitC64

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"

import jax
jax.config.update("jax_enable_x64", _enable_x64)

import datetime
import hashlib
import math

import jax.numpy as jnp
import numpy as np
import optax
import pennylane as qml
from flax import linen as nn
from scipy.stats import special_ortho_group
from sklearn.metrics import confusion_matrix

from gates_fast import Spin_twirling, create_singlet

tree_leaves = jax.tree_util.tree_leaves
tree_map = jax.tree_util.tree_map

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

def get_Theta(npz):
    x = npz["train_dataset_x"]
    x = jnp.array(x)
    if x.ndim == 4:
        x = x.reshape(-1, x.shape[-2], x.shape[-1])
    x = x - jnp.mean(x, axis=0)
    norms = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1))
    return jnp.max(norms) * 1.2

def random_3d_rotation():
    return special_ortho_group.rvs(3, random_state=scipy_rng)

def apply_3d_rotation(points):
    rotation_matrix = random_3d_rotation()
    rotation_matrix_jax = jnp.array(rotation_matrix)
    return jnp.dot(points, rotation_matrix_jax.T)

def add_jitter(points, key_, sigma_):
    noise = jax.random.normal(key_, points.shape) * sigma_
    return points + noise

def apply_permutation(points):
    num_points_ = points.shape[0]
    perm_indices = scipy_rng.permutation(num_points_)
    return points[perm_indices]

def apply_data_augmentation(points, key_, sigma_, is_training=True):
    if not is_training:
        return points
    _, key2 = jax.random.split(key_)
    augmented_points = add_jitter(points, key2, sigma_)
    return augmented_points

def augment_batch(batch_points, key_, sigma_, is_training=True):
    if not is_training:
        return batch_points
    batch_size_ = batch_points.shape[0]
    keys = jax.random.split(key_, batch_size_)
    augment_fn = jax.vmap(lambda pts, k: apply_data_augmentation(pts, k, sigma_, is_training))
    return augment_fn(batch_points, keys)

def epoch_shuffle_numpy(x, y):
    idx = np.random.permutation(x.shape[0])
    return x[idx], y[idx]

class MyNNLight(nn.Module):
    num_pairs: int
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = jnp.expand_dims(x, axis=-1).reshape(x.shape[0], self.num_pairs, 2)
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

        x = jnp.concatenate([mean_pool, max_pool, min_pool, sum_pool, std_pool, var_pool], axis=-1)

        x = nn.Dense(features=24)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=24)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x

class MyNNMid(nn.Module):
    num_pairs: int
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = jnp.expand_dims(x, -1).reshape(x.shape[0], self.num_pairs, 2)
        x = nn.Dense(8)(x)
        x = nn.tanh(x)
        x = nn.Dense(16)(x)
        x = nn.tanh(x)
        x = nn.Dense(32)(x)
        x = nn.tanh(x)

        mean_pool = jnp.mean(x, axis=1)
        max_pool = jnp.max(x, axis=1)
        min_pool = jnp.min(x, axis=1)
        std_pool = jnp.std(x, axis=1)
        sum_pool = jnp.sum(x, axis=1)
        var_pool = jnp.var(x, axis=1)

        x = jnp.concatenate([mean_pool, max_pool, min_pool, sum_pool, std_pool, var_pool], axis=-1)

        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        x = nn.Dense(16)(x)
        x = nn.tanh(x)
        x = nn.Dense(8)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.num_classes)(x)
        return x

def calculate_final_metrics(y_true, y_pred, num_classes_):
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()
    cm = confusion_matrix(y_true_np, y_pred_np, labels=range(num_classes_))
    class_accuracies = []
    for i in range(num_classes_):
        if np.sum(cm[i, :]) > 0:
            class_acc = cm[i, i] / np.sum(cm[i, :])
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    overall_accuracy = np.trace(cm) / np.sum(cm)
    return cm, class_accuracies, overall_accuracy

def analyze_gradient_norms(grad):
    q_grad_sqr_sum = float(jnp.sum(jnp.square(grad["q"])))
    c_grad_leaves = jax.tree.leaves(grad["c"])
    c_grad_sqr_sum = sum(jnp.sum(jnp.square(g)) for g in c_grad_leaves)
    return jnp.sqrt(q_grad_sqr_sum), jnp.sqrt(c_grad_sqr_sum), jnp.sqrt(q_grad_sqr_sum + c_grad_sqr_sum)

def ensure_reupload_dim(x, num_reupload):
    if x.ndim == 3:
        return x.reshape(x.shape[0], num_reupload, -1, 3)
    if x.ndim == 4:
        return x
    raise ValueError(f"Unexpected shape for point cloud: {x.shape}")


train_dataset_x = None
train_dataset_y = None
val_dataset_x = None
val_dataset_y = None
test_dataset_x = None
test_dataset_y = None

num_pairs = None
num_points = None
num_classes = None
variant = None
dev = None

def NN_apply(params_c, x):
    if variant == "light":
        return MyNNLight(num_pairs=num_pairs, num_classes=num_classes).apply(params_c, x)
    return MyNNMid(num_pairs=num_pairs, num_classes=num_classes).apply(params_c, x)

def train(
    gate_type,
    minibatch_size,
    Theta_,
    epochs,
    key_,
    init_scale,
    num_blocks_reupload,
    num_qubit_,
    num_reupload,
    use_augmentation,
    sigma_,
    l2_,
    learning_rate,
):
    global train_dataset_x, train_dataset_y, val_dataset_x, val_dataset_y, test_dataset_x, test_dataset_y

    assert len(train_dataset_x) == len(train_dataset_y)
    assert len(train_dataset_x) % minibatch_size == 0

    batch_size = len(train_dataset_x)

    if gate_type != "u2":
        raise ValueError("Only gate_type='u2' is supported in this script.")

    init_u2 = init_scale * math.pi / (2 * (int(num_qubit_ / 2) - 1) * (num_blocks_reupload * num_reupload))
    qkey = jax.random.PRNGKey(make_subseed(_global_subseed, "init_q"))
    params_q = init_u2 * jax.random.uniform(
        qkey, shape = (num_blocks_reupload, (int(num_qubit_ / 2) - 1), 2)
    )

    dummy_input = jnp.ones((1, 2 * num_pairs))
    if variant == "light":
        params_c = MyNNLight(num_pairs=num_pairs, num_classes=num_classes).init(key_, dummy_input)
    else:
        params_c = MyNNMid(num_pairs=num_pairs, num_classes=num_classes).init(key_, dummy_input)

    params = {"q": params_q, "c": params_c}
    
    _cpp_circuit = QRPCircuit(num_qubit_ // 2)
    cpu_device = jax.devices("cpu")[0]

    if jax.devices('gpu'):
        gpu_device = jax.devices('gpu')[0]
    else:
        gpu_device = jax.devices('cpu')[0]

    def circuit(points):
        points_cpu = jax.device_put(points, device=cpu_device)

        @jax.custom_vjp
        def f(params):
            params_cpu = jax.device_put(params, device=cpu_device)
            return jax.device_put(_cpp_circuit.f(points_cpu, params_cpu), gpu_device)

        def f_fwd(params):
            params_cpu = jax.device_put(params, device=cpu_device)
            val, states = _cpp_circuit.forward(points_cpu, params_cpu)
            return jax.device_put(val, gpu_device), (states, params_cpu)

        def f_bwd(res, g):
            states, params_cpu = res
            g_cpu = jax.device_put(g, device=cpu_device)
            vjp = _cpp_circuit.vjp(params_cpu, states, g_cpu)
            return (jax.device_put(vjp, gpu_device),)

        f.defvjp(f_fwd, f_bwd)
        return f

    NN_jit = jax.jit(NN_apply)
    
    def forward_expval(params_, x_batch):
        qnode = circuit(x_batch)
        expval_ham = qnode(params_["q"])
        logits = NN_jit(params_["c"], expval_ham)
        return logits

    def loss_fn(params_, x_batch, y_batch):
        logits = forward_expval(params_, x_batch)
        loss = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(logits, y_batch))
        if l2_ == 0:
            return loss
        l2_penalty = l2_ * jnp.sum(jnp.array([jnp.sum(jnp.square(p)) for p in tree_leaves(params_)]))
        return loss + l2_penalty

    def accuracy_fn(params_, x, y):
        logits = forward_expval(params_, x)
        preds = jnp.argmax(logits, axis=-1)
        return jnp.mean((preds == y.squeeze()).astype(jnp.float32))

    solver = optax.adam(learning_rate=learning_rate)
    opt_state = solver.init(params)

    best_val_acc = -jnp.inf
    best_epoch = -1
    params_best = params

    for epoch in range(epochs):
        print(f"epoch {epoch}")

        np.random.seed(make_subseed(_global_subseed, "shuffle", epoch))
        xs, ys = epoch_shuffle_numpy(train_dataset_x, train_dataset_y)

        if use_augmentation:
            epoch_key = jax.random.fold_in(key_, epoch)
            current_train_x = augment_batch(xs, epoch_key, sigma_, is_training=True)
        else:
            current_train_x = xs

        num_batches = batch_size // minibatch_size
        current_train_x = current_train_x.reshape(num_batches, minibatch_size, -1, 3)
        train_y_batched = ys.reshape(num_batches, minibatch_size)

        epoch_train_loss = 0.0
        epoch_q_grad_norms = []
        epoch_c_grad_norms = []
        epoch_total_grad_norms = []

        for i in range(num_batches):
            loss, grad = jax.value_and_grad(loss_fn)(params, current_train_x[i], train_y_batched[i])

            qn, cn, tn = analyze_gradient_norms(grad)
            epoch_q_grad_norms.append(qn)
            epoch_c_grad_norms.append(cn)
            epoch_total_grad_norms.append(tn)

            updates, opt_state = solver.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            epoch_train_loss += loss / num_batches

        avg_q = jnp.mean(jnp.array(epoch_q_grad_norms))
        avg_c = jnp.mean(jnp.array(epoch_c_grad_norms))
        avg_tot = jnp.mean(jnp.array(epoch_total_grad_norms))

        print(f"\nTrain Loss: {epoch_train_loss}")

        val_loss = loss_fn(params, val_dataset_x, val_dataset_y)
        val_acc = accuracy_fn(params, val_dataset_x, val_dataset_y)

        now = datetime.datetime.now()
        print(f"Iteration done at {now}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Accuracy: {val_acc}")
        print(f"Epoch Avg Gradients - Q: {avg_q:.1e}, C: {avg_c:.1e}, Total: {avg_tot:.1e}")
        print("-" * 50)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            params_best = tree_map(lambda x: x.copy(), params)

    print("\n=== Best on Validation ===")
    print(f"Best Val Accuracy: {float(best_val_acc):.4f} @ epoch {best_epoch}")

    logits = forward_expval(params_best, test_dataset_x)
    preds = jnp.argmax(logits, axis=-1)
    cm, cls_accs, overall = calculate_final_metrics(test_dataset_y, preds, num_classes)

    print("\n=== Test @ Best-Validation Checkpoint ===")
    print(f"Test Overall Accuracy: {float(overall):.4f}")
    print("Class-wise Accuracy:")
    for i, acc in enumerate(cls_accs):
        print(f"  Class {i}: {acc:.4f}")

    return overall


def main():
    global scipy_rng, key, _global_subseed
    global train_dataset_x, train_dataset_y, val_dataset_x, val_dataset_y, test_dataset_x, test_dataset_y
    global num_pairs, num_points, num_classes, variant, dev

    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("--dataset", type=str, choices=["modelnet", "shapenet", "suo"], required=True)
    parser.add_argument("--num_qubit", type=int, required=True)
    parser.add_argument("--variant", type=str, choices=["light", "mid"], required=True)
    args = parser.parse_args()

    base_seed = args.seed
    dataset_tag = args.dataset
    num_qubit = args.num_qubit
    variant = args.variant

    assert num_qubit % 2 == 0
    num_points = num_qubit // 2
    num_pairs = num_points * (num_points - 1) // 2

    # define hyperparameters
    num_reupload = 1
    Theta = 1.7
    sigma = 0.02
    use_augmentation = True
    epochs = 1000
    l2 = 0
    gate_type = "u2"
    num_blocks_reupload = 12
    init_scale = 0.02
    learning_rate = 0.001

    dev = qml.device("default.qubit", wires=num_qubit)

    if dataset_tag == "modelnet":
        num_classes = 5
        npz_name = f"modelnet40_5classes_{num_points}_{num_reupload}_fps_train700_val100_test200_new.npz"
    elif dataset_tag == "shapenet":
        num_classes = 5
        npz_name = f"shapenet_5classes_{num_points}_{num_reupload}_fps_train700_val100_test200_new.npz"
    else:
        num_classes = 3
        npz_name = f"SUO_3classes_{num_points}_{num_reupload}_fps_train700_val100_test200_new.npz"

    HERE = Path(__file__).resolve().parent
    REPO = HERE.parent

    if dataset_tag == "modelnet":
        dataset_path = REPO / "data" / "ModelNet" / npz_name
    elif dataset_tag == "shapenet":
        dataset_path = REPO / "data" / "ShapeNet" / npz_name
    else:
        dataset_path = REPO / "data" / "Sydney_Urban_Objects" / npz_name

    dataset = np.load(dataset_path)

    print(f"Using seed={base_seed}")
    print(f"dataset={dataset_tag}, variant={variant}, num_qubit={num_qubit}, num_points={num_points}, num_pairs={num_pairs}")
    print(f"Theta_fixed={Theta}, sigma={sigma}, epochs={epochs}, lr={learning_rate}")
    print(f"Theta_from_data(debug)={float(get_Theta(dataset))}")
    print(f"jax_enable_x64={_enable_x64}")

    train_dataset_x = ensure_reupload_dim(dataset["train_dataset_x"], num_reupload)
    train_dataset_y = dataset["train_dataset_y"]
    val_dataset_x = dataset["val_dataset_x"]
    val_dataset_y = dataset["val_dataset_y"]
    test_dataset_x = dataset["test_dataset_x"]
    test_dataset_y = dataset["test_dataset_y"]

    rng_pack = make_rng_pack(base_seed, num_points, dataset_tag)
    scipy_rng = rng_pack["scipy_rs"]
    key = rng_pack["base_key"]
    _global_subseed = rng_pack["subseed"]

    overall = train(
        gate_type=gate_type,
        minibatch_size=35,
        Theta_=Theta,
        epochs=epochs,
        key_=key,
        init_scale=init_scale,
        num_blocks_reupload=num_blocks_reupload,
        num_qubit_=num_qubit,
        num_reupload=num_reupload,
        use_augmentation=use_augmentation,
        sigma_=sigma,
        l2_=l2,
        learning_rate=learning_rate,
    )

    print(float(overall))

if __name__ == "__main__":
    main()
