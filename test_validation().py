#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pennylane as qml
import jax
import jax.numpy as jnp
import optax
from gates_fast import Spin_twirling, create_singlet
import math
import pennylane.numpy as pnp
from flax import linen as nn
from scipy.stats import special_ortho_group
import hashlib  # [RNG] 추가

# --------------------------- Seed & JAX opts ---------------------------
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

# --------------------------- Utils ---------------------------
def get_Theta(point):
    point = point['train_dataset_x']
    point = point - jnp.mean(point, axis=0)
    point_sqr = jnp.power(point, 2)
    norms = jnp.sqrt(jnp.sum(point_sqr, axis=-1))
    Theta = jnp.max(norms) * 1.2
    return Theta

def random_3d_rotation():
    # [RNG] 전역 scipy_rng 사용(동일 이름이므로 다른 코드 수정 없음)
    return special_ortho_group.rvs(3, random_state=scipy_rng)

def apply_3d_rotation(points):
    rotation_matrix = random_3d_rotation()
    rotation_matrix_jax = jnp.array(rotation_matrix)
    return jnp.dot(points, rotation_matrix_jax.T)

def add_jitter(points, key, sigma):
    noise = jax.random.normal(key, points.shape) * sigma
    return points + noise

def apply_permutation(points):
    num_points = points.shape[0]
    # [RNG] 전역 scipy_rng 사용 유지
    perm_indices = scipy_rng.permutation(num_points)
    return points[perm_indices]

def apply_data_augmentation(points, key, is_training=True):
    if not is_training:
        return points
    key1, key2 = jax.random.split(key)
    augmented_points = add_jitter(points, key2, sigma)
    # 필요 시 아래 두 줄 활성화
    # augmented_points = apply_3d_rotation(augmented_points)
    # augmented_points = apply_permutation(augmented_points)
    return augmented_points

def augment_batch(batch_points, key, is_training=True):
    if not is_training:
        return batch_points
    batch_size = batch_points.shape[0]
    keys = jax.random.split(key, batch_size)
    augment_fn = jax.vmap(lambda points, k: apply_data_augmentation(points, k, is_training))
    return augment_fn(batch_points, keys)

# ====== (ADD) 에폭마다 샘플 순서만 섞기: DataLoader(shuffle=True)와 동일한 의미 ======
def epoch_shuffle_numpy(x, y):
    """샘플 순서(데이터셋 인덱스)만 섞습니다. 포인트 내부 순서는 그대로."""
    idx = np.random.permutation(x.shape[0])
    return x[idx], y[idx]

# --------------------------- Model ---------------------------
class MyNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.expand_dims(x, axis=-1)
        x = nn.Dense(features=4)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=4)(x)
        x = nn.tanh(x)

        mean_pool = jnp.mean(x, axis=1)
        max_pool  = jnp.max(x, axis=1)
        min_pool  = jnp.min(x, axis=1)
        std_pool  = jnp.std(x, axis=1)
        sum_pool  = jnp.sum(x, axis=1)
        var_pool  = jnp.var(x, axis=1)

        x = jnp.concatenate([mean_pool, max_pool, min_pool, sum_pool, std_pool, var_pool], axis=-1)

        x = nn.Dense(features=24)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=24)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=num_classes)(x)  
        return x

def NN_circuit(dataset, params):
    return MyNN().apply(params["c"], dataset)

# --------------------------- Metrics/Plots ---------------------------
def calculate_class_accuracies(y_true, y_pred, num_classes):
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()
    class_accuracies = []
    for i in range(num_classes):
        mask = (y_true_np == i)
        if np.sum(mask) > 0:
            class_acc = np.sum((y_pred_np[mask] == i)) / np.sum(mask)
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    return class_accuracies

def calculate_final_metrics(y_true, y_pred, num_classes):
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()
    cm = confusion_matrix(y_true_np, y_pred_np, labels=range(num_classes))
    class_accuracies = []
    for i in range(num_classes):
        if np.sum(cm[i, :]) > 0:
            class_acc = cm[i, i] / np.sum(cm[i, :])
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    overall_accuracy = np.trace(cm) / np.sum(cm)
    return cm, class_accuracies, overall_accuracy

def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names or range(len(cm)),
                yticklabels=class_names or range(len(cm)))
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def analyze_gradient_norms(grad):
    q_individual_grads = grad["q"]
    q_grad_norm = jnp.linalg.norm(grad["q"])
    c_grad_leaves = jax.tree.leaves(grad["c"])
    c_grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in c_grad_leaves))
    total_grad_norm = jnp.sqrt(jnp.sum(jnp.square(grad["q"])) +
                               sum(jnp.sum(jnp.square(g)) for g in c_grad_leaves))
    return q_individual_grads, q_grad_norm, c_grad_norm, total_grad_norm

# --------------------------- Quantum Encoding / Ham ---------------------------
def encode(point, num_qubit):
    # point: (B, R, N, 3)에서 R축을 내부 루프에서 인덱싱해 들어오므로 여기서는 (B, N, 3)
    point_sqr = jnp.power(point, 2)
    norms = jnp.sqrt(jnp.sum(point_sqr, axis=-1))
    nx, ny, nz = point[:,:,0] / norms, point[:,:,1] / norms, point[:,:,2] / norms

    norms_T = norms.T
    nx_T, ny_T, nz_T = nx.T, ny.T, nz.T

    for i in range(int(num_qubit / 2)):
        cos_n = jnp.cos(norms_T[i])
        sin_n = jnp.sin(norms_T[i])
        nx_i, ny_i, nz_i = nx_T[i], ny_T[i], nz_T[i]
        matrix_i = jnp.array([
            [cos_n + 1j * sin_n * nz_i, 1j * sin_n * nx_i + sin_n * ny_i],
            [1j * sin_n * nx_i - sin_n * ny_i, cos_n - 1j * sin_n * nz_i]
        ]).transpose(2, 0, 1)  
        qml.QubitUnitary(matrix_i, wires=2 * i)

def create_Hamiltonian(num_point):
    terms = []
    for i in range(num_point - 1):
        for j in range(i + 1, num_point):
            terms.append(
                (qml.PauliX(2 * i) + qml.PauliX(2 * i + 1)) @ (qml.PauliX(2 * j) + qml.PauliX(2 * j + 1)) +
                (qml.PauliY(2 * i) + qml.PauliY(2 * i + 1)) @ (qml.PauliY(2 * j) + qml.PauliY(2 * j + 1)) +
                (qml.PauliZ(2 * i) + qml.PauliZ(2 * i + 1)) @ (qml.PauliZ(2 * j) + qml.PauliZ(2 * j + 1))
            )
            terms.append(
                (qml.PauliX(2 * i) - qml.PauliX(2 * i + 1)) @ (qml.PauliX(2 * j) - qml.PauliX(2 * j + 1)) +
                (qml.PauliY(2 * i) - qml.PauliY(2 * i + 1)) @ (qml.PauliY(2 * j) - qml.PauliY(2 * j + 1)) +
                (qml.PauliZ(2 * i) - qml.PauliZ(2 * i + 1)) @ (qml.PauliZ(2 * j) - qml.PauliZ(2 * j + 1))
            )
    return terms

def prepare_init_state(num_qubit):
    for i in range(0, num_qubit, 2):
        create_singlet(i, i + 1)

def create_twirling_circuit(num_qubit, num_blocks_reupload, num_reupload, Theta, H):
    def twirling_circuit(params, data_pt):
        # data_pt: (B, R, N, 3)
        prepare_init_state(num_qubit)
        k = 0
        for i in range(num_reupload):
            data = data_pt[:, i, :, :]  # (B, N, 3)
            encode(data / Theta, num_qubit)
            for l in range(num_blocks_reupload):
                for p in range(2, int(num_qubit / 2) + 1):
                    Spin_twirling(params["q"][k], params["q"][k + 1], p, wires=range(num_qubit))
                    k += 2
        return [qml.expval(h) for h in H]
    return twirling_circuit

# --------------------------- Training (Val selection -> Test once) ---------------------------
def train(gate_type, minibatch_size, Theta, epochs, key, init_scale,
          num_blocks_reupload, num_qubit, num_reupload, use_augmentation, **adam_opt):

    global train_dataset_x, train_dataset_y, val_dataset_x, val_dataset_y, test_dataset_x, test_dataset_y
    assert len(train_dataset_x) == len(train_dataset_y)
    assert len(train_dataset_x) % minibatch_size == 0

    batch_size = len(train_dataset_x)
    ham = create_Hamiltonian(int(num_qubit / 2))

    if gate_type == "u2":
        init_u2 = init_scale * math.pi / (2 * (int(num_qubit / 2) - 1) * (num_blocks_reupload * num_reupload))
        qkey = jax.random.PRNGKey(make_subseed(_global_subseed, 'init_q'))
        params_q = init_u2 * jax.random.uniform(
            qkey,
            (2 * (int(num_qubit / 2) - 1) * (num_blocks_reupload * num_reupload),)
        )

        dummy_input = jnp.ones((1, math.comb(int(num_qubit / 2), 2)))
        params_c = MyNN().init(key, dummy_input)

        params = {"q": params_q, "c": params_c}

        twirling_circuit = qml.QNode(
            create_twirling_circuit(num_qubit, num_blocks_reupload, num_reupload, Theta, ham),
            device=dev, interface='jax'
        )
        twirling_circuit = jax.jit(twirling_circuit)

        def forward_expval(params, x_batch):
            expval_ham = (jnp.array(twirling_circuit(params, x_batch))).T  # (B, #H)
            logits = NN_circuit(expval_ham, params)                        # (B, C)
            return logits

        def loss_fn(params, x_batch, y_batch, l2):
            logits = forward_expval(params, x_batch)
            loss = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(logits, y_batch))
            l2_penalty = l2 * jnp.sum(jnp.array([jnp.sum(jnp.square(p)) for p in jax.tree.leaves(params)]))
            return loss + l2_penalty

        def accuracy(params, x, y):
            logits = forward_expval(params, x)
            preds = jnp.argmax(logits, axis=-1)
            return jnp.mean((preds == y.squeeze()).astype(jnp.float32))

        solver = optax.adam(**adam_opt)
        opt_state = solver.init(params)

        # 기록
        train_loss_lst, val_loss_lst, val_acc_lst = [], [], []
        q_grad_norms, c_grad_norms, total_grad_norms = [], [], []

        # 최고 val 기준 체크포인트
        best_val_acc = -jnp.inf
        best_epoch = -1
        params_best = params

        # ===== Training loop (PyTorch DataLoader shuffle=True와 동일 의미) =====
        for epoch in range(epochs):
            print(f"epoch {epoch}")

            np.random.seed(make_subseed(_global_subseed, 'shuffle', epoch))  
            xs, ys = epoch_shuffle_numpy(train_dataset_x, train_dataset_y)

            if use_augmentation:
                epoch_key = jax.random.fold_in(key, epoch)        
                current_train_x = augment_batch(xs, epoch_key, is_training=True)
            else:
                current_train_x = xs

            num_batches = batch_size // minibatch_size
            current_train_x = current_train_x.reshape(num_batches, minibatch_size, num_reupload, -1, 3)
            train_y_batched = ys.reshape(num_batches, minibatch_size)

            epoch_train_loss = 0.0
            epoch_q_grad_norms, epoch_c_grad_norms, epoch_total_grad_norms = [], [], []

            for i in range(num_batches):
                loss, grad = jax.value_and_grad(loss_fn, argnums=0)(params, current_train_x[i], train_y_batched[i], l2)

                # grad stats
                q_individual_grads, q_grad_norm, c_grad_norm, total_grad_norm = analyze_gradient_norms(grad)
                epoch_q_grad_norms.append(q_grad_norm)
                epoch_c_grad_norms.append(c_grad_norm)
                epoch_total_grad_norms.append(total_grad_norm)

                updates, opt_state = solver.update(grad, opt_state, params)
                params = optax.apply_updates(params, updates)
                epoch_train_loss += (loss / num_batches)

            # 기록
            avg_q = jnp.mean(jnp.array(epoch_q_grad_norms))
            avg_c = jnp.mean(jnp.array(epoch_c_grad_norms))
            avg_tot = jnp.mean(jnp.array(epoch_total_grad_norms))
            q_grad_norms.append(avg_q); c_grad_norms.append(avg_c); total_grad_norms.append(avg_tot)

            train_loss_lst.append(epoch_train_loss)
            print(f"\nTrain Loss: {epoch_train_loss}")

            # ----- Validation (no shuffle) -----
            val_loss = loss_fn(params, val_dataset_x, val_dataset_y, l2)
            val_acc = accuracy(params, val_dataset_x, val_dataset_y)
            val_loss_lst.append(val_loss); val_acc_lst.append(val_acc)

            print(f"Val Loss: {val_loss}")
            print(f"Val Accuracy: {val_acc}")
            print(f"Epoch Avg Gradients - Q: {avg_q:.1e}, C: {avg_c:.1e}, Total: {avg_tot:.1e}")
            print("-" * 50)

            # 베스트 갱신 시 파라미터 스냅샷
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                params_best = jax.tree.map(lambda x: x.copy(), params)

        # ===== 학습 종료 후: 베스트 val 기준으로 TEST 1회 평가 =====
        print("\n=== Best on Validation ===")
        print(f"Best Val Accuracy: {float(best_val_acc):.4f} @ epoch {best_epoch}")

        def evaluate_on_test(params_eval):
            logits = forward_expval(params_eval, test_dataset_x)
            preds = jnp.argmax(logits, axis=-1)
            cm, cls_accs, overall = calculate_final_metrics(test_dataset_y, preds, num_classes)
            return preds, cm, cls_accs, overall

        final_predictions, final_cm, final_class_acc, final_overall_acc = evaluate_on_test(params_best)

        print("\n=== Test @ Best-Validation Checkpoint ===")
        print(f"Test Overall Accuracy: {float(final_overall_acc):.4f}")
        print("Class-wise Accuracy:")
        for i, acc in enumerate(final_class_acc):
            print(f"  Class {i}: {acc:.4f}")
        # print("\nConfusion Matrix:\n", final_cm)

        # # ---- Plot ----
        # plt.figure(figsize=(15, 10))
        # plt.subplot(2, 2, 1)
        # plt.plot(train_loss_lst, label='Train Loss', marker='o')
        # plt.plot(val_loss_lst,   label='Val Loss',   marker='o')
        # plt.title('Loss')
        # plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend()

        # plt.subplot(2, 2, 2)
        # plt.plot(val_acc_lst, label='Val Accuracy', marker='o')
        # plt.axvline(best_epoch, linestyle='--', label='Best Val', alpha=0.6)
        # plt.title('Val Accuracy')
        # plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.grid(True); plt.legend()

        # plt.subplot(2, 2, 3)
        # plt.semilogy(q_grad_norms, label='Quantum', marker='o')
        # plt.semilogy(c_grad_norms, label='Classical', marker='s')
        # plt.semilogy(total_grad_norms, label='Total', marker='^')
        # plt.title('Gradient Norms (log)')
        # plt.xlabel('Epoch'); plt.ylabel('Grad Norm'); plt.grid(True); plt.legend()

        # plt.subplot(2, 2, 4)
        # sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues')
        # plt.title('Confusion Matrix (Test @ Best Val)')
        # plt.xlabel('Predicted'); plt.ylabel('True')

        # plt.tight_layout(); plt.show()

        # print("\n=== Classification Report (Test @ Best Val) ===")
        # print(classification_report(np.array(test_dataset_y).flatten(),
        #                             np.array(final_predictions).flatten(),
        #                             target_names=[f'Class {i}' for i in range(num_classes)]))
    return final_overall_acc

# --------------------------- Config / Dataset ---------------------------
results = []
num_qubit = 8
num_reupload = 1
gate_type = "u2" 
test_learning_rate = 0.001
num_blocks_reupload = 12
init_scale = 0.02
dev = qml.device("default.qubit", wires=num_qubit)
sigma = 0.02
Theta = 1.7
use_augmentation = True
_base_seed = 1557
num_classes = 5
epochs = 1000
l2 = 0

for _base_seed in [1557, 831, 121, 2023, 2024, 2025, 2026]:

    dataset_tag = 'modelnet'  ; dataset = np.load(f'modelnet40_5classes_{int(num_qubit/2)}_{num_reupload}_fps_train700_val100_test200_new.npz')
    print(get_Theta(dataset))
    print(_base_seed)

    def ensure_reupload_dim(x):
        if x.ndim == 3:
            return x.reshape(x.shape[0], num_reupload, -1, 3)
        elif x.ndim == 4:
            return x
        else:
            raise ValueError(f"Unexpected shape for point cloud: {x.shape}")

    train_dataset_x = ensure_reupload_dim(dataset['train_dataset_x'])  
    train_dataset_y = dataset['train_dataset_y']                       
    val_dataset_x   = ensure_reupload_dim(dataset['val_dataset_x'])   
    val_dataset_y   = dataset['val_dataset_y']                        
    test_dataset_x  = ensure_reupload_dim(dataset['test_dataset_x'])   
    test_dataset_y  = dataset['test_dataset_y']                       



    def result(gate_type, test_learning_rate, num_blocks_reupload, init_scale, use_augmentation):
        global scipy_rng, key, key_r, _global_subseed, _num_point

        _num_point = int(num_qubit / 2)

        _rng_pack = make_rng_pack(_base_seed, _num_point, dataset_tag)
        scipy_rng       = _rng_pack['scipy_rs']   
        key             = _rng_pack['base_key']   
        _global_subseed = _rng_pack['subseed']   
        key_r = key                               

        print(dataset)
        a = train(gate_type, 35, Theta, epochs, key_r, init_scale,
            num_blocks_reupload, num_qubit, num_reupload, use_augmentation,
            learning_rate=test_learning_rate)
        
        return a


    a = result(gate_type, test_learning_rate, num_blocks_reupload, init_scale, use_augmentation)
    results.append(a)



    dataset_tag = 'shapenet' ; dataset = np.load(f'shapenet_5classes_{int(num_qubit/2)}_{num_reupload}_fps_train700_val100_test200_new.npz')
    print(get_Theta(dataset))
    print(_base_seed)

    train_dataset_x = ensure_reupload_dim(dataset['train_dataset_x'])  
    train_dataset_y = dataset['train_dataset_y']                       
    val_dataset_x   = ensure_reupload_dim(dataset['val_dataset_x'])   
    val_dataset_y   = dataset['val_dataset_y']                        
    test_dataset_x  = ensure_reupload_dim(dataset['test_dataset_x'])   
    test_dataset_y  = dataset['test_dataset_y']

    a = result(gate_type, test_learning_rate, num_blocks_reupload, init_scale, use_augmentation)
    results.append(a)

print(results)