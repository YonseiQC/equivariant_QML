import argparse
import atexit
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

_enable_x64 = True

import jax

jax.config.update("jax_enable_x64", _enable_x64)

import jax.numpy as jnp
import numpy as np
import optax
import pennylane as qml
from flax import linen as nn
from scipy.linalg import eigh as scipy_eigh
from sklearn.metrics import confusion_matrix

from gates_fast import create_singlet

tree_leaves = jax.tree_util.tree_leaves
tree_map = jax.tree_util.tree_map

def make_subseed(base_seed: int, *keys) -> int:
    digest = hashlib.sha256(str((base_seed,) + tuple(keys)).encode()).hexdigest()
    return int(digest[:8], 16)

def make_rng_pack(base_seed: int, num_point: int, dataset_tag: str):
    subseed = make_subseed(base_seed, num_point, dataset_tag)
    return {"subseed": subseed, "base_key": jax.random.PRNGKey(subseed)}

_METRICS_FH = None

class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

def _metrics_write(obj):
    if _METRICS_FH is not None:
        _METRICS_FH.write(json.dumps(obj, ensure_ascii=False) + "\n")
        _METRICS_FH.flush()

def epoch_shuffle_numpy(x, y):
    indices = np.random.permutation(x.shape[0])
    return x[indices], y[indices]

class MyNNLight(nn.Module):
    num_pairs: int
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = jnp.expand_dims(x, axis=-1).reshape(x.shape[0], self.num_pairs, 2)
        x = nn.tanh(nn.Dense(features=4)(x))
        x = nn.tanh(nn.Dense(features=4)(x))

        pooled = [
            jnp.mean(x, axis=1),
            jnp.max(x, axis=1),
            jnp.min(x, axis=1),
            jnp.sum(x, axis=1),
            jnp.std(x, axis=1),
            jnp.var(x, axis=1),
        ]
        x = jnp.concatenate(pooled, axis=-1)
        x = nn.tanh(nn.Dense(features=24)(x))
        x = nn.tanh(nn.Dense(features=24)(x))
        return nn.Dense(features=self.num_classes)(x)

class MyNNMid(nn.Module):
    num_pairs: int
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = jnp.expand_dims(x, -1).reshape(x.shape[0], self.num_pairs, 2)
        x = nn.tanh(nn.Dense(8)(x))
        x = nn.tanh(nn.Dense(16)(x))
        x = nn.tanh(nn.Dense(32)(x))

        pooled = [
            jnp.mean(x, axis=1),
            jnp.max(x, axis=1),
            jnp.min(x, axis=1),
            jnp.sum(x, axis=1),
            jnp.std(x, axis=1),
            jnp.var(x, axis=1),
        ]
        x = jnp.concatenate(pooled, axis=-1)
        x = nn.tanh(nn.Dense(32)(x))
        x = nn.tanh(nn.Dense(16)(x))
        x = nn.tanh(nn.Dense(8)(x))
        return nn.Dense(self.num_classes)(x)

def nn_apply(variant, num_pairs, num_classes, params_c, x):
    module = (
        MyNNLight(num_pairs=num_pairs, num_classes=num_classes)
        if variant == "light"
        else MyNNMid(num_pairs=num_pairs, num_classes=num_classes)
    )
    return module.apply(params_c, x)

def calculate_final_metrics(y_true, y_pred, num_classes):
    y_true_np = np.asarray(y_true).reshape(-1)
    y_pred_np = np.asarray(y_pred).reshape(-1)
    matrix = confusion_matrix(y_true_np, y_pred_np, labels=range(num_classes))
    class_accuracies = []
    for class_index in range(num_classes):
        denominator = np.sum(matrix[class_index, :])
        class_accuracies.append(
            float(matrix[class_index, class_index] / denominator) if denominator else 0.0
        )
    overall = float(np.trace(matrix) / np.sum(matrix))
    return matrix, class_accuracies, overall

def prepare_init_state(num_qubit):
    for wire in range(0, num_qubit, 2):
        create_singlet(wire, wire + 1)

def encode(points, num_qubit):
    norms = jnp.sqrt(jnp.sum(jnp.square(points), axis=-1))
    nx = points[:, :, 0] / norms
    ny = points[:, :, 1] / norms
    nz = points[:, :, 2] / norms

    norms_t = norms.T
    nx_t, ny_t, nz_t = nx.T, ny.T, nz.T
    for point_index in range(num_qubit // 2):
        cos_norm = jnp.cos(norms_t[point_index])
        sin_norm = jnp.sin(norms_t[point_index])
        nx_i, ny_i, nz_i = nx_t[point_index], ny_t[point_index], nz_t[point_index]
        unitary = jnp.array(
            [
                [
                    cos_norm + 1j * sin_norm * nz_i,
                    1j * sin_norm * nx_i + sin_norm * ny_i,
                ],
                [
                    1j * sin_norm * nx_i - sin_norm * ny_i,
                    cos_norm - 1j * sin_norm * nz_i,
                ],
            ]
        ).transpose(2, 0, 1)
        qml.QubitUnitary(unitary, wires=2 * point_index)

def create_hamiltonians(num_points):
    terms = []
    for i in range(num_points - 1):
        for j in range(i + 1, num_points):
            terms.append(
                (qml.PauliX(2 * i) + qml.PauliX(2 * i + 1))
                @ (qml.PauliX(2 * j) + qml.PauliX(2 * j + 1))
                + (qml.PauliY(2 * i) + qml.PauliY(2 * i + 1))
                @ (qml.PauliY(2 * j) + qml.PauliY(2 * j + 1))
                + (qml.PauliZ(2 * i) + qml.PauliZ(2 * i + 1))
                @ (qml.PauliZ(2 * j) + qml.PauliZ(2 * j + 1))
            )
            terms.append(
                (qml.PauliX(2 * i) - qml.PauliX(2 * i + 1))
                @ (qml.PauliX(2 * j) - qml.PauliX(2 * j + 1))
                + (qml.PauliY(2 * i) - qml.PauliY(2 * i + 1))
                @ (qml.PauliY(2 * j) - qml.PauliY(2 * j + 1))
                + (qml.PauliZ(2 * i) - qml.PauliZ(2 * i + 1))
                @ (qml.PauliZ(2 * j) - qml.PauliZ(2 * j + 1))
            )
    return terms

class ReflectionGeneratorBank:
    def __init__(
        self,
        generator_dir,
        num_qubit,
        num_generators=10,
        rebuild_cache=False,
        eigen_dtype="float64",
    ):
        self.generator_dir = Path(generator_dir)
        manifest_path = self.generator_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Generator manifest not found: {manifest_path}\n"
                f"Run create_complete_sym_generators.py --num-qubit {num_qubit} first."
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(self.manifest["num_qubit"]) != num_qubit:
            raise ValueError(
                f"Generator bank uses {self.manifest['num_qubit']} qubits, expected {num_qubit}."
            )
        construction = self.manifest.get("construction", [])
        required_formulas = {
            "H_plus=(A+A^dagger)/2",
            "H_minus=(A-A^dagger)/(2i)",
            "A=T_Spair[Pi(pi)]",
        }
        if not isinstance(construction, list) or not required_formulas.issubset(
            construction
        ):
            raise ValueError(
                "The manifest is not a complete Hermitian symmetric-generator bank."
            )
        if int(self.manifest.get("full_hermitian_dimension", -1)) != 206:
            raise ValueError(
                "Expected the 10-qubit complete bank to have Hermitian dimension 206."
            )
        if int(self.manifest["num_generators"]) != len(self.manifest["generators"]):
            raise ValueError("Generator count in manifest is inconsistent.")
        if int(self.manifest.get("w_breaking_generators", 0)) <= 0:
            raise ValueError("Generator bank contains no W-breaking direction.")

        total_generators = int(self.manifest["num_generators"])
        if num_generators <= 0:
            raise ValueError("num_generators must be a positive integer.")
        if num_generators > total_generators:
            raise ValueError(
                f"Requested {num_generators} generators, but the bank contains "
                f"only {total_generators}."
            )
        selected_entries = self.manifest["generators"][:num_generators]
        self.num_generators = len(selected_entries)
        selected_w_breaking = sum(
            float(entry.get("w_commutator_relative_norm", 0.0)) > 1e-7
            for entry in selected_entries
        )
        if selected_w_breaking == 0:
            raise ValueError("The selected generators contain no W-breaking direction.")
        self.eigenvalues = []
        self.eigenvectors = []
        self.component_counts = {
            component: sum(
                entry.get("component") == component
                for entry in selected_entries
            )
            for component in ("adjoint_even", "adjoint_odd")
        }
        if self.component_counts["adjoint_odd"] == 0:
            raise ValueError(
                "The selected generators contain no (A-A^dagger)/(2i) direction."
            )

        cache_dir = self.generator_dir / f"EigenCache_complex_{eigen_dtype}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        numpy_dtype = np.complex64 if eigen_dtype == "float32" else np.complex128

        matrix_dimension = 2**num_qubit
        estimated_eigenvector_gib = (
            self.num_generators
            * matrix_dimension
            * matrix_dimension
            * np.dtype(numpy_dtype).itemsize
            / 1024**3
        )

        print(
            f"Loading {self.num_generators}/{total_generators} generators "
            f"from {self.generator_dir}; "
            f"adjoint-even={self.component_counts['adjoint_even']}, "
            f"adjoint-odd={self.component_counts['adjoint_odd']}, "
            f"W-breaking={selected_w_breaking}"
        )
        print(
            f"Estimated stored eigenvector memory: {estimated_eigenvector_gib:.2f} GiB "
            f"at complex{'64' if eigen_dtype == 'float32' else '128'} precision"
        )
        for position, entry in enumerate(selected_entries):
            matrix_path = self.generator_dir / entry["filename"]
            if not matrix_path.is_file():
                raise FileNotFoundError(f"Missing generator matrix: {matrix_path}")
            cache_path = cache_dir / f"{matrix_path.stem}_eigh.npz"
            source_mtime_ns = matrix_path.stat().st_mtime_ns

            use_cache = False
            if cache_path.is_file() and not rebuild_cache:
                with np.load(cache_path) as cached:
                    cached_mtime = int(cached["source_mtime_ns"])
                    if cached_mtime == source_mtime_ns:
                        eigenvalues = cached["eigenvalues"]
                        eigenvectors = cached["eigenvectors"]
                        use_cache = True

            if not use_cache:
                stored_matrix = np.load(matrix_path)
                expected_shape = (matrix_dimension, matrix_dimension)
                if stored_matrix.shape != expected_shape:
                    raise ValueError(
                        f"{matrix_path.name} has shape {stored_matrix.shape}, "
                        f"expected {expected_shape}."
                    )

                                                                             
                                                                            
                                                                             
                                                                        
                stored_is_single_precision = stored_matrix.dtype in (
                    np.dtype(np.float32),
                    np.dtype(np.complex64),
                )
                hermitian_tolerance = (
                    1e-5 if stored_is_single_precision else 1e-10
                )
                hermitian_error = float(
                    np.max(np.abs(stored_matrix - stored_matrix.conj().T))
                )
                if hermitian_error > hermitian_tolerance:
                    raise ValueError(
                        f"{matrix_path.name} is not Hermitian; max error={hermitian_error}."
                    )

                matrix = np.asarray(stored_matrix, dtype=numpy_dtype)
                del stored_matrix
                                                                          
                matrix = 0.5 * (matrix + matrix.conj().T)
                eigenvalues, eigenvectors = scipy_eigh(
                    matrix, overwrite_a=True, check_finite=False, driver="evd"
                )
                np.savez(
                    cache_path,
                    source_mtime_ns=np.asarray(source_mtime_ns, dtype=np.int64),
                    eigenvalues=eigenvalues,
                    eigenvectors=eigenvectors,
                )
                del matrix

            self.eigenvalues.append(jnp.asarray(eigenvalues))
            self.eigenvectors.append(jnp.asarray(eigenvectors))
            print(
                f"  eigendecomposition {position + 1:3d}/{self.num_generators} "
                f"({'cache' if use_cache else 'computed'})"
            )

    def apply(self, theta, generator_index, wires):
        eigenvalues = self.eigenvalues[generator_index]
        eigenvectors = self.eigenvectors[generator_index]
        qml.QubitUnitary(eigenvectors.conj().T, wires=wires)
        qml.DiagonalQubitUnitary(jnp.exp(1j * theta * eigenvalues), wires=wires)
        qml.QubitUnitary(eigenvectors, wires=wires)

def create_reflection_circuit(
    num_qubit, depth, theta_scale, hamiltonians, generator_bank
):
    def circuit(params, data_pt):
        prepare_init_state(num_qubit)
        points = data_pt[:, 0, :, :]
        encode(points / theta_scale, num_qubit)
        for layer in range(depth):
            for generator_index in range(generator_bank.num_generators):
                generator_bank.apply(
                    params["q"][layer, generator_index],
                    generator_index,
                    wires=range(num_qubit),
                )
        return [qml.expval(hamiltonian) for hamiltonian in hamiltonians]

    return circuit

def ensure_reupload_dim(x):
    if x.ndim == 3:
        return x.reshape(x.shape[0], 1, -1, 3)
    if x.ndim == 4:
        return x
    raise ValueError(f"Unexpected point-cloud shape: {x.shape}")

def validate_reflection_dataset(dataset):
    expected = {"train": (70, 35), "val": (10, 5), "test": (20, 10)}
    for split, (total, per_class) in expected.items():
        x_key = f"{split}_dataset_x"
        y_key = f"{split}_dataset_y"
        if x_key not in dataset or y_key not in dataset:
            raise KeyError(f"Dataset must contain {x_key!r} and {y_key!r}.")
        x, y = dataset[x_key], dataset[y_key]
        if x.shape != (total, 5, 3):
            raise ValueError(f"{x_key} has shape {x.shape}; expected {(total, 5, 3)}.")
        if y.shape != (total,):
            raise ValueError(f"{y_key} has shape {y.shape}; expected {(total,)}.")
        if not np.array_equal(np.bincount(y, minlength=2), np.array([per_class, per_class])):
            raise ValueError(f"{y_key} must contain {per_class} samples from each class.")
        if np.max(np.abs(x.mean(axis=1))) > 1e-10:
            raise ValueError(f"{x_key} is not centered at the origin.")

def print_reflection_probe(label, params, probe_original, probe_reflected, quantum_features, forward):
    original_features = np.asarray(jax.device_get(quantum_features(params, probe_original)))[0]
    reflected_features = np.asarray(jax.device_get(quantum_features(params, probe_reflected)))[0]
    original_logits = np.asarray(jax.device_get(forward(params, probe_original)))[0]
    reflected_logits = np.asarray(jax.device_get(forward(params, probe_reflected)))[0]

    print(f"\n=== Reflection probe ({label}) ===")
    print(f"max |quantum feature difference|: {np.max(np.abs(original_features - reflected_features)):.12e}")
    print(f"Original logits  : {np.array2string(original_logits, precision=12)}")
    print(f"Reflected logits : {np.array2string(reflected_logits, precision=12)}")
    print(f"Logit difference : {np.array2string(original_logits - reflected_logits, precision=12)}")
    print("==========================================\n")

def train(
    *,
    data,
    rng_pack,
    dev,
    variant,
    num_pairs,
    num_classes,
    minibatch_size,
    theta_scale,
    epochs,
    init_scale,
    depth,
    num_qubit,
    l2,
    learning_rate,
    generator_bank,
):
    train_x, train_y = data["train_x"], data["train_y"]
    val_x, val_y = data["val_x"], data["val_y"]
    test_x, test_y = data["test_x"], data["test_y"]
    global_subseed = rng_pack["subseed"]

    if len(train_x) != len(train_y) or len(train_x) % minibatch_size:
        raise ValueError("Training split size must match labels and be divisible by minibatch size.")

                                                                          
                                                   
    parameter_scale = init_scale * math.pi / math.sqrt(generator_bank.num_generators)
    qkey = jax.random.PRNGKey(make_subseed(global_subseed, "init_q_reflection"))
    params_q = parameter_scale * jax.random.uniform(
        qkey,
        (depth, generator_bank.num_generators),
        minval=-1.0,
        maxval=1.0,
    )

    dummy_input = jnp.ones((1, 2 * num_pairs))
    module = (
        MyNNLight(num_pairs=num_pairs, num_classes=num_classes)
        if variant == "light"
        else MyNNMid(num_pairs=num_pairs, num_classes=num_classes)
    )
    params = {"q": params_q, "c": module.init(rng_pack["base_key"], dummy_input)}

    hamiltonians = create_hamiltonians(num_qubit // 2)
    qnode = qml.QNode(
        create_reflection_circuit(
            num_qubit, depth, theta_scale, hamiltonians, generator_bank
        ),
        device=dev,
        interface="jax",
    )
    qnode = jax.jit(qnode)

    def quantum_features(params_, x_batch):
        return jnp.asarray(qnode(params_, x_batch)).T

    def forward(params_, x_batch):
        features = quantum_features(params_, x_batch)
        return nn_apply(variant, num_pairs, num_classes, params_["c"], features)

    def loss_fn(params_, x_batch, y_batch):
        logits = forward(params_, x_batch)
        loss = jnp.mean(
            optax.losses.softmax_cross_entropy_with_integer_labels(logits, y_batch)
        )
        if l2:
            loss += l2 * sum(jnp.sum(jnp.square(leaf)) for leaf in tree_leaves(params_))
        return loss

    def accuracy_fn(params_, x, y):
        predictions = jnp.argmax(forward(params_, x), axis=-1)
        return jnp.mean((predictions == y.reshape(-1)).astype(jnp.float32))

    probe_index = int(np.flatnonzero(np.asarray(train_y) == 0)[0])
    probe_original = jnp.asarray(train_x[probe_index : probe_index + 1])
    probe_reflected = probe_original.at[..., 0].set(-probe_original[..., 0])
    print_reflection_probe(
        "before training", params, probe_original, probe_reflected, quantum_features, forward
    )

    solver = optax.adam(learning_rate=learning_rate)
    opt_state = solver.init(params)
    best_val_acc = -jnp.inf
    best_epoch = -1
    params_best = tree_map(lambda value: value.copy(), params)

    batch_size = len(train_x)
    num_batches = batch_size // minibatch_size
    for epoch in range(epochs):
        np.random.seed(make_subseed(global_subseed, "shuffle", epoch))
        shuffled_x, shuffled_y = epoch_shuffle_numpy(train_x, train_y)
        x_batches = shuffled_x.reshape(num_batches, minibatch_size, 1, -1, 3)
        y_batches = shuffled_y.reshape(num_batches, minibatch_size)
        epoch_loss = 0.0

        for batch_index in range(num_batches):
            loss, gradient = jax.value_and_grad(loss_fn)(
                params, x_batches[batch_index], y_batches[batch_index]
            )
            updates, opt_state = solver.update(gradient, opt_state, params)
            params = optax.apply_updates(params, updates)
            epoch_loss += loss / num_batches

        val_loss = loss_fn(params, val_x, val_y)
        val_acc = accuracy_fn(params, val_x, val_y)
        print(
            f"epoch {epoch}/{epochs - 1} | train loss: {float(epoch_loss):.4f} | "
            f"val loss: {float(val_loss):.4f} | val accuracy: {float(val_acc):.4f}"
        )
        _metrics_write(
            {
                "epoch": epoch,
                "train_loss": float(epoch_loss),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }
        )
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            params_best = tree_map(lambda value: value.copy(), params)

    print_reflection_probe(
        "best validation checkpoint",
        params_best,
        probe_original,
        probe_reflected,
        quantum_features,
        forward,
    )

    logits = forward(params_best, test_x)
    predictions = jnp.argmax(logits, axis=-1)
    matrix, class_accuracies, overall = calculate_final_metrics(
        test_y, predictions, num_classes
    )
    print("\n=== Results ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {float(best_val_acc):.4f}")
    print(f"Test accuracy: {overall:.4f}")
    print("Confusion matrix:")
    print(matrix)
    for class_index, accuracy in enumerate(class_accuracies):
        print(f"Class {class_index}: {accuracy:.4f}")

    _metrics_write(
        {
            "final": True,
            "best_epoch": best_epoch,
            "best_val_acc": float(best_val_acc),
            "test_acc": overall,
            "class_acc": class_accuracies,
        }
    )
    return overall

def main():
    global _METRICS_FH

    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("--num_qubit", type=int, required=True)
    parser.add_argument("--variant", type=str, choices=["light", "mid"], required=True)
    parser.add_argument("--num_generators", type=int, choices=[10, 206], default=206)
    args = parser.parse_args()

    base_seed = args.seed
    num_qubit = args.num_qubit
    variant = args.variant
    num_generators = args.num_generators

    if num_qubit != 10:
        raise ValueError("The reflection dataset contains five points and requires --num_qubit 10.")

    num_points = num_qubit // 2
    num_measurement_pairs = num_points * (num_points - 1) // 2

    theta_scale = 1.7
    epochs = 100
    depth = 1
    init_scale = 0.02
    learning_rate = 0.001
    l2 = 0.0
    eigen_dtype = "float64"

    here = Path(__file__).resolve().parent
    repository_root = here.parent
    generator_dir = here / "ReflectionMatrix"
    data_path = repository_root / "data" / "Reflection" / "reflection_5points_train70_val10_test20.npz"

    run_id = (
        f"{Path(__file__).stem}_{base_seed}_{num_points}_{variant}"
        f"_g{num_generators}"
    )
    stdout_path = repository_root / f"{run_id}.stdout.log"
    config_path = repository_root / f"{run_id}.config.json"
    metrics_path = repository_root / f"{run_id}.metrics.jsonl"

    original_stdout, original_stderr = sys.stdout, sys.stderr
    stdout_file = open(stdout_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(original_stdout, stdout_file)
    sys.stderr = _Tee(original_stderr, stdout_file)
    _METRICS_FH = open(metrics_path, "w", encoding="utf-8", buffering=1)

    def cleanup():
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        for file_handle in (stdout_file, _METRICS_FH):
            try:
                file_handle.close()
            except Exception:
                pass

    atexit.register(cleanup)

    generator_bank = ReflectionGeneratorBank(
        generator_dir,
        num_qubit,
        num_generators=num_generators,
        rebuild_cache=False,
        eigen_dtype=eigen_dtype,
    )

    config = {
        "model": Path(__file__).stem,
        "seed": int(base_seed),
        "dataset": "reflection",
        "variant": str(variant),
        "num_qubit": int(num_qubit),
        "num_points": int(num_points),
        "epochs": int(epochs),
        "lr": float(learning_rate),
        "Theta": float(theta_scale),
        "depth": int(depth),
        "num_quantum_generators": int(generator_bank.num_generators),
        "generator_component_counts": generator_bank.component_counts,
        "generator_construction": "complete_H_plus_and_H_minus_basis",
        "generator_dir": str(generator_dir),
        "init_scale": float(init_scale),
        "eigen_dtype": str(eigen_dtype),
        "jax_enable_x64": bool(_enable_x64),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Place reflection_5points_train70_val10_test20.npz under data/Reflection/."
        )

    dataset = np.load(data_path)
    validate_reflection_dataset(dataset)
    data = {
        "train_x": ensure_reupload_dim(dataset["train_dataset_x"]),
        "train_y": dataset["train_dataset_y"],
        "val_x": ensure_reupload_dim(dataset["val_dataset_x"]),
        "val_y": dataset["val_dataset_y"],
        "test_x": ensure_reupload_dim(dataset["test_dataset_x"]),
        "test_y": dataset["test_dataset_y"],
    }

    print(
        f"seed={base_seed}, dataset=reflection, variant={variant}, "
        f"num_points={num_points}, generators={generator_bank.num_generators}, "
        f"epochs={epochs}, lr={learning_rate}"
    )

    train(
        data=data,
        rng_pack=make_rng_pack(base_seed, num_points, "reflection"),
        dev=qml.device("default.qubit", wires=num_qubit),
        variant=variant,
        num_pairs=num_measurement_pairs,
        num_classes=2,
        minibatch_size=5,
        theta_scale=theta_scale,
        epochs=epochs,
        init_scale=init_scale,
        depth=depth,
        num_qubit=num_qubit,
        l2=l2,
        learning_rate=learning_rate,
        generator_bank=generator_bank,
    )

if __name__ == "__main__":
    main()
