import json
from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pennylane as qml
from scipy.linalg import eigh as scipy_eigh

ROT_ONLY_DIR = Path(__file__).resolve().parent / "RotOnlyMatrix"

@lru_cache(maxsize=None)
def _metadata(num_qubit):
    path = ROT_ONLY_DIR / f"perm_matrix_rot_only_block_{num_qubit}_metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Run create_perm_matrix_rot_only.py --num_qubit {num_qubit} first."
        )
    with path.open(encoding="utf-8") as file:
        return json.load(file)

@lru_cache(maxsize=None)
def _decompositions(num_qubit, cycle_length):
    entries = _metadata(num_qubit)["generators"].get(str(cycle_length))
    if entries is None:
        raise ValueError(f"No rot-only generators for qubits={num_qubit}, k={cycle_length}.")

    cache_dir = ROT_ONLY_DIR / "EigenCache_complex_float64"
    cache_dir.mkdir(parents=True, exist_ok=True)

    decompositions = []
    for position, entry in enumerate(entries):
        matrix_path = ROT_ONLY_DIR / entry["filename"]
        if not matrix_path.is_file():
            raise FileNotFoundError(f"Missing generator matrix: {matrix_path}")

        cache_path = cache_dir / f"{matrix_path.stem}_eigh.npz"
        source_mtime_ns = matrix_path.stat().st_mtime_ns
        use_cache = False

        if cache_path.is_file():
            with np.load(cache_path) as cached:
                if int(cached["source_mtime_ns"]) == source_mtime_ns:
                    eigenvalues = cached["eigenvalues"]
                    eigenvectors = cached["eigenvectors"]
                    use_cache = True

        if not use_cache:
            matrix = np.load(matrix_path)
            expected_shape = (2**num_qubit, 2**num_qubit)
            if matrix.shape != expected_shape:
                raise ValueError(
                    f"{matrix_path.name} has shape {matrix.shape}, expected {expected_shape}."
                )
            if not np.allclose(matrix, matrix.conj().T):
                raise ValueError(f"{matrix_path.name} is not Hermitian.")

            matrix = np.asarray(matrix, dtype=np.complex128)
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

        decompositions.append((eigenvalues, eigenvectors))
        status = "cache" if use_cache else "computed"
        print(
            f"  eigendecomposition k={cycle_length} "
            f"{position + 1:3d}/{len(entries)} ({status})"
        )

    return tuple(decompositions)

def rot_only_generator_count(num_qubit, cycle_length):
    return len(_decompositions(num_qubit, cycle_length))

def num_rot_only_params(num_qubit, depth):
    num_pairs = num_qubit // 2
    return depth * sum(
        rot_only_generator_count(num_qubit, cycle_length)
        for cycle_length in range(2, num_pairs + 1)
    )

def _apply_exp_i_hermitian(theta, eigenvalues, eigenvectors, wires):
    unitary = jnp.asarray(eigenvectors)
    values = jnp.asarray(eigenvalues)
    qml.QubitUnitary(unitary.conj().T, wires=wires)
    qml.DiagonalQubitUnitary(jnp.exp(1j * theta * values), wires=wires)
    qml.QubitUnitary(unitary, wires=wires)

def Spin_twirling_rot_only(params, cycle_length, wires):
    num_qubit = len(wires)
    decompositions = _decompositions(num_qubit, cycle_length)
    if len(params) != len(decompositions):
        raise ValueError(
            f"Expected {len(decompositions)} parameters for k={cycle_length}; got {len(params)}."
        )
    for index, (eigenvalues, eigenvectors) in enumerate(decompositions):
        _apply_exp_i_hermitian(params[index], eigenvalues, eigenvectors, wires)

def create_singlet(i, j):
    qml.Hadamard(wires=i)
    qml.PauliZ(wires=i)
    qml.CNOT(wires=[i, j])
    qml.PauliX(wires=j)
