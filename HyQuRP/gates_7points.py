from pathlib import Path
import pennylane as qml
import jax.numpy as jnp
import numpy as np

EIGEN_CACHE = {}

def precompute_UDU_decomposition(num_qubit, num_perm):
    base = Path("/workspace")
    n = num_qubit // 2

    eigenvals_1 = np.load(base / f"plus_matrix_n{n}_order{num_perm}_evals.npy", mmap_mode="r")
    U_1 = np.load(base / f"plus_matrix_n{n}_order{num_perm}_evecs.npy", mmap_mode="r")

    eigenvals_2 = np.load(base / f"minus_matrix_n{n}_order{num_perm}_evals.npy", mmap_mode="r")
    U_2 = np.load(base / f"minus_matrix_n{n}_order{num_perm}_evecs.npy", mmap_mode="r")

    return eigenvals_1, U_1, eigenvals_2, U_2


class Spin_twirling(qml.operation.Operation):
    num_params = 3
    num_wires = qml.operation.AnyWires
    grad_method = None

    @staticmethod
    def compute_decomposition(theta0, theta1, num_perm, wires):
        num_qubit = len(wires)
        cache_key = (num_qubit, num_perm)

        if cache_key not in EIGEN_CACHE:
            print(f"Computing UDU† decomposition for qubits={num_qubit}, perm={num_perm}")
            EIGEN_CACHE[cache_key] = precompute_UDU_decomposition(num_qubit, num_perm)
            print("UDU† decomposition cached!")

        eigenvals_1, U_1, eigenvals_2, U_2 = EIGEN_CACHE[cache_key]

        eigenvals_1_jax = jnp.array(eigenvals_1)
        U_1_jax = jnp.array(U_1)
        eigenvals_2_jax = jnp.array(eigenvals_2)
        U_2_jax = jnp.array(U_2)

        return [
            qml.QubitUnitary(U_1_jax.conj().T, wires=wires),
            qml.DiagonalQubitUnitary(jnp.exp(1j * theta0 * eigenvals_1_jax), wires=wires),
            qml.QubitUnitary(U_1_jax, wires=wires),
            qml.QubitUnitary(U_2_jax.conj().T, wires=wires),
            qml.DiagonalQubitUnitary(jnp.exp(1j * theta1 * eigenvals_2_jax), wires=wires),
            qml.QubitUnitary(U_2_jax, wires=wires),
        ]


def create_singlet(i, j):
    qml.Hadamard(wires=i)
    qml.PauliZ(wires=i)
    qml.CNOT(wires=[i, j])
    qml.PauliX(wires=j)
