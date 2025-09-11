from scipy.linalg import eigh as scipy_eigh
import pennylane as qml
import jax.numpy as jnp
import numpy as np

EIGEN_CACHE = {}

def precompute_UDU_decomposition(num_qubit, num_perm):

    filename_1 = f"perm_matrix_{num_qubit}_{num_perm}_plus_normalized.npy"
    matrix_1 = np.load(filename_1)
    Gate_1 = np.array(matrix_1, dtype=complex)
    eigenvals_1, U_1 = scipy_eigh(Gate_1)  
    
    filename_2 = f"perm_matrix_{num_qubit}_{num_perm}_minus_normalized.npy"
    matrix_2 = np.load(filename_2)
    Gate_2 = np.array(matrix_2, dtype=complex)
    eigenvals_2, U_2 = scipy_eigh(Gate_2) 
    
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
            qml.QubitUnitary(U_2_jax, wires=wires) 
        ]


def create_singlet(i, j):
    qml.Hadamard(wires=i)
    qml.PauliZ(wires=i)
    qml.CNOT(wires=[i, j])
    qml.PauliX(wires=j)
