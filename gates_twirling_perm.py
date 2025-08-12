from jax.numpy.linalg import eigh
import pennylane as qml
import jax.numpy as jnp
import numpy as np



class Spin_twirling(qml.operation.Operation):
    num_params = 3
    num_wires = qml.operation.AnyWires
    grad_method = None

    @staticmethod
    def compute_decomposition(theta0, theta1, num_perm, wires):
        num_qubit = len(wires)
        filename = f"perm_matrix_{num_qubit}_{num_perm}_plus.npy"
        matrix_1 = np.load(filename)
        Gate_1 = jnp.array(matrix_1, dtype=complex)
    
        eigenvals_1, U_1 = eigh(Gate_1)
        
        # max_eigenval_1 = jnp.max(jnp.abs(eigenvals_1))
        # normalized_eigenvals_1 = eigenvals_1 / max_eigenval_1
        
        # normalized_eigenvals_1 = jnp.where(
        #     jnp.abs(normalized_eigenvals_1) < 1e-13,
        #     0.0,
        #     normalized_eigenvals_1
        # )

        filename = f"perm_matrix_{num_qubit}_{num_perm}_minus.npy"
        matrix_2 = np.load(filename)
        Gate_2 = jnp.array(matrix_2, dtype=complex)
    
        eigenvals_2, U_2 = eigh(Gate_2)
        
        # max_eigenval_2 = jnp.max(jnp.abs(eigenvals_2))
        # normalized_eigenvals_2 = eigenvals_2 / max_eigenval_2
        
        # normalized_eigenvals_2 = jnp.where(
        #     jnp.abs(normalized_eigenvals_2) < 1e-13,
        #     0.0,
        #     normalized_eigenvals_2
        # )
        
        return [
            qml.QubitUnitary(U_1.conj().T, wires=wires),
            qml.DiagonalQubitUnitary(jnp.exp(1j * theta0 * eigenvals_1), wires=wires),
            qml.QubitUnitary(U_1, wires=wires),
            qml.QubitUnitary(U_2.conj().T, wires=wires),
            qml.DiagonalQubitUnitary(jnp.exp(1j * theta1 * eigenvals_2), wires=wires),
            qml.QubitUnitary(U_2, wires=wires)
        ]