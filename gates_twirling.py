from jax.numpy.linalg import eigh
from scipy.sparse import kron, eye
import pennylane as qml
import itertools
import jax.numpy as jnp


class Spin_2_twirling(qml.operation.Operation):
    num_params = 1
    num_wires = qml.operation.AnyWires
    grad_method = "F"

    @staticmethod
    def i_and_j_and_k(num_qubit, i, j, k, pauli):
        ops = []
        for idx in range(0, num_qubit):
            if idx == i:
                ops.append(pauli[0])
            elif idx == j:
                ops.append(pauli[1])
            elif idx == k:
                ops.append(pauli[2])
            else:
                ops.append(jnp.array([[1,0],[0,1]]))
        result = ops[0]
        for op in ops[1:]:
            result = jnp.kron(result, op)
        return result
    
    @staticmethod
    def compute_decomposition(theta, wires):
        num_qubit = len(wires)
        dim = 2 ** num_qubit
        X = jnp.array([[0, 1], [1, 0]], dtype=complex)
        Y = jnp.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = jnp.array([[1, 0], [0, -1]], dtype=complex)
        operator = [X, Y, Z]
        Gate = jnp.zeros((dim, dim), dtype=complex)
        for perm in itertools.permutations(operator):
            for i in range(0, num_qubit - 2):
                for j in range(i + 1, num_qubit - 1):
                    for k in range(j + 1, num_qubit):
                        Gate += Spin_2_twirling.i_and_j_and_k(num_qubit, i, j, k, perm)
        eigenvals, U = eigh(Gate)

        return [qml.QubitUnitary(U.conj().T, wires=range(num_qubit)),
        qml.DiagonalQubitUnitary(jnp.exp(1j * theta * eigenvals), wires=range(num_qubit)),
        qml.QubitUnitary(U, wires=range(num_qubit))]

class Spin_3_twirling(qml.operation.Operation):
    num_params = 1
    num_wires = qml.operation.AnyWires
    grad_method = "F"
    
    @staticmethod
    def i_and_j(num_qubit, i, j, pauli):
        ops = []
        for idx in range(0, num_qubit):
            if idx == i or idx == j:
                ops.append(pauli)
            else:
                ops.append(jnp.array([[1,0],[0,1]]))
        result = ops[0]
        for op in ops[1:]:
            result = jnp.kron(result, op)
        return result

    @staticmethod
    def compute_decomposition(theta, wires):
        num_qubit = len(wires)
        dim = 2 ** num_qubit
        X = jnp.array([[0, 1], [1, 0]], dtype=complex)
        Y = jnp.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = jnp.array([[1, 0], [0, -1]], dtype=complex)
        operator = [X, Y, Z]
        Gate = jnp.zeros((dim, dim), dtype=complex)
        for i in range(0, num_qubit - 1):
                for j in range(i + 1, num_qubit):
                    Gate += Spin_3_twirling.i_and_j(num_qubit, i, j, operator[0]) + Spin_3_twirling.i_and_j(num_qubit, i, j, operator[1]) + Spin_3_twirling.i_and_j(num_qubit, i, j, operator[2])
        eigenvals, U = eigh(Gate)

        return [qml.QubitUnitary(U.conj().T, wires=range(num_qubit)),
        qml.DiagonalQubitUnitary(jnp.exp(1j * theta * eigenvals), wires=range(num_qubit)),
        qml.QubitUnitary(U, wires=range(num_qubit))]
