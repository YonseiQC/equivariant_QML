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


def decompose_diagonal_unitary_exact(diagonal_elements, wires):
    num_qubit = len(wires)
    temp_dev = qml.device('reference.qubit', wires=num_qubit)
    
    @qml.qnode(temp_dev)
    def temp_circuit():
        qml.DiagonalQubitUnitary(diagonal_elements, wires=wires)
        return qml.expval(qml.PauliZ(0))
    
    gate_set = {"RZ", "IsingZZ", "MultiRZ"}
    max_expansion = 1000

    decomposed_qnode = qml.transforms.decompose(temp_circuit, gate_set=gate_set, max_expansion=max_expansion)

    specs = qml.specs(decomposed_qnode)()
    resources = specs['resources']
    gate_counts = resources.gate_types
    print("\n--- Gate Count ---")
    total_gates = 0
    for gate, count in gate_counts.items():
        print(f"{gate}: {count} 개")
        total_gates += count
    print("------------------")
    print(f"Total Gates: {total_gates} 개")

    
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

        operations = []

        operations.append(qml.QubitUnitary(U_1.conj().T, wires=wires))
        
        diagonal_1 = jnp.exp(1j * theta0 * eigenvals_1_jax)
        decomposed_ops_1 = decompose_diagonal_unitary_exact(diagonal_1, wires)
        operations.extend(decomposed_ops_1)

        operations.append(qml.QubitUnitary(U_1, wires=wires))
        
        operations.append(qml.QubitUnitary(U_2.conj().T, wires=wires))
        
        diagonal_2 = jnp.exp(1j * theta1 * eigenvals_2_jax)
        decomposed_ops_2 = decompose_diagonal_unitary_exact(diagonal_2, wires)
        operations.extend(decomposed_ops_2)
        
        operations.append(qml.QubitUnitary(U_2, wires=wires))
        
        total_gate_counts = {}
        for op in operations:
            gate_name = op.name
            total_gate_counts[gate_name] = total_gate_counts.get(gate_name, 0) + 1
        
        return operations

# class Spin_twirling(qml.operation.Operation):
#     num_params = 3
#     num_wires = qml.operation.AnyWires
#     grad_method = None

#     @staticmethod
#     def compute_decomposition(theta0, theta1, num_perm, wires):
#         num_qubit = len(wires)
#         cache_key = (num_qubit, num_perm)

#         if cache_key not in EIGEN_CACHE:
#             print(f"Computing UDU† decomposition for qubits={num_qubit}, perm={num_perm}")
#             EIGEN_CACHE[cache_key] = precompute_UDU_decomposition(num_qubit, num_perm)
#             print("UDU† decomposition cached!")
        
#         eigenvals_1, U_1, eigenvals_2, U_2 = EIGEN_CACHE[cache_key]

#         eigenvals_1_jax = jnp.array(eigenvals_1)
#         U_1_jax = jnp.array(U_1)
#         eigenvals_2_jax = jnp.array(eigenvals_2)
#         U_2_jax = jnp.array(U_2)

#         return [
#             qml.DiagonalQubitUnitary(jnp.exp(1j * theta0 * eigenvals_1_jax), wires=wires),  
#             qml.DiagonalQubitUnitary(jnp.exp(1j * theta1 * eigenvals_2_jax), wires=wires),  
#         ]

def create_singlet(i, j):
    qml.Hadamard(wires=i)
    qml.PauliZ(wires=i)
    qml.CNOT(wires=[i, j])
    qml.PauliX(wires=j)
