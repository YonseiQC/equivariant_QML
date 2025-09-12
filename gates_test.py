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

def decompose_unitary_exact(unitary_matrix, wires):
    num_qubit = len(wires)
    temp_dev = qml.device('reference.qubit', wires=num_qubit)
    
    @qml.qnode(temp_dev)
    def temp_circuit():
        qml.QubitUnitary(unitary_matrix, wires=wires)
        return qml.expval(qml.PauliZ(0))
    
    gate_set = {"RZ", "RY", "RX", "CNOT", "GlobalPhase"}
    max_expansion = 1000
    decomposed_qnode = qml.transforms.decompose(temp_circuit, gate_set=gate_set, max_expansion=max_expansion)
    
    try:
        with qml.queuing.AnnotatedQueue() as q:
            decomposed_qnode()
        
        ops = [op for op in q.queue if hasattr(op, 'name')]
        
        gate_counts = {}
        for op in ops:
            gate_name = op.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        
        print(f"QubitUnitary decomposed into {len(ops)} operations")
        print("Gate breakdown:")
        for gate_name, count in sorted(gate_counts.items()):
            print(f"  {gate_name}: {count}")
        
        return ops
        
    except Exception as e:
        print(f"Error decomposing QubitUnitary: {e}")
        print("Fallback: returning original QubitUnitary")
        return [qml.QubitUnitary(unitary_matrix, wires=wires)]

def decompose_diagonal_unitary_exact(diagonal_elements, wires):
    num_qubit = len(wires)
    temp_dev = qml.device('reference.qubit', wires=num_qubit)
    
    @qml.qnode(temp_dev)
    def temp_circuit():
        qml.DiagonalQubitUnitary(diagonal_elements, wires=wires)
        return qml.expval(qml.PauliZ(0))
    
    gate_set = {"RZ", "RY", "RX", "CNOT", "GlobalPhase"}
    max_expansion = 1000  

    print(f"Decomposing DiagonalQubitUnitary with max_expansion={max_expansion}...")
    
    decomposed_qnode = qml.transforms.decompose(temp_circuit, gate_set=gate_set, max_expansion=max_expansion)
    
    try:
        with qml.queuing.AnnotatedQueue() as q:
            decomposed_qnode()
        
        ops = [op for op in q.queue if hasattr(op, 'name')]
        
        gate_counts = {}
        for op in ops:
            gate_name = op.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        
        print(f"DiagonalQubitUnitary decomposed into {len(ops)} operations")
        print("Gate breakdown:")
        for gate_name, count in sorted(gate_counts.items()):
            print(f"  {gate_name}: {count}")
        
        return ops
        
    except Exception as e:
        print(f"Error decomposing QubitUnitary: {e}")
        print("Fallback: returning original QubitUnitary")
        return [qml.QubitUnitary(diagonal_elements, wires=wires)]
        

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

        print("Decomposing U_1†...")
        decomposed_U1_dag = decompose_unitary_exact(U_1_jax.conj().T, wires)
        operations.extend(decomposed_U1_dag)
        
        print("Decomposing first DiagonalQubitUnitary...")
        diagonal_1 = jnp.exp(1j * theta0 * eigenvals_1_jax)
        decomposed_ops_1 = decompose_diagonal_unitary_exact(diagonal_1, wires)
        operations.extend(decomposed_ops_1)
        
        print("Decomposing U_1...")
        decomposed_U1 = decompose_unitary_exact(U_1_jax, wires)
        operations.extend(decomposed_U1)
        
        print("Decomposing U_2†...")
        decomposed_U2_dag = decompose_unitary_exact(U_2_jax.conj().T, wires)
        operations.extend(decomposed_U2_dag)
        
        print("Decomposing second DiagonalQubitUnitary...")
        diagonal_2 = jnp.exp(1j * theta1 * eigenvals_2_jax)
        decomposed_ops_2 = decompose_diagonal_unitary_exact(diagonal_2, wires)
        operations.extend(decomposed_ops_2)
        
        print("Decomposing U_2...")
        decomposed_U2 = decompose_unitary_exact(U_2_jax, wires)
        operations.extend(decomposed_U2)
        
        total_gate_counts = {}
        for op in operations:
            gate_name = op.name
            total_gate_counts[gate_name] = total_gate_counts.get(gate_name, 0) + 1
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Total operations: {len(operations)}")
        print("Final gate breakdown:")
        for gate_name, count in sorted(total_gate_counts.items()):
            print(f"  {gate_name}: {count}")
        
        problematic_gates = ['QubitUnitary', 'DiagonalQubitUnitary', 'IsingZZ', 'MultiRZ']
        remaining_problematic = [gate for gate in problematic_gates if gate in total_gate_counts]
        
        if remaining_problematic:
            print(f"⚠️  WARNING: Still contains gates that might not be adjoint-compatible: {remaining_problematic}")
        else:
            print("✅ All gates should be adjoint-compatible!")
        
        return operations

def create_singlet(i, j):
    qml.Hadamard(wires=i)
    qml.PauliZ(wires=i)
    qml.CNOT(wires=[i, j])
    qml.PauliX(wires=j)