from jax.numpy.linalg import eigh
import pennylane as qml
import itertools
import jax.numpy as jnp
import jax
import numpy as np



class Spin_2_twirling(qml.operation.Operation):
    num_params = 1
    num_wires = qml.operation.AnyWires
    grad_method = None
    
    @staticmethod
    def pauli_sum_function(i, num_qubit):
        qubit1 = 2 * i
        qubit2 = 2 * i + 1
        
        X = jnp.array([[0, 1], [1, 0]], dtype=complex)
        Y = jnp.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = jnp.array([[1, 0], [0, -1]], dtype=complex)
        identity = jnp.array([[1, 0], [0, 1]], dtype=complex)
        operators = [X, Y, Z]
        
        results = []
        for op in operators: 
            matrices1 = [identity] * num_qubit
            matrices1[qubit1] = op
            result1 = matrices1[0]
            for j in range(1, num_qubit):
                result1 = jnp.kron(result1, matrices1[j])
            
            matrices2 = [identity] * num_qubit
            matrices2[qubit2] = op
            result2 = matrices2[0]
            for j in range(1, num_qubit):
                result2 = jnp.kron(result2, matrices2[j])
            
            results.append(result1 + result2)
    
        return tuple(results) 

    @staticmethod
    def compute_decomposition(theta, wires):
        num_qubit = len(wires)
        dim = 2 ** num_qubit
        num_point = int(num_qubit / 2)
        Gate = jnp.zeros((dim, dim), dtype=complex)
        for i in range(0, num_point):
                for j in range(i + 1, num_point):
                    operator_1 = Spin_2_twirling.pauli_sum_function(i, num_qubit)
                    operator_2 = Spin_2_twirling.pauli_sum_function(j, num_qubit)
                    Gate += operator_1[0] @ operator_2[0] + operator_1[1] @ operator_2[1] + operator_1[2] @ operator_2[2]
        eigenvals, U = eigh(Gate)
        
        max_eigenval = jnp.max(jnp.abs(eigenvals))
        normalized_eigenvals = eigenvals / max_eigenval
        
        normalized_eigenvals = jnp.where(
            jnp.abs(normalized_eigenvals) < 1e-13,
            0.0,
            normalized_eigenvals
        )
        normalized_eigenvals = normalized_eigenvals * (10 ** 14)
        
        # jax.debug.print("Spin_2 normalized eigenvals: {}", normalized_eigenvals)

        return [
            qml.QubitUnitary(U.conj().T, wires=range(num_qubit)),
            qml.DiagonalQubitUnitary(jnp.exp(1j * theta * normalized_eigenvals), wires=range(num_qubit)),
            qml.QubitUnitary(U, wires=range(num_qubit))
        ]






class Spin_3_twirling(qml.operation.Operation):
    num_params = 1
    num_wires = qml.operation.AnyWires
    grad_method = None

    @staticmethod
    def pauli_sum_function(i, num_qubit):
        qubit1 = 2 * i
        qubit2 = 2 * i + 1
        
        X = jnp.array([[0, 1], [1, 0]], dtype=complex)
        Y = jnp.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = jnp.array([[1, 0], [0, -1]], dtype=complex)
        identity = jnp.array([[1, 0], [0, 1]], dtype=complex)
        operators = [X, Y, Z]
        
        results = []
        for op in operators: 
            matrices1 = [identity] * num_qubit
            matrices1[qubit1] = op
            result1 = matrices1[0]
            for j in range(1, num_qubit):
                result1 = jnp.kron(result1, matrices1[j])
            
            matrices2 = [identity] * num_qubit
            matrices2[qubit2] = op
            result2 = matrices2[0]
            for j in range(1, num_qubit):
                result2 = jnp.kron(result2, matrices2[j])
            
            results.append(result1 + result2)
        return tuple(results)

    @staticmethod
    def compute_i_j_k_term(operator_1, operator_2, operator_3):

        X_i, Y_i, Z_i = operator_1
        X_j, Y_j, Z_j = operator_2  
        X_k, Y_k, Z_k = operator_3
        
        terms = [
            X_i @ Y_j @ Z_k,  # X_i Y_j Z_k
            X_i @ Z_j @ Y_k,  # X_i Z_j Y_k  
            Y_i @ Z_j @ X_k,  # Y_i Z_j X_k
            Y_i @ X_j @ Z_k,  # Y_i X_j Z_k
            Z_i @ X_j @ Y_k,  # Z_i X_j Y_k
            Z_i @ Y_j @ X_k,  # Z_i Y_j X_k
        ]
        
        return terms[0] - terms[1] + terms[2] - terms[3] + terms[4] - terms[5]

    @staticmethod
    def compute_decomposition(theta, wires):
        num_qubit = len(wires)
        dim = 2 ** num_qubit
        num_point = int(num_qubit / 2)
        Gate = jnp.zeros((dim, dim), dtype=complex) 
        
        for i in range(0, num_point):
            for j in range(i + 1, num_point):
                for k in range(j + 1, num_point):

                    operator_1 = Spin_3_twirling.pauli_sum_function(i, num_qubit)
                    operator_2 = Spin_3_twirling.pauli_sum_function(j, num_qubit)
                    operator_3 = Spin_3_twirling.pauli_sum_function(k, num_qubit)
                    Gate += Spin_3_twirling.compute_i_j_k_term(operator_1, operator_2, operator_3)
    
        eigenvals, U = eigh(Gate)
        
        max_eigenval = jnp.max(jnp.abs(eigenvals))
        normalized_eigenvals = eigenvals / max_eigenval
        
        normalized_eigenvals = jnp.where(
            jnp.abs(normalized_eigenvals) < 1e-13,
            0.0,
            normalized_eigenvals
        )
        
        # jax.debug.print("Spin_3 normalized eigenvals: {}", normalized_eigenvals)
        
        return [
            qml.QubitUnitary(U.conj().T, wires=range(num_qubit)),
            qml.DiagonalQubitUnitary(jnp.exp(1j * theta * normalized_eigenvals), wires=range(num_qubit)),
            qml.QubitUnitary(U, wires=range(num_qubit))
        ]