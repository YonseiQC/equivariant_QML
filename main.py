import pennylane as qml
import numpy as np
import jax
import jax.numpy as jnp
import optax
from gates import Spin_2, Spin_3, create_singlet
import pandas as pd
import math
import pennylane.numpy as pnp
import itertools

jax.config.update("jax_enable_x64", True)

dev = qml.device("default.qubit", wires=18)  
key = jax.random.key(1557)

def loss(num_qubit, H, answer_train, index): 
    H_sigmoid = 1 / (1 + pnp.exp(-H))
    cross_entropy = 0

    for i  in range (num_qubit * index, num_qubit * (index + 1)):
        cross_entropy += answer_train[i] * pnp.log(H_sigmoid) + (1 - answer_train[i]) * pnp.log(1 - H_sigmoid)
    
    cross_entropy = -cross_entropy
    cross_entropy = cross_entropy / num_qubit

    return cross_entropy

def encode(num_wire, vector):

    # theta = Alpha * np.arccos(vector[2])
    # phi = Beta * np.arctan2(vector[1], vector[0])
    
    # qml.RY(theta, wires = num_wire)  
    # qml.RZ(phi, wires = num_wire)  
    # qml.RY(theta, wires = num_wire + 1)  
    # qml.RZ(phi, wires = num_wire + 1)  

    # qml.RX(vector[0] * Alpha, wires = num_wire)
    # qml.RY(vector[1] * Beta, wires = num_wire)
    # qml.RZ(vector[2], wires = num_wire)
    # qml.RX(vector[0] * Alpha, wires = num_wire + 1)
    # qml.RY(vector[1] * Beta, wires = num_wire + 1)
    # qml.RZ(vector[2], wires = num_wire + 1)

    # theta = 2 * Alpha * np.arctan(- (np.cos(vector[0]) * np.sin(vector[1]) * np.cos(vector[2]) + np.sin(vector[0]) * np.cos(vector[1]) * np.sin(vector[2])) / (np.cos(vector[0]) * np.sin(vector[1]) * np.sin(vector[2]) - np.sin(vector[0]) * np.cos(vector[1]) * np.cos(vector[2])))
    # phi = 2 * Beta * np.arctan((np.cos(vector[0]) * np.sin(vector[1]) * np.sin(vector[2]) - np.sin(vector[0]) * np.cos(vector[1]) * np.cos(vector[2])) / np.cos(vector[0]) * np.cos(vector[1]) * np.cos(vector[2]) + np.sin(vector[0]) * np.sin(vector[1]) * np.sin(vector[2]))

    # Alpha = 2 * np.arctan(vector[2] * np.tan(Theta/2))
    # Beta = 2 * np.arctan(vector[0] * np.tan(Theta/2))
    # if vector[0] * vector[2] * np.tan(Theta/2) == vector[1]: 
    #     print("1")
    # else:
    #     print("0")

    # qml.RZ(Alpha, wires = num_wire)
    # qml.RX(Beta, wires = num_wire)
    # qml.RZ(Alpha, wires = num_wire + 1)
    # qml.RX(Beta, wires = num_wire + 1)

    Alpha = np.arctan(vector[2] * np.tan(Theta/2)) + np.arctan(vector[1]/vector[0])
    Gamma = np.arctan(vector[2] * np.tan(Theta/2)) - np.arctan(vector[1]/vector[0])
    Beta = 2 * np.arccos(vector[2] * np.sin(Theta/2) / np.sin((Alpha + Gamma)/2))
    qml.RZ(Alpha, wires = num_wire)
    qml.RX(Beta, wires = num_wire)
    qml.RZ(Gamma, wires = num_wire)
    qml.RZ(Alpha, wires = num_wire + 1)
    qml.RX(Beta, wires = num_wire + 1)
    qml.RZ(Gamma, wires = num_wire + 1) 



def create_Hamiltonian(num_qubit):
    terms = []
    for i in range(num_qubit-1):
        for j in range(i+1, num_qubit):
            terms.append(qml.PauliX(i) @ qml.PauliX(j))
            terms.append(qml.PauliY(i) @ qml.PauliY(j))
            terms.append(qml.PauliZ(i) @ qml.PauliZ(j))
    
    return sum(terms)


def prepare_init_state(num_qubit):
    for i in range(0, num_qubit, 2):
        create_singlet(i, i+1) 

def create_u2_circuit(num_qubit, num_blocks, H, Theta):
    def circuit_2qubits(params, data_pt):
        prepare_init_state(num_qubit)
        for i, point in enumerate(data_pt):
            encode(i, point / Theta)

        k = 0
        for l in range(num_blocks):
            for i in range(0, num_qubit, 2):
                Spin_2(params[k], wires=[i, (i + 1) % num_qubit])
                k += 1

            for i in range(1, num_qubit, 2):
                Spin_2(params[k], wires=[i, (i + 1) % num_qubit])
                k += 1

            for i in range(0, num_qubit):
                Spin_2(params[k], wires=[i, (i + 2) % num_qubit])
                k += 1

        return qml.expval(H)

    return circuit_2qubits


def create_u3_circuit(num_qubit, num_blocks, H, Theta, data, index):

    def circuit_3qubits(params):

        prepare_init_state(num_qubit)
        for i, vector in enumerate(data[num_qubit * index : num_qubit * (index + 1)]):
            num_wire = i % num_qubit
            if i % 2 == 0:
                encode(num_wire, vector, Theta)

        k = 0
        for l in range(num_blocks):
            for i in range(0, num_qubit):
                Spin_3(params[k], params[k + 1], params[k + 2], params[k+3], wires=[i, (i + 1) % num_qubit, (i + 2) % num_qubit])
                k += 4
        return qml.expval(H)

    return circuit_3qubits

def train(gate_type, dataset, minibatch_size, Theta, epochs, key, **adam_opt):
    train_dataset_x = dataset['train_dataset_x']
    train_dataset_y = dataset['train_dataset_y']
    test_dataset_x = dataset['test_dataset_x']
    test_dataset_y = dataset['test_dataset_y']

    assert len(train_dataset_x) == len(train_dataset_y)
    assert len(train_dataset_x) % minibatch_size == 0
    batch_size = len(train_dataset_x)

    train_dataset_x = train_dataset_x.reshape(batch_size // minibatch_size, minibatch_size, -1, 3)
    train_dataset_y = train_dataset_y.reshape(batch_size // minibatch_size, minibatch_size)

    ham = create_Hamiltonian(num_qubit)
    ham_sparse = qml.SparseHamiltonian(ham.sparse_matrix(), wires=range(18))

    if gate_type == "u2":
        init_u2 = init_scale * math.pi/(2 * num_qubit * num_blocks)
        params = init_u2 * jax.random.uniform(key, (2 * num_qubit * num_blocks,))

        circuit = qml.QNode(create_u2_circuit(num_qubit, num_blocks, ham_sparse, Theta), device = dev, interface='jax')

        logits = circuit(params, train_dataset_x[0])

        def loss_fn(params, mini_batch_x, mini_batch_y):
            logits = circuit(params, mini_batch_x)
            return optax.losses.sigmoid_binary_cross_entropy(logits, mini_batch_y)

        solver = optax.adam(**adam_opt)
        for epoch in range(epochs):
            for i in range(batch_size // minibatch_size):
                grad = jax.grad(loss_fn, argnums=0)(params, train_dataset_x[i], train_dataset_y[i])
                updates, opt_state = solver.update(grad, opt_state, params)
                params = optax.apply_updates(params, updates)
            test_error = jnp.mean((circuit(params, test_dataset_x) > 0) == test_dataset_y)
            print(f"{epoch}, {test_error}")

    """
    if (gate_type) == ("u3"):
        init_u3 = init_scale*math.pi/(4 * num_qubit * num_blocks)
        params_u3 = init_u3 * pnp.random.rand(4 * num_qubit * num_blocks)
        for index in range(batch_train):
            circuit_u3 = qml.QNode(create_u3_circuit(num_qubit, num_blocks, ham_sparse, Theta, feats_train, index), dev, diff_method="parameter-shift")
            
            for epoch in range(epochs):   
                params_u3, cost = opt_u3.step_and_cost(lambda p: loss(num_qubit, circuit_u3(p), answer_train, index), params_u3)
                print(f"{epoch}\t{cost}")
                print(params_u3)

        params_train = params_u3
        print(np.array(params_train))
        return params_train

    else:
        print("Type Error")
    """


# Load dataset
dataset = np.load('dataset.npz')

num_qubit = 18
Theta = 8.0

num_blocks = 4
init_scale = 1
epochs = 10

key, key_r = jax.random.split(key)
train("u2", dataset, 8, Theta, epochs, key_r, learning_rate = 0.03)