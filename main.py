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
from jax.experimental import sparse as jsparse



jax.config.update("jax_enable_x64", True)

dev = qml.device("default.qubit", wires= 4)
key = jax.random.key(1557)


def encode(point, Theta):
    for i  in range(num_qubit): 
        norm = jnp.sqrt((point[i][0] ** 2) + (point[i][1] ** 2) + (point[i][2] ** 2))
        Alpha = jnp.arctan2(-point[i][2] * Theta * jnp.tan(norm),1) + jnp.arctan2(point[i][1],point[i][0])
        Gamma = jnp.arctan2(-point[i][2] * Theta * jnp.tan(norm),1) - jnp.arctan2(point[i][1],point[i][0])
        Beta = 2 * jnp.arccos(jnp.clip(jnp.cos(norm) / jnp.cos((Alpha + Gamma)/2), -1, 1)) # Beta is not correct
        qml.RZ(Alpha, wires= i)
        qml.RX(Beta, wires = i)
        qml.RZ(Gamma, wires = i)


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
        encode(data_pt / Theta, Theta)

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



def create_u3_circuit(num_qubit, num_blocks, H, Theta):

    def circuit_3qubits(params, data_pt):
        prepare_init_state(num_qubit)
        encode(data_pt / Theta, Theta)

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

    if gate_type == "u2":
        init_u2 = init_scale * math.pi/(2 * num_qubit * num_blocks)
        params = init_u2 * jax.random.uniform(key, (2 * num_qubit * num_blocks,))

        u2_circuit = qml.QNode(create_u2_circuit(num_qubit, num_blocks, ham, Theta), device = dev, interface='jax')
        vmap_u2_circuit = jax.vmap(u2_circuit, in_axes=(None, 0))

        def loss_fn(params, mini_batch_x, mini_batch_y):
            logits = vmap_u2_circuit(params, mini_batch_x)
            loss = optax.losses.sigmoid_binary_cross_entropy(logits, mini_batch_y)
            return jnp.mean(loss)
        
        solver = optax.adam(**adam_opt)
        opt_state = solver.init(params)
        for epoch in range(epochs):
            for i in range(batch_size // minibatch_size):
                grad = jax.grad(loss_fn, argnums=0)(params, train_dataset_x[i], train_dataset_y[i])
                updates, opt_state = solver.update(grad, opt_state, params)
                params = optax.apply_updates(params, updates)
            test_error = jnp.mean((vmap_u2_circuit(params, test_dataset_x) > 0) == test_dataset_y)
            print(params)
            print(vmap_u2_circuit(params, test_dataset_x))
            print(f"{epoch}, {test_error}")

    if (gate_type) == ("u3"):
        init_u3 = init_scale*math.pi/(4 * num_qubit * num_blocks)
        params = init_u3 * pnp.random.rand(4 * num_qubit * num_blocks)

        u3_circuit = qml.QNode(create_u3_circuit(num_qubit, num_blocks, ham, Theta), device = dev, interface='jax')
        vmap_u3_circuit = jax.vmap(u3_circuit, in_axes=(None, 0))

        def loss_fn(params, mini_batch_x, mini_batch_y):
            logits = vmap_u3_circuit(params, mini_batch_x)
            loss = optax.losses.sigmoid_binary_cross_entropy(logits, mini_batch_y)
            return jnp.mean(loss)
        
        solver = optax.adam(**adam_opt)
        opt_state = solver.init(params)
        for epoch in range(epochs):
            for i in range(batch_size // minibatch_size):
                grad = jax.grad(loss_fn, argnums=0)(params, train_dataset_x[i], train_dataset_y[i])
                updates, opt_state = solver.update(grad, opt_state, params)
                params = optax.apply_updates(params, updates)
            test_error = jnp.mean((vmap_u3_circuit(params, test_dataset_x) > 0) == test_dataset_y)
            print(params)
            print(vmap_u3_circuit(params, test_dataset_x))
            print(f"{epoch}, {test_error}")

    else:
        print("Type Error")


# Load dataset
num_qubit = 4
Theta = 1

dataset = np.load(f'dataset_{num_qubit}.npz')

num_blocks = 6
init_scale = 0.0000002 * 100000000
epochs = 10

key, key_r = jax.random.split(key)
train("u2", dataset, 8, Theta, epochs, key_r, learning_rate = 0.03 * 10000000)
