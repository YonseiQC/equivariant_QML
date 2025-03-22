import pennylane as qml
import numpy as np
from pennylane import numpy as np
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


def encode(point):
    point_sqr = jnp.power(point, 2)
    norms = jnp.sqrt(jnp.sum(point_sqr, axis = -1))
    
    ny, nz = point[:,:,1] / norms, point[:,:,2] / norms

    Alpha = (jnp.arctan2(-nz * jnp.tan(norms),1) + jnp.arctan2(point[:,:,0], point[:,:,1]))
    Gamma = (jnp.arctan2(-nz * jnp.tan(norms),1) - jnp.arctan2(point[:,:,0], point[:,:,1]))
    Beta = 2 * jnp.arcsin(jnp.sin(norms) * ny / jnp.cos((Alpha - Gamma) / 2))
    Alpha_Transpose = Alpha.T
    Beta_Transpose = Beta.T
    Gamma_Transpose = Gamma.T
    for i in range(num_qubit):
        qml.RZ(Alpha_Transpose[i], wires= i)
        qml.RY(Beta_Transpose[i], wires = i)
        qml.RZ(Gamma_Transpose[i], wires = i)


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
        print(params)
        prepare_init_state(num_qubit)
        encode(data_pt / Theta)

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

        def loss_fn(params, mini_batch_x, mini_batch_y):
            logits = u2_circuit(params, mini_batch_x)
            loss = optax.losses.sigmoid_binary_cross_entropy(logits, mini_batch_y)
            return jnp.mean(loss)

        solver = optax.adam(**adam_opt)
        opt_state = solver.init(params)
        for epoch in range(epochs):
            for i in range(batch_size // minibatch_size):
                grad = jax.grad(loss_fn, argnums=0)(params, train_dataset_x[i], train_dataset_y[i])
                updates, opt_state = solver.update(grad, opt_state, params)
                params = optax.apply_updates(params, updates)
            test_error = jnp.mean((u2_circuit(params, test_dataset_x) > 0) == test_dataset_y)
            print(params)
            print(u2_circuit(params, test_dataset_x))
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

ham = create_Hamiltonian(num_qubit)
u2_circuit = qml.QNode(create_u2_circuit(num_qubit, num_blocks, ham, Theta), device = dev, interface='jax')
test_dataset_x = dataset['test_dataset_x']
test_dataset_x = np.array(test_dataset_x, requires_grad=False)

params = jnp.array([ 0.85142214,  0.41825706,  1.49712325,  1.13521764, -0.02030997,  0.4071614,
        0.64298772,  0.61219278,  1.7481547,   0.22809032, -0.37397173,  1.63045238,
        0.5437375,   0.33295093,  0.32118305,  1.1081797,   1.23719945,  0.56180076,
        0.51157736,  1.37041288,  0.14383623,  1.24244479, 0.23567364,  0.63750947,
        0.93657527,  0.59049892,  0.90049832,  0.79219983,  0.95790891,  0.06524483,
        0.39271201,  0.8688721,   0.92621896,  0.87231892,  0.66146446,  0.88930409,
        0.97180253,  1.02071485,  0.34301263,  0.01608313,  0.83678728,  0.67140582,
        1.09949514,  0.94271238,  0.6954175,   0.21848174,  0.48004806,  0.70997085])
print(u2_circuit(params, test_dataset_x))
print(qml.draw(u2_circuit)(params, test_dataset_x))

params = jnp.array([ 1000000000,  0.41825706, 10000000000000000,  1, 1,  -1,
        0.64298772,  0.61219278,  1,   0.22809032, -0.37397173,  1.63045238,
        0.5437375,   1,  0.32118305,  1.1081797,   1.23719945,  0.56180076,
        0.51157736,  1.37041288,  0.14383623,  0, 0.23567364,  0.63750947,
        0.93657527,  2,  0.90049832,  0.79219983,  1,  0.06524483,
        0.39271201,  0.8688721,   0.92621896,  0.87231892,  0.66146446,  0.88930409,
        0.97180253,  1.02071485,  0,  0.01608313,  0.83678728,  0.67140582,
        1.09949514,  0.94271238,  0.6954175,   0.21848174,  0.48004806,  0.70997085])
print(u2_circuit(params, test_dataset_x))

#train("u2", dataset, 8, Theta, epochs, key_r, learning_rate = 0.03 * 10000000)
