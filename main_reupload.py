import pennylane as qml
import numpy as np
import jax
import jax.numpy as jnp
import optax
from gates import Spin_2, Spin_3, create_singlet
import math
import pennylane.numpy as pnp
from flax import linen as nn



jax.config.update("jax_enable_x64", True)

key = jax.random.key(1557)

class MyNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features = math.comb(num_qubit, 2))(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x
def NN_circuit(dataset, params):
    return MyNN().apply(params["c"], dataset)


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
    for i in range(num_qubit - 1):
        for j in range(i + 1, num_qubit):
            terms.append(qml.PauliX(i) @ qml.PauliX(j) + qml.PauliY(i) @ qml.PauliY(j) + qml.PauliZ(i) @ qml.PauliZ(j))
    return terms


def prepare_init_state(num_qubit):
    for i in range(0, num_qubit, 2):
        create_singlet(i, i+1) 


def create_u2_circuit(num_qubit, num_blocks_reupload, num_blocks_circuit, H, Theta):

    def u2_circuit(params, data_pt): 
        prepare_init_state(num_qubit)
        k = 0
        for i in range(num_reupload):
            data = data_pt[i]
            encode(data / Theta)

            for l in range(num_blocks_reupload):
                for i in range(0, num_qubit, 2):
                    Spin_2(params["q"][k], wires=[(i) % num_qubit, (i + 1) % num_qubit])
                    k += 1

                for i in range(1, num_qubit, 2):
                    Spin_2(params["q"][k], wires=[(i) % num_qubit, (i + 1) % num_qubit])
                    k += 1

                for i in range(0, num_qubit):
                    Spin_2(params["q"][k], wires=[(i) % num_qubit, (i + 2) % num_qubit])
                    k += 1

        for l in range(num_blocks_circuit):
            for i in range(0, num_qubit, 2):
                Spin_2(params["q"][k], wires=[(i) % num_qubit, (i + 1) % num_qubit])
                k += 1

            for i in range(1, num_qubit, 2):
                Spin_2(params["q"][k], wires=[(i) % num_qubit, (i + 1) % num_qubit])
                k += 1

            for i in range(0, num_qubit):
                Spin_2(params["q"][k], wires=[(i) % num_qubit, (i + 2) % num_qubit])
                k += 1

        return [qml.expval(h) for h in H]

    return u2_circuit


def create_u3_circuit(num_qubit, num_blocks, H, Theta):

    def u3_circuit(params, data_pt):
        prepare_init_state(num_qubit)
        encode(data_pt / Theta)

        k = 0
        for l in range(num_blocks):
            for i in range(0, num_qubit):
                Spin_3(params["q"][k], params["q"][k + 1], params["q"][k + 2], params["q"][k+3], wires=[(i) % num_qubit, (i + 1) % num_qubit, (i + 2) % num_qubit])
                k += 4

        return [qml.expval(h) for h in H]
    
    return u3_circuit

    

def train(gate_type, dataset, minibatch_size, Theta, epochs, key, init_scale, num_blocks_reupload, num_blocks_circuit, **adam_opt):
    train_dataset_x = dataset['train_dataset_x']
    train_dataset_y = dataset['train_dataset_y']
    test_dataset_x = dataset['test_dataset_x']
    test_dataset_y = dataset['test_dataset_y']
    assert len(train_dataset_x) == len(train_dataset_y)
    assert len(train_dataset_x) % minibatch_size == 0
    batch_size = len(train_dataset_x)

    test_dataset_x = test_dataset_x.reshape(num_reupload, 256, -1, 3)
    train_dataset_x = train_dataset_x.reshape(batch_size // minibatch_size, num_reupload, minibatch_size, -1, 3)
    train_dataset_y = train_dataset_y.reshape(batch_size // minibatch_size, minibatch_size)
    ham = create_Hamiltonian(num_qubit)

    if gate_type == "u2":
        init_u2 = init_scale * math.pi/(2 * num_qubit * (num_blocks_reupload + num_blocks_circuit))
        params_q = init_u2 * jax.random.uniform(key, (2 * num_qubit * (num_blocks_reupload * num_reupload + num_blocks_circuit),))

        dummy_input = jnp.ones((1, math.comb(num_qubit, 2)))  
        params_c = MyNN().init(key, dummy_input)

        params ={"q" : params_q, "c" : params_c}


        u2_circuit = qml.QNode(create_u2_circuit(num_qubit, num_blocks_reupload, num_blocks_circuit, ham, Theta), device = dev, interface='jax')
        u2_circuit = jax.jit(u2_circuit)


        def loss_fn(params, mini_batch_x, mini_batch_y):
            expval_ham = (jnp.array(u2_circuit(params, mini_batch_x))).T
            logits = NN_circuit(expval_ham, params)
            mini_batch_y = mini_batch_y[:, None]
            loss = optax.losses.sigmoid_binary_cross_entropy(logits, mini_batch_y)
            return jnp.mean(loss)

        """
        @jax.jit
        def training_loop_inner(params, opt_state, train_dataset_x, train_dataset_y):
            grad = jax.grad(loss_fn, argnums=0)(params, train_dataset_x[i], train_dataset_y[i])
            updates, opt_state = solver.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params
        """
        

        solver = optax.adam(**adam_opt)
        opt_state = solver.init(params)
        test_dataset_y = test_dataset_y[:, None]
        for epoch in range(epochs):
            for i in range(batch_size // minibatch_size):
                grad = jax.grad(loss_fn, argnums=0)(params, train_dataset_x[i], train_dataset_y[i])
                updates, opt_state = solver.update(grad, opt_state, params)
                params = optax.apply_updates(params, updates)            
                # expval_ham = (jnp.array(u2_circuit(params, test_dataset_x))).T
                # logits = NN_circuit(expval_ham, params)
                # print(jnp.mean((logits > 0) == test_dataset_y))
            expval_ham = (jnp.array(u2_circuit(params, test_dataset_x))).T
            logits = NN_circuit(expval_ham, params)
            succed = jnp.mean(((logits) > 0) == test_dataset_y)
            print(f"{epoch}, {succed}")


    elif gate_type == "u3":
        init_u3 = init_scale * math.pi/(4 * num_qubit * (num_blocks_reupload + num_blocks_circuit))
        params_q = init_u3 * pnp.random.rand(4 * num_qubit * (num_blocks_reupload * num_reupload + num_blocks_circuit))

        dummy_input = jnp.ones((1, math.comb(num_qubit, 2)))  
        params_c = MyNN().init(key, dummy_input)

        params ={"q" : params_q, "c" : params_c}

        u3_circuit = qml.QNode(create_u3_circuit(num_qubit, num_blocks_reupload, num_blocks_circuit, ham, Theta), device = dev, interface='jax')
        u3_circuit = jax.jit(u3_circuit)

        def loss_fn(params, mini_batch_x, mini_batch_y):
            expval_ham = (jnp.array(u3_circuit(params, mini_batch_x))).T
            logits = NN_circuit(expval_ham, params)
            mini_batch_y = mini_batch_y[:, None]
            loss = optax.losses.sigmoid_binary_cross_entropy(logits, mini_batch_y)
            return jnp.mean(loss)
        
        solver = optax.adam(**adam_opt)
        opt_state = solver.init(params)
        test_dataset_y = test_dataset_y[:, None]
        for epoch in range(epochs):
            for i in range(batch_size // minibatch_size):
                grad = jax.grad(loss_fn, argnums=0)(params, train_dataset_x[i], train_dataset_y[i])
                updates, opt_state = solver.update(grad, opt_state, params)
                params = optax.apply_updates(params, updates)
                # expval_ham = (jnp.array(u3_circuit(params, test_dataset_x))).T
                # logits = NN_circuit(expval_ham, params)
                # print(jnp.mean((logits > 0) == test_dataset_y))
            expval_ham = (jnp.array(u3_circuit(params, test_dataset_x))).T
            logits = NN_circuit(expval_ham, params)
            succed = jnp.mean(((logits) > 0) == test_dataset_y)
            print(f"{epoch}, {succed}")

    else:
        print("Type Error")


# Load dataset
num_qubit = 4
Theta = 1
num_reupload = 4

dev = qml.device("default.qubit", wires = num_qubit)
dataset = np.load(f'dataset_{num_qubit}_{num_reupload}.npz')
epochs = 20

key, key_r = jax.random.split(key)

def result(gate_type, test_learning_rate, num_blocks_reupload, num_blocks_circuit, init_scale):
    print(f'qubits = {num_qubit}, gate_type = {gate_type}, test_learning_rate = {test_learning_rate}, num_blocks_reupload = {num_blocks_reupload}, num_blocks_circuit = {num_blocks_circuit}, init_scale= {init_scale}')
    train(gate_type, dataset, 16, Theta, epochs, key_r, init_scale, num_blocks_reupload, num_blocks_circuit, learning_rate = test_learning_rate)

result("u2", 0.01, 2, 6, 0.05)
