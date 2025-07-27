import pennylane as qml
import numpy as np
import jax
import jax.numpy as jnp
import optax
from gates import create_singlet
from gates_twirling import Spin_2_twirling, Spin_3_twirling
import math
import pennylane.numpy as pnp
from flax import linen as nn
import matplotlib.pyplot as plt


jax.config.update("jax_enable_x64", True)

key = jax.random.key(1557)

def get_Theta(point):
    point = point['train_dataset_x']
    point_sqr = jnp.power(point, 2)
    norms = jnp.sqrt(jnp.sum(point_sqr, axis = -1))
    Theta = jnp.max(norms) * 1.2
    return Theta


class MyNN(nn.Module):
    @nn.compact
    def __call__(self, x):

        x = jnp.expand_dims(x, axis=-1)
        x = nn.Dense(features=16)(x)  
        x = nn.relu(x)
        x = nn.Dense(features=16)(x) 
        x = nn.relu(x)
        x = nn.Dense(features=8)(x) 

        x = jnp.mean(x, axis=1) 
        x = nn.Dense(features=32)(x) 
        x = nn.relu(x)
        x = nn.Dense(features=16)(x) 
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)  
        
        return x

def NN_circuit(dataset, params):
    return MyNN().apply(params["c"], dataset)


def encode(point):
    point_sqr = jnp.power(point, 2)
    norms = jnp.sqrt(jnp.sum(point_sqr, axis = -1))
    nx, ny, nz = point[:,:,0] / norms, point[:,:,1] / norms, point[:,:,2] / norms

    Alpha = jnp.arctan2(-(nz * jnp.tan(norms)),1) + jnp.arctan2(-nx, ny)
    Gamma = jnp.arctan2(-(nz * jnp.tan(norms)),1) - jnp.arctan2(-nx, ny)
    Beta = 2 * jnp.arcsin(jnp.sin(norms) * nx / jnp.sin((Alpha - Gamma) / 2))
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


def create_twirling_circuit(num_qubit, num_blocks_reupload, num_reupload, H):

    def twirling_circuit(params, data_pt): 
        prepare_init_state(num_qubit)
        k = 0
        for i in range(num_reupload):
            data = data_pt[:,i,:,:]
            encode(data / Theta)

            for l in range(num_blocks_reupload):
                Spin_2_twirling(params["q"][k], wires = range(num_qubit))
                Spin_3_twirling(params["q"][k + 1], wires = range(num_qubit))
                k += 2

        return [qml.expval(h) for h in H]

    return twirling_circuit

    
def train(gate_type, dataset, minibatch_size, epochs, key, init_scale, num_blocks_reupload, **adam_opt):
    train_dataset_x = dataset['train_dataset_x']
    train_dataset_y = dataset['train_dataset_y']
    test_dataset_x = dataset['test_dataset_x']
    test_dataset_y = dataset['test_dataset_y']
    assert len(train_dataset_x) == len(train_dataset_y)
    assert len(train_dataset_x) % minibatch_size == 0
    batch_size = len(train_dataset_x)

    test_dataset_x = test_dataset_x.reshape(256, num_reupload, -1, 3)
    # point_sqr = jnp.power(test_dataset_x, 2)
    # norms = jnp.sqrt(jnp.sum(point_sqr, axis = -1))
    # test_dataset_x = test_dataset_x / norms.reshape(test_dataset_x.shape[0], num_reupload, num_qubit, 1)

    train_dataset_x = train_dataset_x.reshape(batch_size // minibatch_size, minibatch_size, num_reupload, -1, 3)
    # point_sqr = jnp.power(train_dataset_x, 2)
    # norms = jnp.sqrt(jnp.sum(point_sqr, axis = -1))
    # train_dataset_x = train_dataset_x / norms.reshape(train_dataset_x.shape[0], minibatch_size, num_reupload, num_qubit, 1)
    train_dataset_y = train_dataset_y.reshape(batch_size // minibatch_size, minibatch_size)

    ham = create_Hamiltonian(num_qubit)

    if gate_type == "u2":
        init_u2 = init_scale * math.pi/(2 * (num_blocks_reupload * num_reupload))
        params_q = init_u2 * jax.random.uniform(key, (2 * (num_blocks_reupload * num_reupload),))

        dummy_input = jnp.ones((1, math.comb(num_qubit, 2)))  
        params_c = MyNN().init(key, dummy_input)

        params = {"q" : params_q, "c" : params_c}

        twirling_circuit = qml.QNode(create_twirling_circuit(num_qubit, num_blocks_reupload, num_reupload, ham), device = dev, interface='jax')
        twirling_circuit = jax.jit(twirling_circuit)

        # drawer = qml.draw(u2_circuit, show_all_wires=True, decimals=None)
        # print(drawer(params, test_dataset_x))


        def loss_fn(params, mini_batch_x, mini_batch_y, l2):
            expval_ham = (jnp.array(twirling_circuit(params, mini_batch_x))).T
            logits = NN_circuit(expval_ham, params)
            mini_batch_y = mini_batch_y[:, None]
            loss = jnp.mean(optax.losses.sigmoid_binary_cross_entropy(logits, mini_batch_y))
            l2_penalty = l2 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
            return (loss + l2_penalty)
            # return loss

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
        train_loss_lst = []
        test_loss_lst = []
        succed_lst = []
        for epoch in range(epochs):
            train_loss = 0
            for i in range(batch_size // minibatch_size):
                loss, grad = jax.value_and_grad(loss_fn, argnums=0)(params, train_dataset_x[i], train_dataset_y[i], l2)
                # loss, grad = jax.value_and_grad(loss_fn, argnums=0)(params, train_dataset_x[i], train_dataset_y[i])
                updates, opt_state = solver.update(grad, opt_state, params)
                params = optax.apply_updates(params, updates)
                train_loss += loss / (batch_size // minibatch_size)           
                # expval_ham = (jnp.array(u2_circuit(params, test_dataset_x))).T
                # logits = NN_circuit(expval_ham, params)
                # print(jnp.mean((logits > 0) == test_dataset_y))
            train_loss_lst.append(train_loss)
            print(train_loss) 
            test_loss = loss_fn(params, test_dataset_x, test_dataset_y, l2) 
            expval_ham = (jnp.array(twirling_circuit(params, test_dataset_x))).T
            logits = NN_circuit(expval_ham, params)
            succed = jnp.mean(((logits) > 0) == test_dataset_y)
            test_loss_lst.append(test_loss)
            print(test_loss)
            print(f"{epoch}, {succed}")
            succed_lst.append(succed)
        plt.plot(test_loss_lst, label='Test Loss', marker='o', linestyle='-', color='b')
        plt.plot(train_loss_lst, label='Training Loss', marker='o', linestyle='-', color='r')
        plt.plot(succed_lst, label='Test Accuracy', marker='o', linestyle='-', color='y')
        plt.title('Loss & Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Loss & Test Accuracy')
        plt.legend()  
        plt.grid(True)
        plt.show()



# Load dataset
num_qubit = 8
num_reupload = 2
gate_type = "u2"
test_learning_rate = 0.005
num_blocks_reupload = 6
init_scale = 0.02
dev = qml.device("default.qubit", wires = num_qubit)
dataset = np.load(f'dataset_{num_qubit}_{num_reupload}.npz')
Theta = 10
epochs = 40
l2 = 0.00001
key, key_r = jax.random.split(key)

def result(gate_type, test_learning_rate, num_blocks_reupload, init_scale):
    print(f'qubits = {num_qubit}, gate_type = {gate_type}, test_learning_rate = {test_learning_rate}, num_blocks_reupload = {num_blocks_reupload}, init_scale= {init_scale}, num_reupload = {num_reupload}')
    train(gate_type, dataset, 32, epochs, key_r, init_scale, num_blocks_reupload, learning_rate = test_learning_rate)

result(gate_type, test_learning_rate, num_blocks_reupload, init_scale)


# # test loss가 높은 이유 : sphere, torus 구분이 애매한 point에서 크게 틀림
# # memory : 8 qubits, 6 blocks, 2 reupload, learning rate = 0.005, init_scale = 0.05, l2 = 0.00001, theta = 10 