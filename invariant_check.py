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

'''
def encode(point):
    point_sqr = jnp.power(point, 2)
    px = jnp.array([[0, 1], [1 ,0]])
    py = jnp.array([[0, -1j], [1j, 0]])
    pz = jnp.array([[1, 0], [0, -1]])
    norms = jnp.sqrt(jnp.sum(point_sqr, axis = -1))
    nx, ny, nz = point[:, :, 0] / norms, point[:,:,1] / norms, point[:,:,2] / norms

    unitary_mats = []
    for i in range(norms.shape[0]):
        for j in range(norms.shape[1]):
            unitary_mat = jnp.cos(norms[i,j]) * jnp.eye(2) + 1j*jnp.sin(norms[i,j]) * (nx[i,j] * px + ny[i,j] * py + nz[i,j] * pz)
            unitary_mats.append(unitary_mat)
    unitary_mats = jnp.array(unitary_mats)
    unitary_mats = unitary_mats.reshape(norms.shape[0], norms.shape[1], 2, 2)

    for i in range(num_qubit):
        qml.QubitUnitary(unitary_mats[:,i], wires=[i])
'''

def encode(point):
    point_sqr = jnp.power(point, 2)
    norms = jnp.sqrt(jnp.sum(point_sqr, axis = -1))
    nx, ny, nz = point[:,:,0] / norms, point[:,:,1] / norms, point[:,:,2] / norms

    Alpha = jnp.arctan2(-(nz * jnp.tan(norms)),1) + jnp.arctan2(-point[:,:,0], point[:,:,1])
    Gamma = jnp.arctan2(-(nz * jnp.tan(norms)),1) - jnp.arctan2(-point[:,:,0], point[:,:,1])
    Beta = 2 * jnp.arcsin(jnp.sin(norms) * nx / jnp.sin((Alpha - Gamma) / 2))
    Alpha_Transpose = Alpha.T
    Beta_Transpose = Beta.T
    Gamma_Transpose = Gamma.T
    for i in range(num_qubit):
        qml.RZ(Alpha_Transpose[i], wires= i)
        qml.RY(Beta_Transpose[i], wires = i)
        qml.RZ(Gamma_Transpose[i], wires = i)

"""
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
"""

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
            data = data_pt[:,i,:,:]
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

    
def train(gate_type, dataset, minibatch_size, Theta, epochs, key, init_scale, num_blocks_reupload, num_blocks_circuit, **adam_opt):
    train_dataset_x = dataset['train_dataset_x'][:4096]
    train_dataset_y = dataset['train_dataset_y'][:4096]
    test_dataset_x = dataset['test_dataset_x'][:256]
    test_dataset_y = dataset['test_dataset_y'][:256]
    # train_dataset_x = dataset['train_dataset_x']
    # train_dataset_y = dataset['train_dataset_y']
    # test_dataset_x = dataset['test_dataset_x']
    # test_dataset_y = dataset['test_dataset_y']
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
    train_dataset_y = train_dataset_y.reshape(batch_size // minibatch_size, minibatch_size)
    ham = create_Hamiltonian(num_qubit)
    # for i in range(num_reupload):
    #     print(f'{i + 1}th 8 points')
    #     dataset_x = (test_dataset_x[10,i,:,:] / Theta)
    #     dataset_x_sqr = np.power(dataset_x, 2)
    #     norms = np.sqrt(np.sum(dataset_x_sqr, axis = -1))
    #     nx, ny, nz = dataset_x[:,0] / norms, dataset_x[:,1] / norms, dataset_x[:,2] / norms
    #     Alpha = jnp.arctan2(-(nz * jnp.tan(norms)),1) + jnp.arctan2(-dataset_x[:,0], dataset_x[:,1])
    #     Gamma = jnp.arctan2(-(nz * jnp.tan(norms)),1) - jnp.arctan2(-dataset_x[:,0], dataset_x[:,1])
    #     Beta = 2 * jnp.arcsin(jnp.sin(norms) * nx / jnp.sin((Alpha - Gamma) / 2))
    #     a = jnp.sin(norms) * nx
    #     b = jnp.sin((Alpha - Gamma) / 2)
    #     c = a / b
    #     print(a)
    #     print(b)
    #     print(c)
    #     print("\n")
    #     print(f'{i + 1}th norms')
    #     print(norms)
    #     print("\n")
    #     print("Alpha, Beta, Gamma")
    #     print(f'{Alpha}, {Beta}, {Gamma}')
    #     print("\n")

    if gate_type == "u2":
        init_u2 = init_scale * math.pi/(2 * num_qubit * (num_blocks_reupload + num_blocks_circuit))
        params_q = init_u2 * jax.random.uniform(key, (2 * num_qubit * (num_blocks_reupload * num_reupload + num_blocks_circuit),))

        dummy_input = jnp.ones((1, math.comb(num_qubit, 2)))  
        params_c = MyNN().init(key, dummy_input)

        params ={"q" : params_q, "c" : params_c}

        u2_circuit = qml.QNode(create_u2_circuit(num_qubit, num_blocks_reupload, num_blocks_circuit, ham, Theta), device = dev, interface='jax')
        u2_circuit = jax.jit(u2_circuit)

        test_dataset_y = test_dataset_y[:, None]
        expval_ham = (jnp.array(u2_circuit(params, test_dataset_x))).T
        logits = NN_circuit(expval_ham, params)
        print(logits)
        return logits

    else:
        print("Type Error")


# Load dataset
num_qubit = 8
Theta = 3
num_reupload = 1
gate_type = "u2"
test_learning_rate = 0.001
num_blocks_reupload = 2
num_blocks_circuit = 0
init_scale = 0.05
epochs = 1

dev = qml.device("default.qubit", wires = num_qubit)
dataset = np.load(f'modelnet40_10classes_{num_qubit}_{num_reupload}.npz')
# dataset = np.load(f'dataset_{num_qubit}_{num_reupload}.npz')
key, key_r = jax.random.split(key)

def result(gate_type, test_learning_rate, num_blocks_reupload, num_blocks_circuit, init_scale):
    train(gate_type, dataset, 16, Theta, epochs, key_r, init_scale, num_blocks_reupload, num_blocks_circuit, learning_rate = test_learning_rate)

a = result(gate_type, test_learning_rate, num_blocks_reupload, num_blocks_circuit, init_scale)

dataset = np.load(f'modelnet40_10classes_{num_qubit}_{num_reupload}_rotation.npz')
# dataset = np.load(f'dataset_{num_qubit}_{num_reupload}_rotation.npz')

b = result(gate_type, test_learning_rate, num_blocks_reupload, num_blocks_circuit, init_scale)
if np.array_equal(a, b) == True:
    print("They are rotation invariant")
else:
    print("They are not rotation invariant")


# 위의 equal 부분 틀린 듯
# torus의 norm이 작아서 키움
# 사실 torus의 norm이 너무 편차가 큰 것이 문제여서 norm을 1이하로 줄여주는 Theta 찾아서 나눠줌
# Beta값이 1을 넘어가면 수치불안정성이 커져서 beta값이 1을 넘어가는 주기를 파악하고 그 주기에 수가 오지 못하도록 Theta를 설정함
