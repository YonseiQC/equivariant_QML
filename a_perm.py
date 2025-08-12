import pennylane as qml
import numpy as np
import jax
import jax.numpy as jnp
import optax
from gates import create_singlet
from gates_twirling_perm import Spin_twirling
import math
import pennylane.numpy as pnp
from flax import linen as nn
import matplotlib.pyplot as plt


jax.config.update("jax_enable_x64", True)

key = jax.random.key(1557)

def get_Theta(point):
    point = point['train_dataset_x']
    point = point - jnp.mean(point, axis = 0)
    point_sqr = jnp.power(point, 2)
    norms = jnp.sqrt(jnp.sum(point_sqr, axis = -1))
    Theta = jnp.max(norms) * 1.2
    return Theta


class MyNN(nn.Module):
    @nn.compact
    def __call__(self, x):

        x = jnp.expand_dims(x, axis=-1)
        x = nn.Dense(features=30)(x)  
        x = nn.relu(x)
        x = nn.Dense(features=30)(x) 

        mean_pool = jnp.mean(x, axis=1)
        max_pool = jnp.max(x, axis=1) 
        min_pool = jnp.min(x, axis=1)
        std_pool = jnp.std(x, axis=1)

        x = jnp.concatenate([mean_pool, max_pool, min_pool, std_pool], axis=-1)
        x = nn.Dense(features=30)(x) 
        x = nn.relu(x)
        x = nn.Dense(features=15)(x) 
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)  
        
        return x
def NN_circuit(dataset, params):
    return MyNN().apply(params["c"], dataset)



# def encode(point):
#     point_sqr = jnp.power(point, 2)
#     norms = jnp.sqrt(jnp.sum(point_sqr, axis = -1))
#     nx, ny, nz = point[:,:,0] / norms, point[:,:,1] / norms, point[:,:,2] / norms

#     Alpha = jnp.arctan2(-(nz * jnp.tan(norms)),1) + jnp.arctan2(-nx, ny)
#     Gamma = jnp.arctan2(-(nz * jnp.tan(norms)),1) - jnp.arctan2(-nx, ny)
#     Beta = 2 * jnp.arcsin(jnp.sin(norms) * nx / jnp.sin((Alpha - Gamma) / 2))
#     Alpha_Transpose = Alpha.T
#     Beta_Transpose = Beta.T
#     Gamma_Transpose = Gamma.T
#     for i in range(int(num_qubit/2)):
#         qml.RZ(Alpha_Transpose[i], wires= 2 * i)
#         qml.RY(Beta_Transpose[i], wires = 2 * i)
#         qml.RZ(Gamma_Transpose[i], wires = 2 * i)


def encode(point, num_qubit):  
    point_sqr = jnp.power(point, 2)
    norms = jnp.sqrt(jnp.sum(point_sqr, axis = -1))
    nx, ny, nz = point[:,:,0] / norms, point[:,:,1] / norms, point[:,:,2] / norms
    
    norms_T = norms.T  
    nx_T = nx.T        
    ny_T = ny.T        
    nz_T = nz.T       
    
    for i in range(int(num_qubit / 2)):
        cos_n = jnp.cos(norms_T[i])
        sin_n = jnp.sin(norms_T[i])
        nx_i, ny_i, nz_i = nx_T[i], ny_T[i], nz_T[i]
        matrix_i = jnp.array([
            [cos_n + 1j * sin_n * nz_i, 1j * sin_n * nx_i + sin_n * ny_i],
            [1j * sin_n * nx_i - sin_n * ny_i, cos_n - 1j * sin_n * nz_i]
        ])
        matrix_i = jnp.moveaxis(matrix_i, [0, 1, 2], [1, 2, 0])
        
        qml.QubitUnitary(matrix_i, wires=2 * i)


def create_Hamiltonian(num_point):
    terms = []
    for i in range(num_point):
        for j in range(i + 1, num_point):
            terms.append((qml.PauliX(2 * i) + qml.PauliX(2 * i + 1)) @ (qml.PauliX(2 * j) + qml.PauliX(2 * j + 1)) + (qml.PauliY(2 * i) + qml.PauliY(2 * i + 1)) @ (qml.PauliY(2 * j) + qml.PauliY(2 * j + 1)) + (qml.PauliZ(2 * i) + qml.PauliZ(2 * i + 1)) @ (qml.PauliZ(2 * j) + qml.PauliZ(2 * j + 1)))
    
    # for i in range(num_point):
    #     for j in range(i + 1, num_point):
    #         for k in range(j + 1, num_point): 
    #             operator_i = (
    #                 qml.PauliX(2 * i) + qml.PauliX(2 * i + 1), 
    #                 qml.PauliY(2 * i) + qml.PauliY(2 * i + 1), 
    #                 qml.PauliZ(2 * i) + qml.PauliZ(2 * i + 1))
    #             operator_j = (
    #                 qml.PauliX(2 * j) + qml.PauliX(2 * j + 1),  
    #                 qml.PauliY(2 * j) + qml.PauliY(2 * j + 1), 
    #                 qml.PauliZ(2 * j) + qml.PauliZ(2 * j + 1))
    #             operator_k = (
    #                 qml.PauliX(2 * k) + qml.PauliX(2 * k + 1),  
    #                 qml.PauliY(2 * k) + qml.PauliY(2 * k + 1), 
    #                 qml.PauliZ(2 * k) + qml.PauliZ(2 * k + 1))
    #             terms.append(Spin_3_twirling.compute_i_j_k_term(operator_i, operator_j, operator_k))
    return terms

def prepare_init_state(num_qubit):
    for i in range(0, num_qubit, 2):
        create_singlet(i, i+1) 


def create_twirling_circuit(num_qubit, num_blocks_reupload, num_reupload, Theta, H):  

    def twirling_circuit(params, data_pt): 
        prepare_init_state(num_qubit)
        k = 0
        for i in range(num_reupload):
            data = data_pt[:,i,:,:]
            encode(data / Theta, num_qubit) 

            for l in range(num_blocks_reupload):
                for p in range(2, int(num_qubit / 2) + 1):
                    Spin_twirling(params["q"][k], params["q"][k + 1] , p, wires = range(num_qubit))
                    k += 2

        return [qml.expval(h) for h in H]

    return twirling_circuit

    
def train(gate_type, dataset, minibatch_size, Theta, epochs, key, init_scale, num_blocks_reupload, num_qubit, num_reupload, **adam_opt):  
    train_dataset_x = dataset['train_dataset_x']
    train_dataset_y = dataset['train_dataset_y']
    test_dataset_x = dataset['test_dataset_x']
    test_dataset_y = dataset['test_dataset_y']
    assert len(train_dataset_x) == len(train_dataset_y)
    assert len(train_dataset_x) % minibatch_size == 0
    batch_size = len(train_dataset_x)

    test_dataset_x = test_dataset_x.reshape(80, num_reupload, -1, 3)
    # point_sqr = jnp.power(test_dataset_x, 2)
    # norms = jnp.sqrt(jnp.sum(point_sqr, axis = -1))
    # test_dataset_x = test_dataset_x / norms.reshape(test_dataset_x.shape[0], num_reupload, num_qubit, 1)

    train_dataset_x = train_dataset_x.reshape(batch_size // minibatch_size, minibatch_size, num_reupload, -1, 3)
    # point_sqr = jnp.power(train_dataset_x, 2)
    # norms = jnp.sqrt(jnp.sum(point_sqr, axis = -1))
    # train_dataset_x = train_dataset_x / norms.reshape(train_dataset_x.shape[0], minibatch_size, num_reupload, num_qubit, 1)
    train_dataset_y = train_dataset_y.reshape(batch_size // minibatch_size, minibatch_size)

    ham = create_Hamiltonian(int(num_qubit/2))

    if gate_type == "u2":
        init_u2 = init_scale * math.pi/(2 * (int(num_qubit // 2) - 1) * (num_blocks_reupload * num_reupload))
        params_q = init_u2 * jax.random.uniform(key, (2 * (int(num_qubit // 2) - 1) * (num_blocks_reupload * num_reupload),))

        dummy_input = jnp.ones((1, math.comb(int(num_qubit / 2), 2)))  
        params_c = MyNN().init(key, dummy_input)

        params = {"q" : params_q, "c" : params_c}

        twirling_circuit = qml.QNode(create_twirling_circuit(num_qubit, num_blocks_reupload, num_reupload, Theta, ham), device = dev, interface='jax')  
        twirling_circuit = jax.jit(twirling_circuit)

        # drawer = qml.draw(u2_circuit, show_all_wires=True, decimals=None)
        # print(drawer(params, test_dataset_x))


        def loss_fn(params, mini_batch_x, mini_batch_y, l2):
            expval_ham = (jnp.array(twirling_circuit(params, mini_batch_x))).T
            logits = NN_circuit(expval_ham, params)
            mini_batch_y = mini_batch_y[:, None]
            loss = jnp.mean(optax.losses.sigmoid_binary_cross_entropy(logits, mini_batch_y))
            l2_penalty = l2 * jnp.sum(jnp.array([jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params)]))            
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
            epoch_grads = [] 
            
            for i in range(batch_size // minibatch_size):
                loss, grad = jax.value_and_grad(loss_fn, argnums=0)(params, train_dataset_x[i], train_dataset_y[i], l2)
                updates, opt_state = solver.update(grad, opt_state, params)
                params = optax.apply_updates(params, updates)
                train_loss += loss / (batch_size // minibatch_size)

            # recent_grads.extend(epoch_grads)
            # recent_grads = recent_grads[-10:] 
            
            # if len(recent_grads) >= 10:
            #     avg_grad = jnp.mean(jnp.array(recent_grads))
            #     max_grad = jnp.max(jnp.array(recent_grads))
                
            #     if avg_grad > 5:
            #         current_lr_scale = 0.5
            #         print(f"Epoch {epoch}: Barren plateau detected (avg grad: {avg_grad:.2e}), will amplify next epoch")
            #     else:
            #         current_lr_scale = 1.0
            
            # print(f"Epoch {epoch}: Loss = {train_loss:.4f}, LR Scale = {current_lr_scale}")          


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
        max_success = max(succed_lst)
        print(f"최대 테스트 정확도: {max_success:.4f}")
        plt.plot(test_loss_lst, label='Test Loss', marker='o', linestyle='-', color='b')
        plt.plot(train_loss_lst, label='Training Loss', marker='o', linestyle='-', color='r')
        plt.plot(succed_lst, label='Test Accuracy', marker='o', linestyle='-', color='y')
        plt.title('Loss & Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Loss & Test Accuracy')
        plt.legend()  
        plt.grid(True)
        plt.show()




num_qubit = 8
num_reupload = 1
gate_type = "u2"
test_learning_rate = 0.005
num_blocks_reupload = 8
init_scale = 0.08
dev = qml.device("default.qubit", wires = num_qubit)
# dataset = np.load(f'dataset_{num_qubit}_{num_reupload}.npz')
# dataset = np.load(f'dataset_{int(num_qubit/2)}_{num_reupload}.npz')
dataset = np.load(f'modelnet40_2classes_{int(num_qubit/2)}_{num_reupload}_fps_train960_test40_new.npz')
# dataset = np.load(f'modelnet40_2classes_{int(num_qubit/2)}_{num_reupload}_random_train960_test40_new.npz')

print(get_Theta(dataset))
Theta = 1.7
epochs = 100
l2 = 0.000001
key, key_r = jax.random.split(key)

def result(gate_type, test_learning_rate, num_blocks_reupload, init_scale):  
    print(f'qubits = {num_qubit}, gate_type = {gate_type}, test_learning_rate = {test_learning_rate}, num_blocks_reupload = {num_blocks_reupload}, init_scale= {init_scale}, num_reupload = {num_reupload}, Theta = {Theta}, data = {dataset}')
    train(gate_type, dataset, 32, Theta, epochs, key_r, init_scale, num_blocks_reupload, num_qubit, num_reupload, learning_rate = test_learning_rate)  

result(gate_type, test_learning_rate, num_blocks_reupload, init_scale)


# # test loss가 높은 이유 : sphere, torus 구분이 애매한 point에서 크게 틀림
# # memory : 8 qubits, 6 blocks, 2 reupload, learning rate = 0.005, init_scale = 0.05(0.02), l2 = 0.00001, theta = 10 -> dataset
# # memory : l2 = 0.000001, test_learning_rate = 0.01 num_blocks_reupload = 6 init_scale = 0.02 Theta = 2.5 -> modelnet
# # memory : qubits = 8, gate_type = u2, test_learning_rate = 0.005, num_blocks_reupload = 6, init_scale= 0.02, num_reupload = 2, Theta = 18 -> dataset
# # memory : qubits = 8, gate_type = u2, test_learning_rate = 0.005, num_blocks_reupload = 6, init_scale= 0.02, num_reupload = 2, Theta = 18 -> dataset
# # memory : qubits = 8, gate_type = u2, test_learning_rate = 0.005, num_blocks_reupload = 8, init_scale= 0.08, num_reupload = 1, Theta = 1.7 -> modelnet

# local Theta = 2.5랑 Theta = 1.44랑 비교해봐야함 -> 2.5가 나음
# 중심 바꾸기가 고점이 더 높은듯?
# colab num_reupload 8 -> 별로
# 애초에 noise에 너무 민감하네 0.02에서도 성능하락 명확
# 초반에 오히려 learning rate가 커짐

# Classical params = 16 × C(num_qubit, 2) + 1257
# Quantum params = 2 × num_blocks_reupload × num_reupload
# Total params = 2 × num_blocks_reupload × num_reupload + 16 × C(num_qubit, 2) + 1257
# 8,2,6에서 1729개