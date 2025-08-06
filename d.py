import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pennylane as qml
import jax
import jax.numpy as jnp
import optax
from gates import create_singlet
from gates_twirling import Spin_2_twirling, Spin_3_twirling
import math
import pennylane.numpy as pnp
from flax import linen as nn
import gc


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
        x = nn.Dense(features=num_classes)(x)  
        return x

def NN_circuit(dataset, params):
    return MyNN().apply(params["c"], dataset)


def calculate_class_accuracies(y_true, y_pred, num_classes):
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()
    
    class_accuracies = []
    for i in range(num_classes):
        mask = (y_true_np == i)
        if np.sum(mask) > 0:
            class_acc = np.sum((y_pred_np[mask] == i)) / np.sum(mask)
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    return class_accuracies


def calculate_final_metrics(y_true, y_pred, num_classes):
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()
    
    cm = confusion_matrix(y_true_np, y_pred_np, labels=range(num_classes))
    
    class_accuracies = []
    for i in range(num_classes):
        if np.sum(cm[i, :]) > 0:
            class_acc = cm[i, i] / np.sum(cm[i, :])
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)

    overall_accuracy = np.trace(cm) / np.sum(cm)
    
    return cm, class_accuracies, overall_accuracy


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
        ]).transpose(2,0,1)
        
        qml.QubitUnitary(matrix_i, wires=2 * i)

def create_Hamiltonian(num_point):
    terms = []
    for i in range(num_point - 1):
        for j in range(i + 1, num_point):
            terms.append((qml.PauliX(2 * i) + qml.PauliX(2 * i + 1)) @ (qml.PauliX(2 * j) + qml.PauliX(2 * j + 1)) + (qml.PauliY(2 * i) + qml.PauliY(2 * i + 1)) @ (qml.PauliY(2 * j) + qml.PauliY(2 * j + 1)) + (qml.PauliZ(2 * i) + qml.PauliZ(2 * i + 1)) @ (qml.PauliZ(2 * j) + qml.PauliZ(2 * j + 1)))
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
                Spin_2_twirling(params["q"][k], wires = range(num_qubit))
                Spin_3_twirling(params["q"][k + 1], wires = range(num_qubit))
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

    test_dataset_x = test_dataset_x.reshape(200, num_reupload, -1, 3)  
    train_dataset_x = train_dataset_x.reshape(batch_size // minibatch_size, minibatch_size, num_reupload, -1, 3)
    train_dataset_y = train_dataset_y.reshape(batch_size // minibatch_size, minibatch_size)

    ham = create_Hamiltonian(int(num_qubit/2))

    if gate_type == "u2":
        init_u2 = init_scale * math.pi/(2 * (num_blocks_reupload * num_reupload))
        params_q = init_u2 * jax.random.uniform(key, (2 * (num_blocks_reupload * num_reupload),))

        dummy_input = jnp.ones((1, math.comb(int(num_qubit / 2), 2)))  
        params_c = MyNN().init(key, dummy_input)

        params = {"q" : params_q, "c" : params_c}

        twirling_circuit = qml.QNode(create_twirling_circuit(num_qubit, num_blocks_reupload, num_reupload, Theta, ham), device = dev, interface='jax')  
        twirling_circuit = jax.jit(twirling_circuit)

        def loss_fn(params, mini_batch_x, mini_batch_y, l2):
            expval_ham = (jnp.array(twirling_circuit(params, mini_batch_x))).T
            logits = NN_circuit(expval_ham, params)
            loss = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(logits, mini_batch_y))
            l2_penalty = l2 * jnp.sum(jnp.array([jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params)]))            
            return (loss + l2_penalty)

        solver = optax.adam(**adam_opt)
        opt_state = solver.init(params)
        
        train_loss_lst = []
        test_loss_lst = []
        succed_lst = []
        class_accuracies_history = []

        for epoch in range(epochs):
            train_loss = 0
            
            if epoch % 2 == 0:
                gc.collect()
                jax.clear_caches()
            
            for i in range(batch_size // minibatch_size):
                loss, grad = jax.value_and_grad(loss_fn, argnums=0)(params, train_dataset_x[i], train_dataset_y[i], l2)
                updates, opt_state = solver.update(grad, opt_state, params)
                params = optax.apply_updates(params, updates)
                train_loss += loss / (batch_size // minibatch_size)

            train_loss_lst.append(train_loss)
            
            test_loss = loss_fn(params, test_dataset_x, test_dataset_y, l2) 
            expval_ham = (jnp.array(twirling_circuit(params, test_dataset_x))).T
            logits = NN_circuit(expval_ham, params)

            predictions = jnp.argmax(logits, axis=-1)
            succed = jnp.mean(predictions == test_dataset_y.squeeze())
            class_accuracies = calculate_class_accuracies(test_dataset_y, predictions, num_classes)
            
            test_loss_lst.append(test_loss)
            succed_lst.append(succed)
            class_accuracies_history.append(class_accuracies)
            
            print(f"Epoch {epoch}")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Test Loss: {test_loss:.6f}")
            print(f"Test Accuracy: {succed:.4f}")
            
            for i, acc in enumerate(class_accuracies):
                print(f"Class {i} Accuracy: {acc:.4f}")
            print("-" * 40)
        
        max_success = max(succed_lst)
        print(f"\n=== FINAL RESULTS ===")
        print(f"Maximum Test Accuracy: {max_success:.4f}")
        
        final_expval_ham = (jnp.array(twirling_circuit(params, test_dataset_x))).T
        final_logits = NN_circuit(final_expval_ham, params)
        final_predictions = jnp.argmax(final_logits, axis=-1)
        
        final_cm, final_class_acc, final_overall_acc = calculate_final_metrics(
            test_dataset_y, final_predictions, num_classes)
        
        print(f"Final Overall Accuracy: {final_overall_acc:.4f}")
        print("Final Class Accuracies:")
        for i, acc in enumerate(final_class_acc):
            print(f"  Class {i}: {acc:.4f}")


num_qubit = 16
num_reupload = 1
gate_type = "u2"
test_learning_rate = 0.003
num_blocks_reupload = 7
init_scale = 0.01
dev = qml.device("default.qubit", wires = num_qubit)
point_class_train = 960
point_class_test = 40

Theta = 1.7
dataset = np.load(f'modelnet40_5classes_{int(num_qubit/2)}_{num_reupload}_fps_train{point_class_train}_test{point_class_test}_new.npz')

print("Calculated Theta:", get_Theta(dataset))
test_dataset_y = dataset['test_dataset_y']
num_classes = len(np.unique(test_dataset_y))
epochs = 5
l2 = 0.000001
key, key_r = jax.random.split(key)

def result(gate_type, test_learning_rate, num_blocks_reupload, init_scale):  
    print(f'qubits = {num_qubit}, gate_type = {gate_type}, test_learning_rate = {test_learning_rate}, num_blocks_reupload = {num_blocks_reupload}, init_scale= {init_scale}, num_reupload = {num_reupload}, Theta = {Theta}, l2 = {l2}')
    train(gate_type, dataset, 4, Theta, epochs, key_r, init_scale, num_blocks_reupload, num_qubit, num_reupload, learning_rate = test_learning_rate) 

result(gate_type, test_learning_rate, num_blocks_reupload, init_scale)