import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pennylane as qml
import jax
import jax.numpy as jnp
import optax
from gates_test import Spin_twirling, create_singlet
import math
import pennylane.numpy as pnp
from flax import linen as nn
import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group

scipy_rng = np.random.RandomState(1557)
# jax.config.update("jax_default_matmul_precision", "highest")
np.random.seed(1557)
jax.config.update("jax_enable_x64", True)
key = jax.random.key(1557)

def get_Theta(point):
    point = point['train_dataset_x']
    point = point - jnp.mean(point, axis = 0)
    point_sqr = jnp.power(point, 2)
    norms = jnp.sqrt(jnp.sum(point_sqr, axis = -1))
    Theta = jnp.max(norms) * 1.2
    return Theta

def random_3d_rotation():
    return special_ortho_group.rvs(3, random_state=scipy_rng)

def apply_3d_rotation(points):
    rotation_matrix = random_3d_rotation()
    rotation_matrix_jax = jnp.array(rotation_matrix)
    return jnp.dot(points, rotation_matrix_jax.T)

def add_jitter(points, key, sigma):
    noise = jax.random.normal(key, points.shape) * sigma
    return points + noise

def apply_permutation(points):
    num_points = points.shape[0]
    perm_indices = scipy_rng.permutation(num_points)
    return points[perm_indices]

def apply_data_augmentation(points, key, is_training=True):
    if not is_training:
        return points
    
    key1, key2 = jax.random.split(key)
    augmented_points = add_jitter(points, key2, sigma)
    augmented_points = apply_3d_rotation(augmented_points)
    augmented_points = apply_permutation(augmented_points)

    return augmented_points

def augment_batch(batch_points, key, is_training=True):
    if not is_training:
        return batch_points
    
    batch_size = batch_points.shape[0]
    keys = jax.random.split(key, batch_size)
    
    augment_fn = jax.vmap(lambda points, k: apply_data_augmentation(points, k, is_training))
    return augment_fn(batch_points, keys)

class MyNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.expand_dims(x, axis=-1)
        x = nn.Dense(features=6)(x) 
        x = nn.tanh(x)
        x = nn.Dense(features=6)(x) 
        x = nn.tanh(x)

        # 기존 pooling (모두 permutation invariant)
        mean_pool = jnp.mean(x, axis=1)
        max_pool = jnp.max(x, axis=1) 
        min_pool = jnp.min(x, axis=1)
        std_pool = jnp.std(x, axis=1)
        sum_pool = jnp.sum(x, axis=1)
        var_pool = jnp.var(x, axis=1)
       
        x = jnp.concatenate([
            mean_pool, max_pool, min_pool, sum_pool, std_pool, var_pool               
        ], axis=-1)
        
        x = nn.Dense(features=36)(x) 
        x = nn.tanh(x)
        x = nn.Dense(features=36)(x) 
        x = nn.tanh(x)
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


def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names or range(len(cm)),
                yticklabels=class_names or range(len(cm)))
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def analyze_gradient_norms(grad):
    q_individual_grads = grad["q"]
    q_grad_norm = jnp.linalg.norm(grad["q"])
    c_grad_leaves = jax.tree_leaves(grad["c"])
    c_grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in c_grad_leaves))
    
    total_grad_norm = jnp.sqrt(jnp.sum(jnp.square(grad["q"])) + sum(jnp.sum(jnp.square(g)) for g in c_grad_leaves))
    
    return q_individual_grads, q_grad_norm, c_grad_norm, total_grad_norm


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
                for p in range(2, int(num_qubit / 2) + 1):
                    Spin_twirling(params["q"][k], params["q"][k + 1] , p, wires = range(num_qubit))
                    k += 2

        return [qml.expval(h) for h in H]

    return twirling_circuit

    
def train(gate_type, dataset, minibatch_size, Theta, epochs, key, init_scale, num_blocks_reupload, num_qubit, num_reupload, use_augmentation, **adam_opt):  
    train_dataset_x = dataset['train_dataset_x']
    train_dataset_y = dataset['train_dataset_y']
    test_dataset_x = dataset['test_dataset_x']
    test_dataset_y = dataset['test_dataset_y']
    assert len(train_dataset_x) == len(train_dataset_y)
    assert len(train_dataset_x) % minibatch_size == 0
    batch_size = len(train_dataset_x)

    test_dataset_x = test_dataset_x.reshape(200, num_reupload, -1, 3)  

    ham = create_Hamiltonian(int(num_qubit/2))

    if gate_type == "u2":
        # init_u2 = init_scale * math.pi/(2 * (num_blocks_reupload * num_reupload))
        # params_q = init_u2 * jax.random.uniform(key, (2 * (num_blocks_reupload * num_reupload),))
        init_u2 = init_scale * math.pi/(2 * (int(num_qubit/2) - 1) * (num_blocks_reupload * num_reupload))
        params_q = init_u2 * jax.random.uniform(key, (2 * (int(num_qubit/2) - 1) * (num_blocks_reupload * num_reupload),))

        dummy_input = jnp.ones((1, math.comb(int(num_qubit / 2), 2)))  
        params_c = MyNN().init(key, dummy_input)

        params = {"q" : params_q, "c" : params_c}

        twirling_circuit = qml.QNode(create_twirling_circuit(num_qubit, num_blocks_reupload, num_reupload, Theta, ham), device = dev, interface='jax', diff_method='adjoint')  

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
        
        q_grad_norms = []
        c_grad_norms = []
        total_grad_norms = []

        for epoch in range(epochs):
            train_loss = 0
            epoch_q_grad_norms = []
            epoch_c_grad_norms = []
            epoch_total_grad_norms = []
            
            print(f"epoch {epoch}")
            
            current_train_x = train_dataset_x.copy()
            if use_augmentation:
                epoch_key = jax.random.fold_in(key, epoch)
                current_train_x = augment_batch(current_train_x, epoch_key, is_training=True)
            
            current_train_x = current_train_x.reshape(batch_size // minibatch_size, minibatch_size, num_reupload, -1, 3)
            train_dataset_y_batched = train_dataset_y.reshape(batch_size // minibatch_size, minibatch_size)
            
            for i in range(batch_size // minibatch_size):
                loss, grad = jax.value_and_grad(loss_fn, argnums=0)(params, current_train_x[i], train_dataset_y_batched[i], l2)
                
                q_individual_grads, q_grad_norm, c_grad_norm, total_grad_norm = analyze_gradient_norms(grad)
                epoch_q_grad_norms.append(q_grad_norm)
                epoch_c_grad_norms.append(c_grad_norm)
                epoch_total_grad_norms.append(total_grad_norm)

                # print(f"Step {i:2d} - Q: {q_grad_norm:.1e}, C: {c_grad_norm:.1e}, Total: {total_grad_norm:.1e}")
                # print("    Individual Quantum Gradients:", end=" ")
                # for j, q_grad in enumerate(q_individual_grads):
                #     print(f"Q{j}:{q_grad:.1e}", end=" ")
                # print()
                # print("    Current Quantum Parameters:", end=" ")
                # for j, q_param in enumerate(params["q"]):
                #     print(f"Q{j}:{q_param:.4f}", end=" ")
                # print()

                
                # if total_grad_norm < 1e-5:
                #     print(f"Step {i:2d} - Q: {q_grad_norm:.1e}, C: {c_grad_norm:.1e}, Total: {total_grad_norm:.1e}")
                #     print("    ⚠️  Very small gradients - possible Barren Plateau!")
                #     print("    Individual Quantum Gradients:", end=" ")
                #     for j, q_grad in enumerate(q_individual_grads):
                #         print(f"Q{j}:{q_grad:.1e}", end=" ")
                #     print()
                #     print("    Current Quantum Parameters:", end=" ")
                #     for j, q_param in enumerate(params["q"]):
                #         print(f"Q{j}:{q_param:.4f}", end=" ")
                #     print()
                    
                # elif total_grad_norm > 2.5:
                #     print(f"Step {i:2d} - Q: {q_grad_norm:.1e}, C: {c_grad_norm:.1e}, Total: {total_grad_norm:.1e}")
                #     print("    ⚠️  Large gradients - possible exploding gradients!")
                #     print("    Individual Quantum Gradients:", end=" ")
                #     for j, q_grad in enumerate(q_individual_grads):
                #         print(f"Q{j}:{q_grad:.1e}", end=" ")
                #     print()
                #     print("    Current Quantum Parameters:", end=" ")
                #     for j, q_param in enumerate(params["q"]):
                #         print(f"Q{j}:{q_param:.4f}", end=" ")
                #     print()
                
                updates, opt_state = solver.update(grad, opt_state, params)
                params = optax.apply_updates(params, updates)
                train_loss += loss / (batch_size // minibatch_size)

            avg_q_grad_norm = jnp.mean(jnp.array(epoch_q_grad_norms))
            avg_c_grad_norm = jnp.mean(jnp.array(epoch_c_grad_norms))
            avg_total_grad_norm = jnp.mean(jnp.array(epoch_total_grad_norms))
            
            q_grad_norms.append(avg_q_grad_norm)
            c_grad_norms.append(avg_c_grad_norm)
            total_grad_norms.append(avg_total_grad_norm)

            train_loss_lst.append(train_loss)
            print(f"\nTrain Loss: {train_loss}") 
            
            test_loss = loss_fn(params, test_dataset_x, test_dataset_y, l2) 
            expval_ham = (jnp.array(twirling_circuit(params, test_dataset_x))).T
            logits = NN_circuit(expval_ham, params)

            predictions = jnp.argmax(logits, axis=-1)
            succed = jnp.mean(predictions == test_dataset_y.squeeze())
            class_accuracies = calculate_class_accuracies(test_dataset_y, predictions, num_classes)
            
            test_loss_lst.append(test_loss)
            succed_lst.append(succed)
            class_accuracies_history.append(class_accuracies)
            
            print(f"Test Loss: {test_loss}")
            print(f"Test Accuracy: {succed}")
            print(f"Epoch Avg Gradients - Q: {avg_q_grad_norm:.1e}, C: {avg_c_grad_norm:.1e}, Total: {avg_total_grad_norm:.1e}")
            
            for i, acc in enumerate(class_accuracies):
                print(f"Class {i}: {acc}")
            print("-" * 50)
        
        max_success = max(succed_lst)
        print(f"최대 테스트 정확도: {max_success}")
        final_expval_ham = (jnp.array(twirling_circuit(params, test_dataset_x))).T
        final_logits = NN_circuit(final_expval_ham, params)
        final_predictions = jnp.argmax(final_logits, axis=-1)
        
        final_cm, final_class_acc, final_overall_acc = calculate_final_metrics(
            test_dataset_y, final_predictions, num_classes)
        
        print("\n=== Final Results ===")
        print(f"Final Overall Accuracy: {final_overall_acc:.4f}")
        print("Final Class Accuracies:")
        for i, acc in enumerate(final_class_acc):
            print(f"  Class {i}: {acc}")
        
        print("\n=== Confusion Matrix ===")
        print(final_cm)
        
        plot_confusion_matrix(final_cm, title='Final Confusion Matrix')
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(test_loss_lst, label='Test Loss', marker='o', linestyle='-', color='b')
        plt.plot(train_loss_lst, label='Training Loss', marker='o', linestyle='-', color='r')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()  
        plt.grid(True)
        
        plt.subplot(2, 3, 2)
        plt.plot(succed_lst, label='Test Accuracy', marker='o', linestyle='-', color='y')
        plt.title('Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy')
        plt.legend()  
        plt.grid(True)
        
        plt.subplot(2, 3, 3)
        class_accuracies_array = np.array(class_accuracies_history)
        for i in range(num_classes):
            plt.plot(class_accuracies_array[:, i], label=f'Class {i}', marker='o', linestyle='-')
        plt.title('Class-wise Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 4)
        plt.bar(range(num_classes), final_class_acc, color='skyblue', edgecolor='black')
        plt.title('Final Class-wise Accuracy')
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.xticks(range(num_classes))
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 5)
        plt.semilogy(q_grad_norms, label='Quantum', marker='o', linestyle='-', color='red')
        plt.semilogy(c_grad_norms, label='Classical', marker='s', linestyle='-', color='blue')
        plt.semilogy(total_grad_norms, label='Total', marker='^', linestyle='-', color='green')
        plt.title('Gradient Norms (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 6)
        ratio = np.array(q_grad_norms) / np.array(c_grad_norms)
        plt.plot(ratio, marker='o', linestyle='-', color='purple')
        plt.title('Quantum/Classical Gradient Ratio')
        plt.xlabel('Epoch')
        plt.ylabel('Q_grad / C_grad')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("\n=== Classification Report ===")
        print(classification_report(np.array(test_dataset_y).flatten(), 
                                    np.array(final_predictions).flatten(),
                                    target_names=[f'Class {i}' for i in range(num_classes)]))
        
        print("\n=== Gradient Norm Summary ===")
        print(f"Final Quantum Gradient Norm: {q_grad_norms[-1]:.1e}")
        print(f"Final Classical Gradient Norm: {c_grad_norms[-1]:.1e}")
        print(f"Final Total Gradient Norm: {total_grad_norms[-1]:.1e}")


num_qubit = 8
num_reupload = 1
gate_type = "u2"
test_learning_rate = 0.001
num_blocks_reupload = 12
init_scale = 0.02
dev = qml.device("lightning.qubit", wires = num_qubit)
point_class_train = 960
point_class_test = 40
sigma = 0.02
# sigma = 0

Theta = 1.7                                     
use_augmentation = True
# dataset = np.load(f'modelnet40_5classes_{int(num_qubit/2)}_{num_reupload}_fps_train{point_class_train}_test{point_class_test}.npz')
# dataset = np.load(f'modelnet40_5classes_{int(num_qubit /2)}_{num_reupload}_fps_train{point_class_train}_test{point_class_test}_new.npz')
# dataset = np.load(f'modelnet40_10classes_{int(num_qubit/2)}_{num_reupload}_fps_train{point_class_train}_test{point_class_test}_new.npz')
# dataset = np.load(f'modelnet40_10classes_{int(num_qubit/2)}_{num_reupload}_random_train{point_class_train}_test{point_class_test}_new.npz')
# dataset = np.load(f'dataset_{int(num_qubit/2)}_{num_reupload}.npz')
# dataset = np.load(f'modelnet40_2classes_{int(num_qubit/2)}_{num_reupload}_fps_train960_test40_new.npz')
# dataset = np.load(f'modelnet40_bowl_cup_{int(num_qubit/2)}_{num_reupload}_fps_train960_test40_new.npz')
dataset = np.load(f'modelnet40_5classes_{int(num_qubit/2)}_{num_reupload}_fps_train960_test40_new.npz')
# dataset = np.load(f'modelnet40_4classes_{int(num_qubit/2)}_{num_reupload}_fps_train960_test40_new.npz')

print(get_Theta(dataset))
test_dataset_y = dataset['test_dataset_y']
num_classes = len(np.unique(test_dataset_y))
epochs = 800
l2 = 0.0000001
key, key_r = jax.random.split(key)

def result(gate_type, test_learning_rate, num_blocks_reupload, init_scale, use_augmentation):  
    train(gate_type, dataset, 32, Theta, epochs, key_r, init_scale, num_blocks_reupload, num_qubit, num_reupload, use_augmentation, learning_rate = test_learning_rate) 

result(gate_type, test_learning_rate, num_blocks_reupload, init_scale, use_augmentation)


# # memory : qubits = 8, gate_type = u2, test_learning_rate = 0.001, num_blocks_reupload = 12, init_scale= 0.02, num_reupload = 1, Theta = 1.7, l2 = 1e-6, layer_1 = 64, layer_2 = 64, attention = 32-> modelnet(5classes) -> 75%(attention 추가) (var추가) (800epoch)


# # memory : qubits = 8, gate_type = u2, test_learning_rate = 0.002, num_blocks_reupload = 12, init_scale= 0.02, num_reupload = 1, Theta = 1.7, l2 = 1e-6, layer_1 = 64, layer_2 = 64, attention = 32-> modelnet(5classes) -> 71%(attention 추가) (var추가) - colab
# # memory : qubits = 8, gate_type = u2, test_learning_rate = 0.002, num_blocks_reupload = 12, init_scale= 0.02, num_reupload = 1, Theta = 1.7, l2 = 1e-6, layer_1 = 64, layer_2 = 64, attention = 32-> modelnet(5classes) -> 72.5%(attention 추가) - m2
# # memory : qubits = 8, gate_type = u2, test_learning_rate = 0.001, num_blocks_reupload = 12, init_scale= 0.02, num_reupload = 1, Theta = 1.7, l2 = 1e-6, layer_1 = 64, layer_2 = 64, attention = 32-> modelnet(5classes) -> 74.5%(attention 추가) (var추가) - colab
# # memory : qubits = 8, gate_type = u2, test_learning_rate = 0.001, num_blocks_reupload = 12, init_scale= 0.02, num_reupload = 1, Theta = 1.7, l2 = 1e-6, layer_1 = 64, layer_2 = 64, attention = 32-> modelnet(5classes) -> 74%(attention 추가) - m2
# # memory : qubits = 8, gate_type = u2, test_learning_rate = 0.001, num_blocks_reupload = 12, init_scale= 0.02, num_reupload = 1, Theta = 1.7, l2 = 1e-6, layer_1 = 64, layer_2 = 64, attention = 32-> modelnet(5classes) -> 72%(attention 추가) (var추가) - m4 (no conda)
# # memory : qubits = 8, gate_type = u2, test_learning_rate = 0.001, num_blocks_reupload = 12, init_scale= 0.02, num_reupload = 1, Theta = 1.7, l2 = 1e-6, layer_1 = 64, layer_2 = 64, attention = 32-> modelnet(5classes) -> 75%(attention 추가) (var추가) -m2
# # memory : qubits = 8, gate_type = u2, test_learning_rate = 0.001, num_blocks_reupload = 12, init_scale= 0.02, num_reupload = 1, Theta = 1.7, l2 = 1e-6, layer_1 = 64, layer_2 = 64-> modelnet(5classes) -> 73.5% (var추가) -m4 

# # memory : qubits = 8, gate_type = u2, test_learning_rate = 0.001, num_blocks_reupload = 10, init_scale= 0.02, num_reupload = 1, Theta = 1.7, l2 = 1e-6, layer_1 = 6, layer_2 = 6, first = 3 * no relu, second = 3 * relu-> modelnet(5classes) -> 70% (var추가) -m4
# # memory : qubits = 8, gate_type = u2, test_learning_rate = 0.001, num_blocks_reupload = 10, init_scale= 0.02, num_reupload = 1, Theta = 1.7, l2 = 1e-6, layer_1 = 6, layer_2 = 6, first = 3 * tanh, second = 3 * tanh-> modelnet(5classes) -> 70.5% (var추가) -m4
# # memory : qubits = 8, gate_type = u2, test_learning_rate = 0.001, num_blocks_reupload = 10, init_scale= 0.02, num_reupload = 1, Theta = 1.7, l2 = 1e-6, layer_1 = 6, layer_2 = 6, first = 2 * tanh, second = 2 * tanh-> modelnet(5classes) -> 71.5% (var추가) -m4
# # memory : qubits = 8, gate_type = u2, test_learning_rate = 0.001, num_blocks_reupload = 12, init_scale= 0.02, num_reupload = 1, Theta = 1.7, l2 = 1e-6, layer_1 = 6, layer_2 = 6, first = 2 * tanh, second = 2 * tanh-> modelnet(5classes) -> 71.5% (var추가) -m4
# # memory : qubits = 8, gate_type = u2, test_learning_rate = 0.001, num_blocks_reupload = 12, init_scale= 0.02, num_reupload = 1, Theta = 1.7, l2 = 1e-6, layer_1 = 64, layer_2 = 64, first = 2 * tanh, second = 2 * tanh-> modelnet(5classes) -> 76% (var추가) -m4
# # memory : qubits = 8, gate_type = u2, test_learning_rate = 0.001, num_blocks_reupload = 12, init_scale= 0.02, num_reupload = 1, Theta = 1.7, l2 = 1e-6, layer_1 = 6, layer_2 = 36, first = 2 * tanh, second = 2 * tanh-> modelnet(5classes) -> 73% (var추가) -m4

# # memory : qubits = 10, gate_type = u2, test_learning_rate = 0.001, num_blocks_reupload = 12, init_scale= 0.02, num_reupload = 1, Theta = 1.7, l2 = 1e-6, layer_1 = 10, layer_2 = 10, first = 2 * tanh, second = 2 * tanh-> modelnet(5classes) -> 76% (var추가) -m4
# # memory : qubits = 10, gate_type = u2, test_learning_rate = 0.001, num_blocks_reupload = 12, init_scale= 0.02, num_reupload = 1, Theta = 1.7, l2 = 1e-6, layer_1 = 6, layer_2 = 36, first = 2 * tanh, second = 2 * tanh-> modelnet(5classes) -> 76.5% (var추가) -m4

