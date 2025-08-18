import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pennylane as qml
import jax
import jax.numpy as jnp
import optax
from gates import create_singlet
from gates_twirling_perm import Spin_twirling
# from gates_twirling import Spin_2_twirling, Spin_3_twirling
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

def random_rotation_z_axis(points, key):
    theta = jax.random.uniform(key, minval=0, maxval=2 * jnp.pi)
    rotation_matrix = jnp.array([
        [jnp.cos(theta), -jnp.sin(theta), 0],
        [jnp.sin(theta), jnp.cos(theta), 0],
        [0, 0, 1]
    ])
    return jnp.dot(points, rotation_matrix.T)

def add_jitter(points, key, sigma):
    noise = jax.random.normal(key, points.shape) * sigma
    return points + noise

def apply_data_augmentation(points, key, is_training=True):
    if not is_training:
        return points
    
    key1, key2 = jax.random.split(key)
    augmented_points = random_rotation_z_axis(points, key1)
    augmented_points = add_jitter(augmented_points, key2, sigma)
    
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
        x = nn.Dense(features=30)(x)  
        x = nn.relu(x)
        x = nn.Dense(features=30)(x) 

        # 기존 pooling (모두 permutation invariant)
        mean_pool = jnp.mean(x, axis=1)
        max_pool = jnp.max(x, axis=1) 
        min_pool = jnp.min(x, axis=1)
        std_pool = jnp.std(x, axis=1)
        sum_pool = jnp.sum(x, axis=1)

        # 기본 통계 (모두 permutation invariant)
        median_pool = jnp.median(x, axis=1)
        var_pool = jnp.var(x, axis=1)
        range_pool = jnp.max(x, axis=1) - jnp.min(x, axis=1)

        # Percentile 기반 (모두 permutation invariant)
        p10_pool = jnp.percentile(x, 10, axis=1)
        p25_pool = jnp.percentile(x, 25, axis=1)
        p75_pool = jnp.percentile(x, 75, axis=1)
        p90_pool = jnp.percentile(x, 90, axis=1)

        # 절댓값 기반 (모두 permutation invariant)
        abs_mean_pool = jnp.mean(jnp.abs(x), axis=1)
        abs_max_pool = jnp.max(jnp.abs(x), axis=1)
        abs_sum_pool = jnp.sum(jnp.abs(x), axis=1)

        # 제곱 기반 (모두 permutation invariant)
        squared_mean_pool = jnp.mean(jnp.square(x), axis=1)
        l2_norm_pool = jnp.sqrt(jnp.sum(jnp.square(x), axis=1))

        # 고차 모멘트 (모두 permutation invariant)
        def compute_skewness(data):
            mean_x = jnp.mean(data, axis=1, keepdims=True)
            std_x = jnp.std(data, axis=1, keepdims=True)
            skew = jnp.mean(((data - mean_x) / (std_x + 1e-8)) ** 3, axis=1)
            return skew

        def compute_kurtosis(data):
            mean_x = jnp.mean(data, axis=1, keepdims=True)
            std_x = jnp.std(data, axis=1, keepdims=True)
            kurt = jnp.mean(((data - mean_x) / (std_x + 1e-8)) ** 4, axis=1) - 3
            return kurt

        def interaction_dominance(x):
            """가장 강한 상호작용이 얼마나 지배적인가 (permutation invariant)"""
            x_abs = jnp.abs(x)
            max_interaction = jnp.max(x_abs, axis=1)
            mean_interaction = jnp.mean(x_abs, axis=1)
            dominance = max_interaction / (mean_interaction + 1e-8)
            return dominance
        
        def ferro_antiferro_balance(x):
            """Ferromagnetic vs Antiferromagnetic 상호작용 균형 (permutation invariant)"""
            positive_strength = jnp.sum(jnp.where(x > 0, x, 0), axis=1)
            negative_strength = jnp.sum(jnp.where(x < 0, -x, 0), axis=1)
            total_strength = positive_strength + negative_strength
            balance = jnp.abs(positive_strength - negative_strength) / (total_strength + 1e-8)
            return balance
        
        def interaction_entropy(x):
            """상호작용의 정보 엔트로피 (무질서도) (permutation invariant)"""
            x_abs = jnp.abs(x) + 1e-8
            x_norm = x_abs / jnp.sum(x_abs, axis=1, keepdims=True)
            entropy = -jnp.sum(x_norm * jnp.log(x_norm + 1e-8), axis=1)
            return entropy

        def interaction_hierarchy(x):
            """상호작용의 계층적 구조 강도 (permutation invariant)"""
            x_sorted = jnp.sort(jnp.abs(x), axis=1)[:, ::-1]  # 내림차순
            ratios = x_sorted[:, :-1] / (x_sorted[:, 1:] + 1e-8)
            hierarchy_strength = jnp.mean(ratios, axis=1)
            return hierarchy_strength

        def effective_interaction_count(x):
            """실제로 의미있는 상호작용 개수 (permutation invariant)"""
            x_abs = jnp.abs(x)
            threshold = 0.1 * jnp.max(x_abs, axis=1, keepdims=True)
            effective_count = jnp.sum(x_abs > threshold, axis=1)
            return effective_count.astype(float)
        
        def interaction_coherence(x):
            """모든 상호작용이 같은 방향성을 갖는가 (permutation invariant)"""
            signs = jnp.sign(x)
            positive_ratio = jnp.mean(signs > 0, axis=1)
            negative_ratio = jnp.mean(signs < 0, axis=1)
            coherence = jnp.maximum(positive_ratio, negative_ratio)
            return coherence

        def participation_ratio(x):
            """얼마나 많은 mode가 활성화되어 있는가 (permutation invariant)"""
            x_squared = jnp.square(x)
            sum_squared = jnp.sum(x_squared, axis=1, keepdims=True)
            normalized = x_squared / (sum_squared + 1e-8)
            pr = 1.0 / jnp.sum(jnp.square(normalized), axis=1)
            return pr

        def spectral_gap(x):
            """가장 큰 두 상호작용 간의 간격 (permutation invariant)"""
            x_sorted = jnp.sort(jnp.abs(x), axis=1)[:, ::-1]
            gap = x_sorted[:, 0] - x_sorted[:, 1]
            return gap

        def correlation_length_proxy(x):
            """Spin Correlation Length (modified to be permutation invariant)"""
            # Use variance-based correlation measure instead of spatial correlation
            x_mean = jnp.mean(x, axis=1, keepdims=True)
            x_centered = x - x_mean
            correlation_strength = jnp.var(x_centered, axis=1)
            correlation_length = 1.0 / (jnp.abs(jnp.log(correlation_strength + 1e-8)) + 1e-8)
            return correlation_length

        def frustration_curvature_proxy(x):
            """Magnetic Frustration Index (permutation invariant)"""
            positive_interactions = jnp.sum(jnp.where(x > 0, x, 0), axis=1)
            negative_interactions = jnp.sum(jnp.where(x < 0, -x, 0), axis=1)
            total = positive_interactions + negative_interactions
            
            # Frustration peaks when positive and negative are balanced
            frustration = 4 * positive_interactions * negative_interactions / ((total**2) + 1e-8)
            return frustration

        def susceptibility_proxy(x):
            """Critical Susceptibility (permutation invariant)"""
            base_susceptibility = jnp.var(x, axis=1)
            return base_susceptibility

        def topological_complexity(x):
            """Topological complexity (modified to be permutation invariant)"""
            # Use magnitude and phase diversity instead of winding
            x_complex = x + 1j * jnp.roll(x, 1, axis=1)
            magnitude_var = jnp.var(jnp.abs(x_complex), axis=1)
            phase_var = jnp.var(jnp.angle(x_complex), axis=1)
            complexity = magnitude_var + phase_var
            return complexity
        
        skewness_pool = compute_skewness(x)
        kurtosis_pool = compute_kurtosis(x)
        dominance_pool = interaction_dominance(x)
        balance_pool = ferro_antiferro_balance(x)
        entropy_pool = interaction_entropy(x)
        hierarchy_pool = interaction_hierarchy(x)
        effective_count_pool = effective_interaction_count(x)
        coherence_pool = interaction_coherence(x)
        participation_pool = participation_ratio(x)
        spectral_gap_pool = spectral_gap(x)

        correlation_length_pool = correlation_length_proxy(x)
        frustration_pool = frustration_curvature_proxy(x)
        susceptibility_pool = susceptibility_proxy(x)
        topological_pool = topological_complexity(x)

        # Attention pooling (permutation invariant)
        attention_features = nn.Dense(32)(x)
        attention_features = nn.relu(attention_features)
        attention_logits = nn.Dense(1)(attention_features)
        attention_weights = nn.softmax(attention_logits, axis=1)
        attention_pool = jnp.sum(x * attention_weights, axis=1)

        # Feature attention
        pooling_stack = jnp.stack([
            mean_pool, max_pool, min_pool, std_pool, sum_pool, attention_pool
        ], axis=1)
        pooling_context = jnp.mean(pooling_stack, axis=1)  
        pooling_attention_features = nn.Dense(32)(pooling_context)
        pooling_attention_features = nn.relu(pooling_attention_features)
        pooling_attention_logits = nn.Dense(6)(pooling_attention_features)  # 6개 pooling 방법에 대한 가중치
        pooling_attention_weights = nn.softmax(pooling_attention_logits, axis=-1)  # (batch_size, 6)
        pooling_weights_expanded = jnp.expand_dims(pooling_attention_weights, axis=-1)  # (batch_size, 6, 1)
        attention_pool_2 = jnp.sum(pooling_stack * pooling_weights_expanded, axis=1)

        # 모든 pooling 결합 
        x = jnp.concatenate([
            mean_pool, max_pool, min_pool, std_pool, sum_pool, attention_pool
            # attention_pool_2,
            # median_pool, var_pool, range_pool
            # p10_pool, p90_pool,
            # p25_pool, p75_pool, 
            # abs_mean_pool, abs_max_pool, abs_sum_pool, 
            # squared_mean_pool, l2_norm_pool,
            # skewness_pool, kurtosis_pool, 
            # dominance_pool, balance_pool,
            # entropy_pool, hierarchy_pool,
            # effective_count_pool, coherence_pool, 
            # participation_pool, spectral_gap_pool,
            # correlation_length_pool, 
            # frustration_pool,                 
            # susceptibility_pool,               
            # topological_pool,                 
        ], axis=-1)
        
        x = nn.Dense(features=40)(x) 
        x = nn.relu(x)
        x = nn.Dense(features=40)(x) 
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
test_learning_rate = 0.005
num_blocks_reupload = 10
init_scale = 0.02
dev = qml.device("default.qubit", wires = num_qubit)
point_class_train = 960
point_class_test = 40
sigma = 0.02
# sigma = 0

# Theta = 1.7
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

print(get_Theta(dataset))
test_dataset_y = dataset['test_dataset_y']
num_classes = len(np.unique(test_dataset_y))
epochs = 600
l2 = 0.0000001
key, key_r = jax.random.split(key)

def result(gate_type, test_learning_rate, num_blocks_reupload, init_scale, use_augmentation):  
    train(gate_type, dataset, 32, Theta, epochs, key_r, init_scale, num_blocks_reupload, num_qubit, num_reupload, use_augmentation, learning_rate = test_learning_rate) 

result(gate_type, test_learning_rate, num_blocks_reupload, init_scale, use_augmentation)

