import pennylane as qml
import numpy as np
from gates import Spin_2, Spin_3, create_singlet
from generation import generate_sphere_point_cloud, generate_torus_point_cloud
import pandas as pd
import math
import pennylane.numpy as pnp
import itertools

dev = qml.device("lightning.qubit", wires = 12)


def loss(num_qubit, H, answer_train, index): 

    H_sigmoid = 1 / (1 + pnp.exp(-H))
    cross_entropy = 0

    for i  in range (num_qubit * index, num_qubit * (index + 1)):
        cross_entropy += answer_train[i] * pnp.log(H_sigmoid) + (1 - answer_train[i]) * pnp.log(1 - H_sigmoid)
    
    cross_entropy = -cross_entropy
    cross_entropy = cross_entropy / num_qubit

    return cross_entropy


def encode(vector, Alpha, Beta):

    theta = Alpha * np.arccos(vector[2])
    phi = Beta * np.arctan2(vector[1], vector[0])
    
    qml.RY(theta, wires=0)  
    qml.RZ(phi, wires=0)  


def create_Hamiltonian(num_qubit):

    H = sum([qml.PauliZ(i) @ qml.PauliZ((i + 1) % num_qubit) for i in range(num_qubit)])
    H += sum([qml.PauliX(i) @ qml.PauliX((i + 1) % num_qubit) for i in range(num_qubit)])
    H += sum([qml.PauliY(i) @ qml.PauliY((i + 1) % num_qubit) for i in range(num_qubit)])
    
    return H


def prepare_init_state(num_qubit):
    for i in range(0, num_qubit, 2):
        create_singlet(i, i+1) # 궁금증이 있는게 3qubit gate도 여기다가 적용하는게 맞나?


def create_u2_circuit(num_qubit, num_blocks, H, Alpha, Beta, data, index):

    def circuit_2qubits(params):
        prepare_init_state(num_qubit)
        for vector in data[num_qubit * index : num_qubit * (index + 1)]:
            encode(vector, Alpha, Beta)

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


def create_u3_circuit(num_qubit, num_blocks, H, Alpha, Beta, data, index):

    def circuit_3qubits(params):

        prepare_init_state(num_qubit)
        for vector in data[num_qubit * index : num_qubit * (index + 1)]:
            encode(vector, Alpha, Beta)

        k = 0
        for l in range(num_blocks):
            for i in range(0, num_qubit):
                Spin_3(params[k], params[k + 1], params[k + 2], params[k+3], wires=[i, (i + 1) % num_qubit, (i + 2) % num_qubit])
                k += 4
        return qml.expval(H)

    return circuit_3qubits


def create_data(outer_radius, inner_radius, sphere_radius, num_vectors):

    sphere_df = generate_sphere_point_cloud(max_radius=sphere_radius, num_vectors=num_vectors)
    torus_df = generate_torus_point_cloud(inner_radius=inner_radius, outer_radius=outer_radius, num_vectors=num_vectors)

    sphere_vector_train = []
    torus_vector_train = []
    sphere_vector_answer = []
    torus_vector_answer = []

    for i in range(num_vectors):

        sphere_vector = sphere_df.iloc[i, :-1].values
        sphere_vector = sphere_vector / np.linalg.norm(sphere_vector) 
        sphere_vector_train.append(sphere_vector)
        sphere_vector_answer.append(sphere_df.iloc[i, -1])

        torus_vector = torus_df.iloc[i, :-1].values
        torus_vector = torus_vector / np.linalg.norm(torus_vector)
        torus_vector_train.append(torus_vector)
        torus_vector_answer.append(torus_df.iloc[i, -1])


    num_train = int(0.75 * num_vectors) 
    num_val = int(0.25 * num_vectors)

    feats_train_sphere = [sphere_vector_train[i] for i in range(num_train)]
    answer_train_sphere = [sphere_vector_answer[i] for i in range(num_train)]
    feats_val_sphere = [sphere_vector_train[i] for i in range(num_train, num_vectors)]
    answer_val_sphere = [sphere_vector_answer[i] for i in range(num_train, num_vectors)]

    feats_train_torus = [torus_vector_train[i] for i in range(num_train)]
    answer_train_torus = [torus_vector_answer[i] for i in range(num_train)]
    feats_val_torus = [torus_vector_train[i] for i in range(num_train, num_vectors)]
    answer_val_torus = [torus_vector_answer[i] for i in range(num_train, num_vectors)]
    
    feats_train = feats_train_sphere + feats_train_torus
    answer_train = answer_train_sphere + answer_train_torus
    feats_val = feats_val_sphere + feats_val_torus
    answer_val = answer_val_sphere + answer_val_torus
    array_1 = []
    array_2 = []
    array_3 = []
    array_4 = []

    np.random.seed(0)

    index_permutation_train = np.random.permutation(int(2 * num_train / num_qubit))
    for i in range(int (2 * num_train / num_qubit)):
        array_1.append(feats_train[num_qubit * index_permutation_train[i] : num_qubit * (index_permutation_train[i] + 1)])
        array_2.append(answer_train[num_qubit * index_permutation_train[i] : num_qubit * (index_permutation_train[i] + 1)])
    feats_train = list(itertools.chain.from_iterable(array_1))
    answer_train = list(itertools.chain.from_iterable(array_2))

    index_permutation_val = np.random.permutation(int(2 * num_val / num_qubit))
    for i in range(int (2 * num_val / num_qubit)):
        array_3.append(feats_val[num_qubit * index_permutation_val[i] : num_qubit * (index_permutation_val[i] + 1)])
        array_4.append(answer_val[num_qubit * index_permutation_val[i] : num_qubit * (index_permutation_val[i] + 1)])
    feats_val = list(itertools.chain.from_iterable(array_3))
    answer_val = list(itertools.chain.from_iterable(array_4))

    return feats_train, answer_train, feats_val, answer_val


def train(gate_type):

    if (gate_type) == ("u2"):
        init_u2 = init_scale * math.pi/(2 * num_qubit * num_blocks)
        params_u2 = init_u2 * pnp.random.rand(2 * num_qubit * num_blocks)
        for index in range(batch_train):
            circuit_u2 = qml.QNode(create_u2_circuit(num_qubit, num_blocks, ham_sparse, Alpha, Beta, feats_train, index), dev, diff_method="adjoint")
            
            for epoch in range(epochs):
                params_u2, cost = opt_u2.step_and_cost(lambda p: loss(num_qubit, circuit_u2(p), answer_train, index), params_u2)
                print(f"{epoch}\t{cost}")

        params_train = params_u2
        print(np.array(params_train))
        return params_train

    if (gate_type) == ("u3"):
        init_u3 = init_scale*math.pi/(4 * num_qubit * num_blocks)
        params_u3 = init_u3 * pnp.random.rand(4 * num_qubit * num_blocks)
        for index in range(batch_train):
            circuit_u3 = qml.QNode(create_u3_circuit(num_qubit, num_blocks, ham_sparse, Alpha, Beta, feats_train, index), dev, diff_method="adjoint")
            
            for epoch in range(epochs):   
                params_u3, cost = opt_u3.step_and_cost(lambda p: loss(num_qubit, circuit_u3(p), answer_train, index), params_u3)
                print(f"{epoch}\t{cost}")

        params_train = params_u3
        print(np.array(params_train))
        return params_train

    else:
        print("Type Error")


def accuracy(labels, predictions):
    acc = sum(abs(l - p) < 0.1 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc


def predict(gate_type, params_train):

    if (gate_type) == ("u2"):
        predictions = []
        for index in range(batch_pre):
            circuit_u2 = qml.QNode(create_u2_circuit(num_qubit, num_blocks, ham_sparse, Alpha, Beta, feats_val, index), dev)
            pred = circuit_u2(params_train)
            pred_sigmoid = 1 / (1 + np.exp(-pred))
            for  i in range(num_qubit):
                predictions.append(float(pred_sigmoid))

        acc = accuracy(answer_val, predictions)
        print(acc)

    if (gate_type) == ("u3"):
        predictions = []        
        for index in range(batch_pre):
            circuit_u3 = qml.QNode(create_u3_circuit(num_qubit, num_blocks, ham_sparse, Alpha, Beta, feats_val, index), dev)
            pred = circuit_u3(params_train)
            pred_sigmoid = 1 / (1 + np.exp(-pred))  
            predictions.append(float(pred_sigmoid)) 
            for  i in range(num_qubit):
                predictions.append(float(pred_sigmoid))

        acc = accuracy(answer_val, predictions)
        print(acc)



inner_radius = 3
outer_radius = 5
num_vectors = 48
sphere_radius = inner_radius + outer_radius
num_qubit = 12
ham = create_Hamiltonian(12)
ham_sparse = qml.SparseHamiltonian(ham.sparse_matrix(), wires=range(12))
Alpha = 1
Beta =1 

batch_train = int((2 * num_vectors * 0.75 / num_qubit) - 1)
batch_pre = int((2 * num_vectors - num_qubit * (batch_train + 1)) / num_qubit)
adam_step = 0.05
num_blocks = 4
init_scale = 1
epochs = 10
opt_u2 = qml.AdamOptimizer(stepsize=adam_step)
opt_u3 = qml.AdamOptimizer(stepsize=adam_step)


feats_train, answer_train, feats_val, answer_val = create_data(outer_radius, inner_radius, sphere_radius, num_vectors) 

gate_type = input("Write gate_type")

params_train = train(gate_type)
print(params_train)
predict(gate_type ,params_train)
