import numpy as np
from generation import generate_sphere_point_cloud, generate_torus_point_cloud
from scipy.stats import special_ortho_group
np.random.seed(42)
rot = special_ortho_group.rvs(3, random_state=42)

def _create_dataset(num_dataset, num_vectors):
    dataset_x = []
    dataset_y = []
    for i in range((num_dataset) // 2):
        sphere_radius = np.random.uniform(3, 7)
        sphere_pts = generate_sphere_point_cloud(radius = sphere_radius, num_vectors = num_vectors)

        dataset_x.append(sphere_pts)
        dataset_y.append(0)

    for i in range((num_dataset) // 2):
        inner_radius = np.random.uniform(3, 4)
        outer_radius = inner_radius + np.random.uniform(1, 3)
        rot = special_ortho_group.rvs(3)
        torus_pts = generate_torus_point_cloud(inner_radius = inner_radius, outer_radius = outer_radius, num_vectors = num_vectors)

        dataset_x.append(torus_pts @ rot)
        dataset_y.append(1)

    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y)

    perm = np.random.permutation(range(num_dataset))
    dataset_x = dataset_x[perm,:,:]
    dataset_y = dataset_y[perm]

    return dataset_x, dataset_y

num_train_dataset = 1024
num_test_dataset = 4

def make_point(type, num_vectors, num_reupload):
    if type == "single":
        train_dataset_x, train_dataset_y = _create_dataset(num_train_dataset, num_vectors)
        test_dataset_x, test_dataset_y = _create_dataset(num_test_dataset, num_vectors)

        np.savez(f'dataset_{num_vectors}.npz', train_dataset_x = train_dataset_x, train_dataset_y = train_dataset_y, test_dataset_x = test_dataset_x, test_dataset_y = test_dataset_y)
        print("succeed")

        train_dataset_x = train_dataset_x @ rot
        test_dataset_x = test_dataset_x @ rot

        np.savez(f'dataset_{num_vectors}_rotation.npz', train_dataset_x = train_dataset_x, train_dataset_y = train_dataset_y, test_dataset_x = test_dataset_x, test_dataset_y = test_dataset_y)
        print("succeed")

    
    elif type == "multiple":
        train_dataset_x, train_dataset_y = _create_dataset(num_train_dataset, num_vectors * num_reupload)
        test_dataset_x, test_dataset_y = _create_dataset(num_test_dataset, num_vectors * num_reupload)

        np.savez(f'dataset_{num_vectors}_{num_reupload}.npz', train_dataset_x = train_dataset_x, train_dataset_y = train_dataset_y, test_dataset_x = test_dataset_x, test_dataset_y = test_dataset_y)
        print("succeed")

        train_dataset_x = train_dataset_x @ rot
        test_dataset_x = test_dataset_x @ rot

        np.savez(f'dataset_{num_vectors}_{num_reupload}_rotation.npz', train_dataset_x = train_dataset_x, train_dataset_y = train_dataset_y, test_dataset_x = test_dataset_x, test_dataset_y = test_dataset_y)
        print("succeed")

make_point("single", 8, 1)

# main_reupload.py에 문제가 있는것 같음. 정확히는 reshape할 때 정답 label이랑 train의 index가 잘 안맞는 것 같음
# test_dataset_x -> (256,20,3) / test_dataset_y -> (256,)
