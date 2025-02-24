import numpy as np
from generation import generate_sphere_point_cloud, generate_torus_point_cloud
from scipy.stats import special_ortho_group

def _create_dataset(num_dataset, num_vectors):
    dataset_x = []
    dataset_y = []
    for i in range(num_dataset // 2):
        sphere_radius = np.random.uniform(3, 7)
        sphere_pts = generate_sphere_point_cloud(radius=sphere_radius, num_vectors=num_vectors)

        dataset_x.append(sphere_pts)
        dataset_y.append(0)

    for i in range(num_dataset // 2):
        inner_radius = np.random.uniform(3, 4)
        outer_radius = inner_radius + np.random.uniform(1, 3)
        rot = special_ortho_group.rvs(3)
        torus_pts = generate_torus_point_cloud(inner_radius=inner_radius, outer_radius=outer_radius, num_vectors=num_vectors)

        dataset_x.append(torus_pts @ rot)
        dataset_y.append(1)

    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y)

    perm = np.random.permutation(range(num_dataset))
    dataset_x = dataset_x[perm,:,:]
    dataset_y = dataset_y[perm]

    return dataset_x, dataset_y

num_vectors = 18
num_train_dataset = 1024
num_test_dataset = 256

train_dataset_x, train_dataset_y = _create_dataset(num_train_dataset, num_vectors)
test_dataset_x, test_dataset_y = _create_dataset(num_test_dataset, num_vectors)

np.savez(f'dataset_{num_vectors}.npz', train_dataset_x = train_dataset_x, train_dataset_y = train_dataset_y, test_dataset_x = test_dataset_x, test_dataset_y = test_dataset_y)
