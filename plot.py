import matplotlib.pyplot as plt
import numpy as np

original_data = np.load('dataset_150_new.npz')
rotated_data = np.load('dataset_150_new_rotation.npz')

fig, axs = plt.subplots(ncols=2, figsize=(10, 5), subplot_kw={"projection":"3d"})

idx = 5
print(original_data["train_dataset_y"][idx])
axs[0].scatter(original_data["train_dataset_x"][idx,:,0], original_data["train_dataset_x"][idx,:,1], original_data["train_dataset_x"][idx,:,2])

axs[1].scatter(rotated_data["train_dataset_x"][idx,:,0], rotated_data["train_dataset_x"][idx,:,1], rotated_data["train_dataset_x"][idx,:,2])
plt.show()
