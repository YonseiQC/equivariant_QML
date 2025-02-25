import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import rv_continuous

def cartesian_to_coordinate(x, y, z):
    radius = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x) + np.pi  

    if radius != 0:
        phi = np.arccos(z / radius) 
    else:
        phi = 0.0  

    return radius, theta, phi

class sin_prob_dist(rv_continuous):
    def _pdf(self, theta):
        return 0.5 * np.sin(theta)

sin_sampler = sin_prob_dist(a=0, b=np.pi)


def generate_sphere_point_cloud(radius, num_vectors):

    sphere_point_cloud = []
    n = num_vectors

    for i in range(num_vectors):
      theta = np.random.uniform(0, 2 * np.pi)
      phi = sin_sampler.rvs(size=1)[0]

      x = radius * np.sin(phi) * np.cos(theta)
      y = radius * np.sin(phi) * np.sin(theta)
      z = radius * np.cos(phi)

      sphere_point_cloud.append([x, y, z]) 

    return np.array(sphere_point_cloud)

def generate_torus_point_cloud(inner_radius, outer_radius, num_vectors):

    torus_point_cloud = []
    n = num_vectors

    for i in range(num_vectors):
      theta = np.random.uniform(0, 2 * np.pi)
      phi_torus = np.random.uniform(0, 2 * np.pi)

      x = (outer_radius + inner_radius * np.cos(phi_torus)) * np.cos(theta)
      y = (outer_radius + inner_radius * np.cos(phi_torus)) * np.sin(theta)
      z = inner_radius * np.sin(phi_torus)

      torus_point_cloud.append([x, y, z])

    return np.array(torus_point_cloud)
