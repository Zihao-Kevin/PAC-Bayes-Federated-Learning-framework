import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulated data
n_clients = 10
n_dim = 5
data_set = [np.random.randn(100, n_dim) for _ in range(n_clients)]

# Initialize w_mu and w_sigma
w_mu = [np.random.randn(n_dim) for _ in range(n_clients)]
w_sigma = [np.random.randn(n_dim) for _ in range(n_clients)]

# Initialize the plot
fig, ax = plt.subplots()
line, = ax.plot(np.arange(n_dim), w_mu[0], 'ro-')

# Define the update function for the animation
def update(frame):
    global w_mu, w_sigma
    # Simulated update of w_mu and w_sigma
    w_mu = [mu + np.random.randn(n_dim) * 0.1 for mu in w_mu]
    w_sigma = [sigma + np.random.randn(n_dim) * 0.1 for sigma in w_sigma]
    line.set_data(np.arange(n_dim), w_mu[0])
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 100), blit=True)

plt.show()
