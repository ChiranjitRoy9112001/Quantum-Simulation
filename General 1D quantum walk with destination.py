import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import expm

# Parameters
N = 100  # Number of positions on the 1D line
t_max = 20  # Maximum time for simulation
dt = 0.1  # Time step for simulation
start_position = 10  # Starting point for the quantum walker
destination_position = 90  # Destination point
hbar = 1  # Reduced Planck constant (set to 1 for simplicity)
potential_strength = -0.5  # Strength of the potential at the destination

# Hamiltonian: nearest-neighbor hopping on a 1D line with a potential at the destination
H = np.zeros((N, N), dtype=complex)
for i in range(N):
    if i > 0:
        H[i, i-1] = -1  # Hopping to the left
    if i < N-1:
        H[i, i+1] = -1  # Hopping to the right
    if i == destination_position:
        H[i, i] += potential_strength  # Potential well at the destination

# Initial state: peaked at the start_position
psi0 = np.zeros(N, dtype=complex)
psi0[start_position] = 1

# Function to calculate the state at time t
def calculate_state(t):
    # Time evolution operator: U(t) = exp(-iHt/hbar)
    U = expm(-1j * H * t / hbar)  # Use scipy.linalg.expm for matrix exponential
    psi_t = U @ psi0  # State at time t
    return np.abs(psi_t)**2  # Probability distribution

# Plotting setup
fig, ax = plt.subplots()
x = np.arange(N)
line, = ax.plot(x, calculate_state(0))
ax.set_ylim(0, 1)
ax.set_xlim(0, N-1)
ax.set_title(f"1D Continuous-Time Quantum Walk: Start = {start_position}, Destination = {destination_position}")
ax.set_xlabel("Position")
ax.set_ylabel("Probability")

# Animation function
def animate(t):
    prob = calculate_state(t*dt)
    line.set_ydata(prob)
    return line,

# Create animation
ani = FuncAnimation(fig, animate, frames=int(t_max/dt), interval=50, blit=True)

# Show plot
plt.show()
