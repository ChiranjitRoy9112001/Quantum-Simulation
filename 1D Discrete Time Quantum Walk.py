import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the coin operator (Hadamard)
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])


# Define the position shift operators
def shift_operator(N):
    S_plus = np.roll(np.eye(N), 1, axis=0)
    S_minus = np.roll(np.eye(N), -1, axis=0)
    S = np.kron(S_plus, np.array([[1, 0], [0, 0]])) + np.kron(S_minus, np.array([[0, 0], [0, 1]]))
    return S


# Initialize the walk
def initialize_walk(N):
    # Initial state: |0> (position at the center) tensor product with |up>
    psi = np.zeros((N, 2), dtype=complex)
    psi[N // 2, 0] = 1  # Start at the center with |up>
    psi = psi.flatten()  # Flatten to a 1D vector
    return psi


# Single step of the quantum walk
def quantum_walk_step(psi, S, H):
    # Apply the coin operator (Hadamard)
    psi = np.kron(np.eye(len(psi) // 2), H).dot(psi)
    # Apply the shift operator
    psi = S.dot(psi)
    return psi


# Perform the quantum walk and return the probability distribution at each step
def quantum_walk_evolution(N, steps):
    psi = initialize_walk(N)
    S = shift_operator(N)

    probability_distributions = []

    for _ in range(steps):
        psi = quantum_walk_step(psi, S, H)
        probability_distribution = np.abs(psi.reshape(N, 2)) ** 2
        probability_distribution = probability_distribution.sum(axis=1)  # Sum over spin states
        probability_distributions.append(probability_distribution)

    return probability_distributions


# Set parameters for the simulation
N = 101  # Number of positions (must be odd to center the walker)
steps = 50  # Number of steps in the walk

# Run the quantum walk and get the evolution
probability_distributions = quantum_walk_evolution(N, steps)

# Set up the figure and axis for animation
fig, ax = plt.subplots(figsize=(10, 6))
positions = np.arange(N) - N // 2
bar_container = ax.bar(positions, probability_distributions[0], color='blue')
ax.set_xlabel('Position')
ax.set_ylabel('Probability')
ax.set_title('Discrete-Time Quantum Walk in 1D')


# Update function for the animation
def update(step):
    for bar, height in zip(bar_container, probability_distributions[step]):
        bar.set_height(height)
    ax.set_title(f'Discrete-Time Quantum Walk in 1D - Step {step + 1}/{steps}')
    return bar_container


# Create the animation with repeat set to True
ani = animation.FuncAnimation(fig, update, frames=range(steps), blit=False, repeat=True)

# Show the animation
plt.show()
