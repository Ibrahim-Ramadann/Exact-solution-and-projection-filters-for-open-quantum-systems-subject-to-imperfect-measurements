import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# --- 1. Simulation Parameters ---
T = 5
dt = (2**-12) * T
steps = int(T / dt)
time_axis = np.linspace(0, T, steps)

# Physical parameters from your paper
omega_eg = 1.0
M = 1.0
eta = 0.5
alpha_fb, beta_fb, gamma_fb = 7.61, 5, 10  # Feedback parameters

# --- 2. Operators & Constants ---
# Use dtype=complex for all quantum operators
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

H0 = (omega_eg / 2.0) * sigma_z
L = (np.sqrt(M) / 2.0) * sigma_z
rho_e = np.array([[0, 0], [0, 1]], dtype=complex)  # Excited state target (z = -1)

# Projection Manifold Projectors
P1 = np.array([[1, 0], [0, 0]], dtype=complex)
P2 = np.array([[0, 0], [0, 1]], dtype=complex)
lambdas = [np.sqrt(M)/2, -np.sqrt(M)/2]

# --- 3. Helper Functions ---
def get_bloch_z(rho):
    """Extracts the z-coordinate from the density matrix."""
    return np.real(np.trace(rho @ sigma_z))

def get_feedback(rho_approx):
    """Calculates control u(rho) from the paper."""
    # V(rho) = sqrt(1 - Tr(rho * rho_e))
    V_rho = np.sqrt(max(0, 1 - np.real(np.trace(rho_approx @ rho_e))))
    # Commutator term for feedback
    comm = 1j * (sigma_y @ rho_approx - rho_approx @ sigma_y)
    term2 = -gamma_fb * np.real(np.trace(comm @ rho_e))
    return alpha_fb * (V_rho**beta_fb) + term2

# --- 4. Initialization ---
# Start from x = -1 on the Bloch Sphere: rho0 = 0.5 * (I - sigma_x)
rho_full = 0.5 * (np.eye(2, dtype=complex) - sigma_x)
theta = np.zeros(2, dtype=float) 

errors = np.zeros(steps)
z_trajectory = np.zeros(steps)

# --- 5. Main Simulation Loop ---
print("Running simulation...")
for i in range(steps):
    # a. Reconstruct approximate state from manifold parameters theta
    # Since P1 and P2 are projectors: exp(0.5 * theta * P) = I + (exp(0.5 * theta) - 1) * P
    exp_op = np.eye(2, dtype=complex) + (np.exp(0.5 * theta[0]) - 1) * P1 + (np.exp(0.5 * theta[1]) - 1) * P2
    rho_approx_tilde = exp_op @ rho_full @ exp_op.conj().T
    rho_approx = rho_approx_tilde / np.trace(rho_approx_tilde)
    
    # b. Compute Feedback based on the current approximation
    u = get_feedback(rho_approx)
    
    # c. Stochastic terms
    dW = np.random.normal(0, np.sqrt(dt))
    expect_L = np.real(np.trace((L + L.conj().T) @ rho_full))
    # Innovation process dY
    dY = dW + np.sqrt(eta) * expect_L * dt
    
    # d. Update Full Quantum Filter (Ground Truth)
    H_total = H0 + u * sigma_y
    drift = -1j * (H_total @ rho_full - rho_full @ H_total) * dt
    diss = (L @ rho_full @ L.conj().T - 0.5 * (L.conj().T @ L @ rho_full + rho_full @ L.conj().T @ L)) * dt
    diff = np.sqrt(eta) * (L @ rho_full + rho_full @ L.conj().T - expect_L * rho_full) * dW
    
    rho_full = rho_full + drift + diss + diff
    # Keep rho_full physical (Hermitian and Trace 1)
    rho_full = (rho_full + rho_full.conj().T) / 2.0
    rho_full /= np.trace(rho_full)
    
    # e. Update Projection Filter parameters (Equation 14)
    for k in range(2):
        theta[k] += -2 * eta * (lambdas[k]**2) * dt + 2 * np.sqrt(eta) * lambdas[k] * dY
        
    # f. Record data
    diff_matrix = rho_full - rho_approx
    errors[i] = np.sqrt(np.real(np.trace(diff_matrix.conj().T @ diff_matrix)))
    z_trajectory[i] = get_bloch_z(rho_full)

print("Simulation complete.")

# --- 6. Plotting Results ---
plt.figure(figsize=(12, 5))

# Plot 1: Error Analysis
plt.subplot(1, 2, 1)
plt.plot(time_axis, errors, color='blue')
plt.title("Approximation Error (Frobenius Norm)")
plt.xlabel("Time (t)")
plt.ylabel("Error")
plt.grid(True, alpha=0.3)

# Plot 2: Stabilization
plt.subplot(1, 2, 2)
plt.plot(time_axis, z_trajectory, color='red', label="Actual z(t)")
plt.axhline(y=-1, color='black', linestyle='--', label="Target (z = -1)")
plt.title("Feedback Stabilization to Excited State")
plt.xlabel("Time (t)")
plt.ylabel("Bloch z-coordinate")
plt.legend()
plt.grid(True, alpha=0.3)



plt.tight_layout()
plt.show()
