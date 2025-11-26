import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Define nonlinear demand and supply equations
# ==========================================================
# Example: demand decreases with Q^2, supply increases with Q^2
def demand(Q):
    return 60 - 0.01 * (Q**2)   # inverse demand: curved downward

def supply(Q):
    return 10 + 0.005 * (Q**2)  # inverse supply: curved upward

# ==========================================================
# Find equilibrium
# ==========================================================
# Solve numerically for where demand(Q) = supply(Q)
Q = np.linspace(0, 100, 1000)
P_d = demand(Q)
P_s = supply(Q)

# Find the index where they are closest (approx equilibrium)
idx_eq = np.argmin(np.abs(P_d - P_s))
Q_eq = Q[idx_eq]
P_eq = P_d[idx_eq]

# ==========================================================
# Plot
# ==========================================================
plt.plot(Q, P_d, label="Demand (curved)", color="blue")
plt.plot(Q, P_s, label="Supply (curved)", color="red")

# Mark equilibrium point
plt.scatter(Q_eq, P_eq, color="black", zorder=5)
plt.text(Q_eq + 2, P_eq + 1, f"Equilibrium\n(Q={Q_eq:.1f}, P={P_eq:.1f})", fontsize=10)

# Labels and title
plt.title("Curved Supply and Demand with Equilibrium")
plt.xlabel("Quantity")
plt.ylabel("Price")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
