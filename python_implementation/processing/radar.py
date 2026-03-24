import numpy as np
import matplotlib.pyplot as plt

# Parameters
M = 0.2
theta_deg = np.arange(-3, 3.01, 0.01)   # -3 to 3 degrees
theta = np.deg2rad(theta_deg)           # Convert to radians

# Preallocate
m90 = np.zeros_like(theta)
m150 = np.zeros_like(theta)

# Calculate modulation depths
for i in range(len(theta)):
    num90 = 1 + np.cos(np.pi * np.sin(theta[i])) + np.sin(np.pi * np.sin(theta[i]))
    num150 = 1 + np.cos(np.pi * np.sin(theta[i])) - np.sin(np.pi * np.sin(theta[i]))
    den = 1 + np.cos(np.pi * np.sin(theta[i]))

    m90[i] = M * num90 / den
    m150[i] = M * num150 / den

# DDM and SDM
DDM = m90 - m150
SDM = m90 + m150

# Plot
plt.figure(figsize=(8, 6))

plt.subplot(2, 1, 1)
plt.plot(theta_deg, DDM, linewidth=2)
plt.grid(True)
plt.xlabel(r'$\theta$ (degrees)')
plt.ylabel('DDM')
plt.title('DDM vs theta')

plt.subplot(2, 1, 2)
plt.plot(theta_deg, SDM, linewidth=2)
plt.grid(True)
plt.xlabel(r'$\theta$ (degrees)')
plt.ylabel('SDM')
plt.title('SDM vs theta')

plt.tight_layout()
plt.show()