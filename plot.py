# Import necessary libraries
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import math
import pandas as pd

# Load the data
data = scio.loadmat('Eng_bsfc_map.mat')
Eng_bsfc_map = data["Eng_bsfc_map"]

# Define the lists
Eng_spd_list_1D = np.arange(0, 4501, 125) * (2 * math.pi) / 60
Eng_trq_list_1D = np.arange(0, 111, 5) * (121 / 110)

# Generate a meshgrid for contour plotting
Eng_spd_list_mesh, Eng_trq_list_mesh = np.meshgrid(Eng_spd_list_1D, Eng_trq_list_1D)

# Load the new data
filename = 'Book5.xlsx'
df_new = pd.read_excel(filename)

# Remove rows where either speed or torque is 0
df_new = df_new[(df_new.iloc[:, 0] != 0) & (df_new.iloc[:, 1] != 0)]

# Convert speed to rad/sec
df_new.iloc[:, 0] = df_new.iloc[:, 0] * (2 * math.pi) / 60

# Find the minimum BSFC value for each speed
min_bsfc_index = np.argmin(Eng_bsfc_map, axis=1)
optimal_trq = Eng_trq_list_1D[min_bsfc_index]

# Plot the BSFC map with even more contour lines

plt.figure(figsize=(10, 8))
contour = plt.contourf(Eng_spd_list_mesh.T, Eng_trq_list_mesh.T, Eng_bsfc_map, cmap='viridis', levels=100)
plt.colorbar(contour, label='Eng_bsfc_map values (g/kWh)')
plt.contour(Eng_spd_list_mesh.T, Eng_trq_list_mesh.T, Eng_bsfc_map, colors='black', linewidths=0.5, levels=100)

# Plot the new data points with smaller size and red color
plt.scatter(df_new.iloc[:, 0], df_new.iloc[:, 1], color='red', s=5)

# Plot the optimal BSFC curve
plt.plot(Eng_spd_list_1D, optimal_trq, color='green', linewidth=1)

plt.xlabel('Eng_spd_list (radians/sec)')
plt.ylabel('Eng_trq_list')
#plt.title('Eng_bsfc_map for Prius with Data Points and Optimal BSFC Curve')
plt.show()
