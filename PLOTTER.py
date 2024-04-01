import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from scipy.signal import savgol_filter

# Open file dialog to select CSV file
root = filedialog.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
print(file_path)
# Read the CSV data
data = pd.read_csv(file_path)

# Convert depth from cm to mm
data['Depth (mm)'] = data['Depth (mm)'] * 10

# 1. Slice Thickness vs. Depth Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(data['Depth (mm)'], data['Thickness (mm)'], fmt='o-', label='Probe Data')
ax.set_xlabel('Depth (mm)')
ax.set_ylabel('Slice Thickness (mm)')
ax.set_title('Slice Thickness    vs. Depth')
ax.legend()

# Find and plot the slice thickness values at every 10 mm
target_depths_mm = np.arange(10, int(data['Depth (mm)'].max() + 1), 10)
for target_depth in target_depths_mm:
    depth_diff = np.abs(data['Depth (mm)'] - target_depth)
    nearest_idx = depth_diff.idxmin()
    thickness_at_target = data.loc[nearest_idx, 'Thickness (mm)']
    ax.plot(target_depth, thickness_at_target, 'ro', markersize=10, label=f'~{target_depth} mm')

ax.legend()

# 2. Savitzky-Golay filter for smoothing
depth = data['Depth (mm)'].values
slice_thickness = data['Thickness (mm)'].values

# Apply Savitzky-Golay filter to smooth the data
window_length = 51  # Adjust this value to control the degree of smoothing
polyorder = 3  # Polynomial order, adjust as needed
smoothed_slice_thickness = savgol_filter(slice_thickness, window_length, polyorder)

# Separate plot for smoothed data
fig_smoothed, ax_smoothed = plt.subplots(figsize=(8, 6))
ax_smoothed.plot(depth, smoothed_slice_thickness, 'r-', label='Smoothed Line', linewidth=2)
ax_smoothed.set_xlabel('Depth (mm)')
ax_smoothed.set_ylabel('Slice Thickness (mm)')
ax_smoothed.set_title('Smoothed Slice Thickness vs. Depth')
ax_smoothed.legend()

# 2. Bar plot of slice thickness every 100 mm
fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
depth_bins = np.arange(np.min(data['Depth (mm)']), np.max(data['Depth (mm)']) + 100, 100)
bin_means = [data['Thickness (mm)'][np.logical_and(data['Depth (mm)'] >= depth_bins[i], data['Depth (mm)'] < depth_bins[i+1])].mean() for i in range(len(depth_bins)-1)]
bin_centers = (depth_bins[1:] + depth_bins[:-1]) / 2
ax_bar.bar(bin_centers, bin_means, width=50)
ax_bar.set_title('Slice Thickness by 100mm Depth Bins')
ax_bar.set_xlabel('Depth (mm)')
ax_bar.set_ylabel('Slice Thickness (mm)')

plt.show()

# 3. Resolution Integral Calculation
depth = data['Depth (mm)'].values
thickness = data['Thickness (mm)'].values
reciprocal_thickness = 1 / thickness

# Numerical integration using the trapezoidal rule
resolution_integral = np.trapz(reciprocal_thickness, depth)
print(f'Resolution Integral (R): {resolution_integral:.2f}')

# 4. Depth of Field (LR) and Characteristic Resolution (DR)
# Assuming the curve is a rectangular area
area = resolution_integral
width = 1 / reciprocal_thickness[-1]  # Minimum slice thickness
height = area / width
depth_of_field = height
characteristic_resolution = width

print(f'Depth of Field (LR): {depth_of_field:.2f} mm')
print(f'Characteristic Resolution (DR): {characteristic_resolution:.2f} mm')

# 5. Probe Comparison Matrix
slice_thicknesses = []
for target_depth in target_depths_mm:
    depth_diff = np.abs(data['Depth (mm)'] - target_depth)
    nearest_idx = depth_diff.idxmin()
    slice_thicknesses.append(data.loc[nearest_idx, 'Thickness (mm)'])

probe_metrics = {
    'Resolution Integral (R)': resolution_integral,
    'Depth of Field (LR)': depth_of_field,
    'Characteristic Resolution (DR)': characteristic_resolution,
}

for i, target_depth in enumerate(target_depths_mm):
    probe_metrics[f'Slice Thickness at ~{target_depth} mm'] = slice_thicknesses[i]

probe_metrics['Maximum Slice Thickness'] = data['Thickness (mm)'].max()
probe_metrics['Minimum Slice Thickness'] = data['Thickness (mm)'].min()

print('\nProbe Comparison Matrix:')
for metric, value in probe_metrics.items():
    print(f'{metric}: {value:.2f}')