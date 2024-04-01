"""
User Guide:

This program analyzes a video file to measure the thickness of a sample at different depths. It provides the following features:

1. Video file selection: The user can select the video file to be analyzed.
2. Save location selection: The user can choose the directory where the results will be saved.
3. User input prompts:
   - Maximum depth (in cm) covered by the video
   - Frame interval (every nth frame to analyze)
   - Desired interval for measurements (in mm)
   - Option to use multiple lines for measurement
   - Option to enable visualizations

4. Exclusion zone selection (optional): The user can set an exclusion zone at the top of the frame to ignore a specific depth range.
5. Video analysis: The program will analyze the video frame by frame, preprocess the frames, extract vertical lines, and calculate the thickness for each line.
6. Results saving: The thickness measurements at different depths are saved in a CSV file.
7. Results plotting: A plot of thickness vs. depth is displayed.
8. Log file generation: A log file is created with the user-provided input parameters, settings, and the time taken for video analysis.

Note: The program assumes that the video file shows a sample with a consistent background, and the intensity profile of the vertical lines represents the thickness of the sample.

Usage:
1. Run the program.
2. Follow the prompts to provide the required inputs.
3. Select the video file and the save location when prompted.
4. If desired, set the exclusion zone depth.
5. Wait for the video analysis and results generation.
6. Check the save location for the CSV file containing the results and the log file.
7. The plot of thickness vs. depth will be displayed.

TODO:
- Add Pixel to MM calibration
- Add a final frame pixel depth selector for this to calculate px to cm
- Calculate pixel range between start of exclusion zone to end of capture area
- Divide total distance by pixels to find this
- Save csv of pixel data and the pixel to mm ratio in the log file!
"""

import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import time

def select_video_file():
    """
    Opens a file dialog to select the video file.
    Returns the file path of the selected video file.
    """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

def select_save_location():
    """
    Opens a directory dialog to select the save location.
    Returns the path of the selected directory.
    """
    root = tk.Tk()
    root.withdraw()
    save_location = filedialog.askdirectory()
    return save_location

def prompt_user():
    """
    Prompts the user for input parameters.
    Returns the maximum depth, frame interval, desired interval, use of multiple lines, and enable visualizations.
    """
    max_depth_cm = float(input("Enter the maximum depth (in cm) covered by the video: "))
    max_depth_mm = max_depth_cm * 10  # Convert to millimeters
    frame_interval = int(input("Enter the frame interval: "))
    desired_interval = float(input("Enter the desired interval for measurements (in mm): "))
    use_multiple_lines = input("Do you want to use multiple lines for measurement? (y/n): ").lower() == "y"
    enable_visualizations = input("Do you want to enable visualizations? (y/n): ").lower() == "y"
    return max_depth_mm, frame_interval, desired_interval, use_multiple_lines, enable_visualizations

def preprocess_frame(frame):
    """
    Preprocesses the input frame by converting it to grayscale and applying Gaussian blur.
    If visualizations are enabled, it displays the preprocessed frame.
    Returns the preprocessed frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    if enable_visualizations:
        visualize_frame(blurred, "Preprocessed Frame")
    return blurred

def extract_vertical_lines(frame, num_lines=1, line_separation_percent=0.3):
    """
    Extracts vertical lines from the input frame.
    Args:
        frame: The input frame.
        num_lines: The number of vertical lines to extract.
        line_separation_percent: The separation between lines as a percentage of the frame width.
    Returns a list of vertical line intensity profiles.
    """
    height, width = frame.shape[:2]
    line_separation = int(width * line_separation_percent)
    middle_x = width // 2
    lines = []
    for i in range(num_lines):
        x = middle_x + (i - num_lines // 2) * line_separation
        line = frame[:, x].copy()
        lines.append(line)
    return lines

def calculate_thickness(intensity_profile, threshold=0.5):
    """
    Calculates the thickness of the sample based on the intensity profile.
    Args:
        intensity_profile: The intensity profile of a vertical line.
        threshold: The threshold for determining the thickness boundaries.
    Returns the calculated thickness.
    """
    max_intensity = np.max(intensity_profile)
    half_max = max_intensity * threshold
    thickness = 0
    in_peak = False
    for i in range(len(intensity_profile)):
        if intensity_profile[i] >= half_max and not in_peak:
            start = i
            in_peak = True
        elif intensity_profile[i] < half_max and in_peak:
            end = i
            thickness = end - start
            in_peak = False
    return thickness

def visualize_frame(frame, title="Original Frame"):
    """
    Displays the input frame with the given title.
    If visualizations are not enabled, this function does nothing.
    """
    if enable_visualizations:
        cv2.imshow(title, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def visualize_vertical_lines(frame, vertical_lines, line_thickness=1):
    """
    Displays the input frame with vertical lines overlaid.
    If visualizations are not enabled, this function does nothing.
    """
    if enable_visualizations:
        height, width = frame.shape[:2]
        line_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # Green, Blue, Red
        for i, line in enumerate(vertical_lines):
            middle_x = width // 2 + (i - len(vertical_lines) // 2) * line_thickness
            cv2.line(frame, (middle_x, 0), (middle_x, height - 1), line_colors[i % len(line_colors)], line_thickness)
        visualize_frame(frame, "Vertical Lines")

def visualize_intensity_profiles(vertical_lines):
    """
    Displays the intensity profiles of the vertical lines.
    If visualizations are not enabled, this function does nothing.
    """
    if enable_visualizations:
        fig, axs = plt.subplots(len(vertical_lines), 1, figsize=(8, 6 * len(vertical_lines)), squeeze=False)
        for i, line in enumerate(vertical_lines):
            axs[i, 0].plot(line)
            axs[i, 0].set_title(f"Intensity Profile (Line {i+1})")
        plt.tight_layout()
        plt.show()

def visualize_intensity_profile(frame, vertical_lines, line_thicknesses, depth, line_separation_percent=0.1):
    """
    Displays the input frame with vertical lines and their thicknesses overlaid.
    If visualizations are not enabled, this function does nothing.
    """
    if enable_visualizations:
        height, width = frame.shape[:2]
        line_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # Green, Blue, Red
        line_thickness = max(1, min(height // 100, width // 100))
        line_separation = int(width * line_separation_percent)

        for i, line in enumerate(vertical_lines):
            middle_x = width // 2 + (i - len(vertical_lines) // 2) * line_separation
            thickness = line_thicknesses[i]
            top_boundary = max(0, height // 2 - thickness // 2)
            bottom_boundary = min(height - 1, height // 2 + thickness // 2)
            start_point = (middle_x, top_boundary)
            end_point = (middle_x, bottom_boundary)
            cv2.line(frame, start_point, end_point, line_colors[i % len(line_colors)], line_thickness)
            cv2.putText(frame, f"Thickness: {thickness:.2f}", (middle_x + 10, top_boundary + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_colors[i % len(line_colors)], 1)

        # Draw a line to indicate the slice depth
        slice_depth_y = int(height * (1 - depth / max_depth))
        cv2.line(frame, (0, slice_depth_y), (width - 1, slice_depth_y), (255, 255, 255), 2)
        cv2.putText(frame, f"Depth: {depth:.2f} cm", (10, slice_depth_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        visualize_frame(frame, "Intensity Profile")

def analyze_video(video_path, frame_interval, max_depth, num_lines=1, threshold=0.5, line_separation_percent=0.1, top_threshold_pixels=0):
    """
    Analyzes the video file and calculates the thickness of the sample at different depths.
    Args:
        video_path: The path to the video file.
        frame_interval: The interval between frames to analyze.
        max_depth: The maximum depth (in cm) covered by the video.
        num_lines: The number of vertical lines to use for measurement.
        threshold: The threshold for determining the thickness boundaries.
        line_separation_percent: The separation between lines as a percentage of the frame width.
        top_threshold_pixels: The number of pixels to exclude from the top of the frame.
    Returns the depths, corresponding thicknesses, analysis time, and total vertical pixels as lists.
    """
    start_time = time.time()  # Start timer

    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    depths = []
    thicknesses = []
    max_frame_num = frame_count  # Use the total number of frames

    if use_exclusion_zone:
        # Display the first frame and prompt for exclusion zone
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = video.read()
        if success:
            frame_with_lines = frame.copy()
            for y in range(25, height, 25):
                cv2.line(frame_with_lines, (0, y), (frame_with_lines.shape[1], y), (0, 255, 0), 1)
            visualize_frame(frame_with_lines, "Exclusion Zone Selection")

            exclusion_zone_depth = float(input(f"Enter the desired exclusion zone depth (in pixels, max {height}): "))
            while exclusion_zone_depth < 0 or exclusion_zone_depth > height:
                exclusion_zone_depth = float(input(f"Invalid input. Enter a value between 0 and {height}: "))

            frame_with_exclusion = frame.copy()
            cv2.line(frame_with_exclusion, (0, int(exclusion_zone_depth)), (frame_with_exclusion.shape[1], int(exclusion_zone_depth)), (0, 0, 255), 2)
            visualize_frame(frame_with_exclusion, "Exclusion Zone Preview")

            top_threshold_pixels = int(exclusion_zone_depth)
            total_pixels = frame_height - top_threshold_pixels
    else:
        total_pixels = frame_height

    depth_increment = max_depth / frame_count  # Increment in depth per frame

    for frame_num in range(0, max_frame_num, frame_interval):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = video.read()

        if not success:
            continue

        if enable_visualizations:
            visualize_frame(frame, "Original Frame")

        preprocessed_frame = preprocess_frame(frame)

        # Apply top threshold
        preprocessed_frame[:top_threshold_pixels, :] = 0

        vertical_lines = extract_vertical_lines(preprocessed_frame, num_lines=num_lines, line_separation_percent=line_separation_percent)
        if enable_visualizations:
            visualize_vertical_lines(frame, vertical_lines)
            visualize_intensity_profiles(vertical_lines)

        line_thicknesses = []
        for line in vertical_lines:
            thickness = calculate_thickness(line, threshold)
            if thickness > 0:  # Skip zero values
                line_thicknesses.append(thickness)

        if line_thicknesses:  # Check if there are valid measurements
            avg_thickness = sum(line_thicknesses) / len(line_thicknesses)
            depth = frame_num * depth_increment  # Calculate depth based on frame number and depth increment
            depths.append(depth)
            thicknesses.append(avg_thickness)

            if enable_visualizations:
                visualize_intensity_profile(frame, vertical_lines, line_thicknesses, depth, line_separation_percent)

    video.release()
    cv2.destroyAllWindows()

    end_time = time.time()  # End timer
    analysis_time = end_time - start_time  # Calculate analysis time

    return depths, thicknesses, analysis_time, total_pixels

# Main code
video_path = select_video_file()
save_location = select_save_location()

# Get frame dimensions
video = cv2.VideoCapture(video_path)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
video.release()

# Prompt the user for input
max_depth_mm, frame_interval, desired_interval, use_multiple_lines, enable_visualizations = prompt_user()

use_exclusion_zone = input("Do you want to set an exclusion zone? (y/n): ").lower() == "y"

if use_multiple_lines:
    num_lines = int(input("Enter the number of lines to use (e.g., 3, 5): "))
    # Analyze the video with multiple lines
    depths, thicknesses, analysis_time, total_pixels = analyze_video(video_path, frame_interval, max_depth_mm, num_lines, threshold=0.5, line_separation_percent=0.1, top_threshold_pixels=top_threshold_pixels if use_exclusion_zone else 0)
else:
    # Analyze the video with a single line
    depths, thicknesses, analysis_time, total_pixels = analyze_video(video_path, frame_interval, max_depth_mm, 1, threshold=0.5, line_separation_percent=0.1, top_threshold_pixels=top_threshold_pixels if use_exclusion_zone else 0)

# Calculate the pixel-to-mm ratio
pixel_to_mm_ratio = max_depth_mm / total_pixels

# Convert thicknesses from pixels to millimeters
thicknesses_mm = [thickness * pixel_to_mm_ratio for thickness in thicknesses]

# Save the results to a CSV file
timestamp = time.strftime("%Y%m%d_%H%M%S")
csv_filename = f"{save_location}/results_{timestamp}.csv"
with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Depth (mm)", "Thickness (pixels)", "Thickness (mm)"])
    for depth, thickness, thickness_mm in zip(depths, thicknesses, thicknesses_mm):
        writer.writerow([depth, thickness, thickness_mm])

print(f"Results saved to {csv_filename}")

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(depths, thicknesses_mm, 'o-')
plt.xlabel('Depth (mm)')
plt.ylabel('Thickness (mm)')
plt.title('Thickness vs. Depth')
plt.show()

# Write log file
log_file_name = f"results_{timestamp}_log.txt"
with open(f"{save_location}/{log_file_name}", "w") as log_file:
    log_file.write(f"Maximum Depth: {max_depth_mm / 10} cm ({max_depth_mm} mm)\n")
    log_file.write(f"Frame Interval: {frame_interval}\n")
    log_file.write(f"Desired Interval: {desired_interval} mm\n")
    log_file.write(f"Use Multiple Lines: {use_multiple_lines}\n")
    log_file.write(f"Number of Lines: {num_lines if use_multiple_lines else 1}\n")
    log_file.write(f"Use Exclusion Zone: {use_exclusion_zone}\n")
    log_file.write(f"Enable Visualizations: {enable_visualizations}\n")
    log_file.write(f"Time to Analyze Video and Calculate Thicknesses: {analysis_time:.2f} seconds\n")
    log_file.write("\n")
    log_file.write(f"Number of Vertical Pixels = {total_pixels}\n")
    log_file.write(f"Max Depth (mm) = {max_depth_mm}\n")
    log_file.write(f"Pixel to MM Ratio = {pixel_to_mm_ratio:.6f}\n")

print(f"Log file saved as {save_location}/{log_file_name}")