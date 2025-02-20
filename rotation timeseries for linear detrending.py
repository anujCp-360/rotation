import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def align_trend_to_x_axis(merged_df):
    """
    Rotates time-series data so that the last point aligns with the positive x-axis.

    Parameters:
        merged_df (pd.DataFrame): Input DataFrame with 'start_time' and 'TimeGPT' columns.
        
    Returns:
        pd.DataFrame: Transformed DataFrame with rotated 'TimeGPT' values.
    """
    # Ensure 'start_time' is datetime
    merged_df = merged_df.copy()
    merged_df['start_time'] = pd.to_datetime(merged_df['start_time'])

    # Convert timestamps to numeric values (elapsed seconds)
    merged_df['time_numeric'] = (merged_df['start_time'] - merged_df['start_time'].iloc[0]).dt.total_seconds()

    # Get the coordinates of the last point
    last_x = merged_df['time_numeric'].iloc[-1]
    last_y = merged_df['TimeGPT'].iloc[-1]

    # Compute the angle between the origin and the last point using arctan2
    angle_rad = np.arctan2(last_y, last_x)  # Angle between the origin (0,0) and (x_n, y_n)
    angle_deg = np.degrees(angle_rad)  # Convert to degrees

    # Create a rotation object using scipy.spatial.transform
    rotation = R.from_euler('z', -angle_deg, degrees=True)  # Rotate by +angle_deg to align with the positive x-axis

    # Prepare the points as a 2D array (shape: [2, N])
    xy = np.vstack((merged_df['time_numeric'].values, merged_df['TimeGPT'].values))

    # Add a dummy third dimension (set to 0) to each point to make it a 3D point (x, y, 0)
    xy_3d = np.vstack((xy, np.zeros(xy.shape[1]))).T  # Shape will be (N, 3)

    # Apply rotation to the 3D points
    rotated_xy_3d = rotation.apply(xy_3d)  # Apply the rotation (shape: N, 3)

    # Extract the rotated x and y values (we only care about the x, y coordinates)
    rotated_x, rotated_y, _ = rotated_xy_3d.T  # Get the rotated x, y (third dimension is not needed)

    # Update DataFrame with rotated values
    merged_df['TimeGPT_rotated'] = rotated_y  # Use the rotated y-values

    # Return only the original 'start_time' and rotated values
    return merged_df[['start_time', 'TimeGPT_rotated']]


# Apply function to merged_df
rotated_df = align_trend_to_x_axis(merged_df)

# Plot before & after
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)  # Original Data
plt.plot(merged_df['start_time'], merged_df['TimeGPT'], label="Original", color='blue')
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel("Start Time")
plt.ylabel("TimeGPT")
plt.title("Original Time Series")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)  # Rotated Data
plt.plot(rotated_df['start_time'], rotated_df['TimeGPT_rotated'], label="Aligned", color='red')
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel("Start Time")
plt.ylabel("TimeGPT (Rotated)")
plt.title("Aligned to X-Axis")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
