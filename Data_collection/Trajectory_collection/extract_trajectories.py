import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point, Polygon
from plot import check_point_in_polygons

def extract_obstacle_trajectories(obstacle_data):
    """
    Extract trajectories from dynamic obstacles data, including velocity, acceleration and orientation.
    """
    trajectories = []
    for obstacle in obstacle_data.get("dynamicObstacle", []):
        obstacle_id = obstacle.get("id", "Unknown")
        trajectory_points = obstacle.get("trajectory", [])
        positions = [{"x": point["position"]["x"], "y": point["position"]["y"]} for point in trajectory_points]
        orientation = [point["orientation"] for point in trajectory_points]
        times = [point["time"] for point in trajectory_points]
        velocities = [point["velocity"] for point in trajectory_points]
        accelerations = [point["acceleration"] for point in trajectory_points]

        trajectories.append({
            "id": obstacle_id,
            "positions": positions,
            "orientation": orientation,
            "times": times,
            "velocities": velocities,
            "accelerations": accelerations
        })

    return trajectories

# Updated dynamic obstacles visualization with lanelet checking
def dynamic_obstacles_with_lanelets(obstacle_data, polygons, output_file_path):
    """
    Process dynamic obstacles' trajectories, check lanelet membership, and save results.
    """
    # Extract the recorded trajectories of dynamic obstacles
    obstacle_trajectories = extract_obstacle_trajectories(obstacle_data)
    trajectory_data = []

    # Process each obstacle's trajectory
    for obstacle in obstacle_trajectories:
        obstacle_id = obstacle["id"]
        positions = obstacle["positions"]
        orientation = obstacle["orientation"]
        times = obstacle["times"]
        velocities = obstacle["velocities"]
        accelerations = obstacle["accelerations"]
        

        for time, pos, vel, acc, theta in zip(times, positions, velocities, accelerations, orientation):
            x, y = pos["x"], pos["y"]
            lanelet_id = check_point_in_polygons(x, y, polygons)
            trajectory_data.append({
                "obstacle_id": obstacle_id,
                "timestep": time,
                "x_position": x,
                "y_position": y,
                "orientation": theta,
                "velocity": vel,
                "acceleration": acc,
                "lanelet_id": lanelet_id
            })

    # Save trajectory data to a new CSV file
    trajectory_df = pd.DataFrame(trajectory_data)
    trajectory_df.to_csv(output_file_path, index=False)
    print(f"Trajectory data saved to {output_file_path}")

    return trajectory_df

def extract_every_nth_timestep(input_csv_path, output_csv_path, n):
    """
    Extract every nth timestep (n, 2n, 3n, ...) for each obstacle_id and consolidate 
    the data for all obstacles into a single DataFrame for the same timesteps.
    Save the resulting DataFrame to a new CSV file.

    Parameters:
    - input_csv_path: str, path to the input CSV file containing obstacle data
    - output_csv_path: str, path to save the output CSV file
    - n: int, the step interval to filter timesteps
    """
    # Load the CSV data
    df = pd.read_csv(input_csv_path)
    # Replace NaN in lanelet_id with -1
    df['lanelet_id'] = df['lanelet_id'].fillna(-1).astype(int)

    # Ensure the timestep column is numeric
    df['timestep'] = pd.to_numeric(df['timestep'])

    # Filter every nth timestep
    filtered_df = df[df['timestep'] % n == 0]

    # Sort the data by timestep and obstacle_id
    filtered_df = filtered_df.sort_values(by=['timestep', 'obstacle_id'])

    # Save the filtered data to a new CSV file
    filtered_df.to_csv(output_csv_path, index=False)
    print(f"Filtered data saved to {output_csv_path}")

    return filtered_df

def extract_ego_every_nth_timestep(input_csv_path, output_csv_path, n):
    """
    Extract every nth timestep (n, 2n, 3n, ...) from the ego car trajectory data.
    Save the resulting data to a new CSV file.

    Parameters:
    - input_csv_path: str, path to the input CSV file containing ego trajectory data
    - output_csv_path: str, path to save the output CSV file
    - n: int, the step interval to filter timesteps
    """
    # Load the CSV data
    df = pd.read_csv(input_csv_path)

    # Ensure the timestep column is numeric
    df['timestep'] = pd.to_numeric(df['timestep'])

    # Filter every nth timestep
    filtered_df = df[df['timestep'] % n == 0]

    # Sort by timestep
    filtered_df = filtered_df.sort_values(by='timestep')

    # Save the filtered data to a new CSV file
    filtered_df.to_csv(output_csv_path, index=False)
    print(f"Ego car data for every {n}th timestep saved to {output_csv_path}")

    return filtered_df

