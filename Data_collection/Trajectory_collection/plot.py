from shapely.geometry import Polygon, Point
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def lanelets_to_polygons(lanelet_data):
    # Initialize dictionary to store polygons keyed by lanelet ID
    polygon_dict = {}

    # Iterate over each lanelet in the dataset
    for lanelet in lanelet_data.get('lanelet', []):
        # Extract the lanelet ID
        lanelet_id = lanelet['id']

        # Extract left and right bounds and parse points as tuples of floats
        left_bound_points = [(float(point['x']), float(point['y'])) for point in lanelet.get('leftBound', [])]
        right_bound_points = [(float(point['x']), float(point['y'])) for point in lanelet.get('rightBound', [])]
        
        # Check for missing boundary points
        if not left_bound_points or not right_bound_points:
            continue  # Skip this lanelet if any boundary is missing

        # Reverse the right bound to ensure the correct clockwise or counter-clockwise order
        right_bound_points.reverse()

        # Combine leftBound and reversed rightBound to form a loop
        polygon_points = left_bound_points + right_bound_points

        # Create a Shapely Polygon from the points
        try:
            polygon = Polygon(polygon_points)
            # Ensure the polygon is valid (may raise ValueError for incomplete or degenerate shapes)
            if not polygon.is_valid:
                raise ValueError(f"Polygon is invalid for lanelet {lanelet_id}")
            
            # Add the polygon to the dictionary with the corresponding ID
            polygon_dict[lanelet_id] = polygon
        except (ValueError, Exception) as e:
            print(f"Error creating a polygon for lanelet {lanelet_id}: {e}")

    # Return the dictionary of lanelet ID to Polygon mappings
    return polygon_dict

# Visualize Dynamic Obstacles
# Step 1: Extract the recorded trajectories for dynamic obstacles
def extract_obstacle_trajectories(data):
    """
    Extract the recorded trajectories for dynamic obstacles.
    Handles missing fields gracefully.
    """
    obstacle_trajectories = []

    for obstacle in data.get("dynamicObstacle", []):
        try:
            obstacle_id = obstacle.get("id", "Unknown")
            trajectory = obstacle.get("trajectory", [])
            
            if not isinstance(trajectory, list) or not trajectory:
                print(f"No recorded trajectory for obstacle {obstacle_id}")
                continue

            # Extract the trajectory data
            positions = [{"x": float(state["position"]["x"]), "y": float(state["position"]["y"])} for state in trajectory]
            times = [float(state["time"]) for state in trajectory]

            # Append data
            obstacle_trajectories.append({
                "id": obstacle_id,
                "positions": positions,
                "times": times,
            })
        except (KeyError, ValueError) as e:
            print(f"Skipping obstacle {obstacle_id} due to missing or invalid data: {e}")

    return obstacle_trajectories

# Step 2: Visualize the recorded trajectories for all dynamic obstacles
def visualize_dynamic_obstacles_with_time(obstacle_data, show_plot=True):
    # Extract the recorded trajectories of dynamic obstacles
    obstacle_trajectories = extract_obstacle_trajectories(obstacle_data)

    # Print information about the starting time and position
    for obstacle in obstacle_trajectories:
        positions = obstacle["positions"]
        times = obstacle["times"]

        # Extract x and y positions
        x_vals = [pos["x"] for pos in positions]
        y_vals = [pos["y"] for pos in positions]

        # Print the starting time and position
        #print(f"Obstacle {obstacle['id']} - Start Time: {times[0]}s, Start Position: ({x_vals[0]}, {y_vals[0]})")

    # Plot recorded trajectories if show_plot is True
    if show_plot:
        plt.figure(figsize=(10, 6))
        for obstacle in obstacle_trajectories:
            positions = obstacle["positions"]
            times = obstacle["times"]

            # Extract x and y positions
            x_vals = [pos["x"] for pos in positions]
            y_vals = [pos["y"] for pos in positions]

            # Plot trajectory
            plt.plot(x_vals, y_vals, label=f"Obstacle {obstacle['id']}")
            plt.scatter(x_vals[0], y_vals[0], label=f"Start {obstacle['id']} (t={times[0]:.2f}s)", zorder=5)

        # Improve visualization
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Dynamic Obstacles Recorded Trajectories with Time Information")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.show()

# Function to check lanelet for a single point
def check_point_in_polygons(x, y, polygons):
    """
    Check if a point (x, y) is located in any of the given polygons.
    """
    current_point = Point(x, y)
    for polygon_id, polygon in polygons.items():
        if polygon.contains(current_point):
            return polygon_id
    return None

# Function to check lanelet for a single point
def extract_ego_and_goal_data(data):
    planning_problem = data.get("planningProblem", {})
    
    # Extract ego ID
    ego_id = planning_problem.get("id", "Unknown")
    
    # Extract goal lanelet ID
    goal_lanelet_id = planning_problem.get("goalState", {}).get("position", {}).get("lanelet", "Unknown")
    
    # Extract time horizon (intervalEnd)
    interval_end = planning_problem.get("goalState", {}).get("time", {}).get("intervalEnd", 10)
    try:
        time_horizon = float(interval_end)  # Convert to float
    except (TypeError, ValueError):
        time_horizon = 10.0  # Default value if not found
    
    return {
        "ego_id": ego_id,
        "goal_lanelet_id": goal_lanelet_id,
        "time_horizon": time_horizon
    }

def calculate_orientation(x1, y1, x2, y2):
    """Calculate orientation (angle in radians) based on two consecutive points."""
    return np.arctan2(y2 - y1, x2 - x1)

# Visualize and save ego trajectory with lanelet information
def visualize_and_save_ego_trajectory_with_lanelets(
    csv_file_path, output_file_path, polygons, planning_problem_data, show_plot=True
):
    try:
        # Load CSV data
        df = pd.read_csv(csv_file_path, delimiter=';')

        # Extract necessary columns and rename them
        trajectory_data = df[[
            "trajectory_number", 
            "x_position_vehicle_m", 
            "y_position_vehicle_m", 
            "velocities_mps", 
            "accelerations_mps2"
        ]].rename(
            columns={
                "trajectory_number": "timestep",
                "x_position_vehicle_m": "x_position",
                "y_position_vehicle_m": "y_position",
                "velocities_mps": "velocity",
                "accelerations_mps2": "acceleration"
            }
        )
        
        # Round x_position and y_position to 4 decimal places
        trajectory_data["x_position"] = trajectory_data["x_position"].round(4)
        trajectory_data["y_position"] = trajectory_data["y_position"].round(4)

        # Extract the second value from velocity and acceleration columns (comma-separated lists)
        trajectory_data["velocity"] = trajectory_data["velocity"].apply(
            lambda x: round(float(x.split(",")[1]), 4)
        )
        trajectory_data["acceleration"] = trajectory_data["acceleration"].apply(
            lambda x: round(float(x.split(",")[1]), 4)
        )

        # Calculate orientation using consecutive x, y positions
        trajectory_data["orientation"] = np.nan  # Initialize orientation column
        for i in range(1, len(trajectory_data)):
            x1, y1 = trajectory_data.loc[i - 1, ["x_position", "y_position"]]
            x2, y2 = trajectory_data.loc[i, ["x_position", "y_position"]]
            trajectory_data.loc[i, "orientation"] = calculate_orientation(x1, y1, x2, y2)

        # Fill the first orientation value (optional: replicate the second orientation)
        trajectory_data.loc[0, "orientation"] = trajectory_data.loc[1, "orientation"]

        # Check lanelet membership for each position
        trajectory_data["lanelet_id"] = trajectory_data.apply(
            lambda row: check_point_in_polygons(row["x_position"], row["y_position"], polygons), axis=1
        )

        # Extract ego and goal data
        ego_goal_data = extract_ego_and_goal_data(planning_problem_data)

        # Add ego and goal data to each row
        trajectory_data["ego_id"] = ego_goal_data["ego_id"]
        trajectory_data["goal_lanelet_id"] = ego_goal_data["goal_lanelet_id"]
        trajectory_data["time_horizon"] = ego_goal_data["time_horizon"]

        # Reorder columns
        trajectory_data = trajectory_data[
            ["ego_id", "timestep", "x_position", "y_position", "orientation","velocity", "acceleration", 
            "lanelet_id", "goal_lanelet_id", "time_horizon"]
        ]

        # Save trajectory data to a new CSV file
        trajectory_data.to_csv(output_file_path, index=False)
        print(f"Trajectory data with orientation saved to {output_file_path}")

        # Plot the trajectory if show_plot is True
        if show_plot:
            x_positions = trajectory_data["x_position"].values
            y_positions = trajectory_data["y_position"].values
            plt.figure(figsize=(12, 8))
            plt.plot(x_positions, y_positions, label="Ego Vehicle Trajectory", linewidth=2, color="blue")
            plt.scatter(x_positions[0], y_positions[0], color="green", label="Ego Start", zorder=5)
            plt.scatter(x_positions[-1], y_positions[-1], color="red", label="Ego End", zorder=5)

            # Optionally plot polygons (lanelets)
            for polygon in polygons:
                plt.fill(*zip(*polygon), alpha=0.3, label="Lanelet Polygon")

            # Improve visualization
            plt.xlabel("X Position (m)")
            plt.ylabel("Y Position (m)")
            plt.title("Ego Vehicle Trajectory with Orientation")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            plt.show()

        return trajectory_data

    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
        return None
    except KeyError as e:
        print(f"Missing column in the data: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

