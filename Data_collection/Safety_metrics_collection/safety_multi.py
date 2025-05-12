import pandas as pd
import math
import csv
import json
import os
import concurrent.futures

def calculate_relative_distances(x_ego, y_ego, theta_ego, x_obs, y_obs, theta_obs):
    delta_x = x_obs - x_ego
    delta_y = y_obs - y_ego
    d_long = delta_x * math.cos(-theta_ego) - delta_y * math.sin(-theta_ego)
    d_lat = delta_x * math.sin(-theta_ego) + delta_y * math.cos(-theta_ego)
    return d_long, d_lat

def calculate_adjusted_relative_distances(d_long, d_lat):
    vehicle_length = 4.508
    vehicle_width = 1.610

    adjusted_d_long = d_long - vehicle_length if d_long > vehicle_length else (d_long + vehicle_length if d_long < -vehicle_length else 0)
    adjusted_d_lat = d_lat - vehicle_width if d_lat > vehicle_width else (d_lat + vehicle_width if d_lat < -vehicle_width else 0)

    return adjusted_d_long, adjusted_d_lat

def calculate_relative_velocity(v_ego, theta_ego, v_obs, theta_obs):
    v_obs_long = v_obs * math.cos(theta_obs - theta_ego)
    v_obs_lat = v_obs * math.sin(theta_obs - theta_ego)
    v_rel_long = v_obs_long - v_ego
    v_rel_lat = v_obs_lat
    return v_rel_long, v_rel_lat

def calculate_relative_acceleration(a_ego, theta_ego, a_obs, theta_obs):
    a_obs_long = a_obs * math.cos(theta_obs)
    a_obs_lat = a_obs * math.sin(theta_obs)
    a_rel_long = a_obs_long - a_ego * math.cos(theta_ego)
    a_rel_lat = a_obs_lat - a_ego * math.sin(theta_ego)
    return a_rel_long, a_rel_lat

def identify_relative_direction(d_long, d_lat):
    ego_length = 4.508
    ego_width = 1.610

    if d_long > ego_length:
        if d_lat > ego_width:
            return "Front-left"
        elif d_lat < -ego_width:
            return "Front-right"
        else:
            return "Front"
    elif d_long < -ego_length:
        if d_lat > ego_width:
            return "Rear-left"
        elif d_lat < -ego_width:
            return "Rear-right"
        else:
            return "Behind"

    if d_lat > ego_width:
        return "Left"
    elif d_lat < -ego_width:
        return "Right"

    return "Collision"

def calculate_time_to_collision(adjusted_d_long, adjusted_d_lat, v_rel_long, v_rel_lat, relative_direction):
    """
    Calculate Time to Collision (TTC) based on relative distances and velocities.
    Determine the motion of the obstacle relative to the ego vehicle.
    
    :param adjusted_d_long: Longitudinal distance to the obstacle
    :param adjusted_d_lat: Lateral distance to the obstacle
    :param v_rel_long: Relative longitudinal velocity of the obstacle
    :param v_rel_lat: Relative lateral velocity of the obstacle
    :param relative_direction: Relative direction of the obstacle (e.g., Front, Rear-Left)
    :return: Tuple (ttc_long, ttc_lat, motion_description)
    """

    # Initialize motion description
    motion_description = ""

    # Longitudinal TTC calculation
    if relative_direction in ["Front", "Front-left", "Front-right"]:
        if v_rel_long > 0:  # Obstacle moving away
            ttc_long = float('inf')
            motion_description = "Obstacle is moving away longitudinally."
        elif v_rel_long < 0:  # Obstacle approaching
            ttc_long = adjusted_d_long / abs(v_rel_long)
            motion_description = "Obstacle is driving toward the ego car longitudinally."
        else:  # No relative motion
            ttc_long = float('inf')
            motion_description = "No longitudinal relative motion."

    elif relative_direction in ["Behind", "Rear-left", "Rear-right"]:
        if v_rel_long > 0:  # Obstacle catching up
            ttc_long = abs(adjusted_d_long) / v_rel_long
            motion_description = "Obstacle is driving toward the ego car from behind."
        elif v_rel_long < 0:  # Obstacle moving away
            ttc_long = float('inf')
            motion_description = "Obstacle is moving away longitudinally."
        else:  # No relative motion
            ttc_long = float('inf')
            motion_description = "No longitudinal relative motion."

    else:  # Exact alignment or unknown case
        ttc_long = 0
        motion_description = "Exact longitudinal alignment or co."

    # Lateral TTC calculation
    if relative_direction in ["Left", "Front-left", "Rear-left"]:
        if v_rel_lat > 0:  # Obstacle moving away laterally
            ttc_lat = float('inf')
            motion_description += " Obstacle is moving away laterally to the left."
        elif v_rel_lat < 0:  # Obstacle approaching laterally
            ttc_lat = abs(adjusted_d_lat / v_rel_lat)
            motion_description += " Obstacle is driving toward the ego car laterally from the left."
        else:  # No relative motion
            ttc_lat = float('inf')
            motion_description += " No lateral relative motion."

    elif relative_direction in ["Right", "Front-right", "Rear-right"]:
        if v_rel_lat > 0:  # Obstacle approaching laterally from the right
            ttc_lat = abs(adjusted_d_lat / v_rel_lat)
            motion_description += " Obstacle is driving toward the ego car laterally from the right."
        elif v_rel_lat < 0:  # Obstacle moving away laterally
            ttc_lat = float('inf')
            motion_description += " Obstacle is moving away laterally to the right."
        else:  # No relative motion
            ttc_lat = float('inf')
            motion_description += " No lateral relative motion."

    else:  # Exact alignment or unknown case
        ttc_lat = 0
        motion_description += " Exact lateral alignment or unknown case."

    return ttc_long, ttc_lat, motion_description

# Processing a single scenario
def process_single_scenario(base_dir, scenario_name):
    ego_path = f"{base_dir}/{scenario_name}/ego_trajectory.csv"
    obstacles_path = f"{base_dir}/{scenario_name}/dynamic_obstacles.csv"

    if not os.path.exists(ego_path) or not os.path.exists(obstacles_path):
        print(f"Missing files for scenario: {scenario_name}")
        return

    ego_df = pd.read_csv(ego_path)
    obstacles_df = pd.read_csv(obstacles_path)
    results = []

    for _, ego_row in ego_df.iterrows():
        timestep = ego_row['timestep']
        ego_data = ego_row
        obs_at_timestep = obstacles_df[obstacles_df['timestep'] == timestep]

        for _, obs_row in obs_at_timestep.iterrows():
            d_long, d_lat = calculate_relative_distances(
                ego_data['x_position'], ego_data['y_position'], ego_data['orientation'],
                obs_row['x_position'], obs_row['y_position'], obs_row['orientation']
            )
            adjusted_d_long, adjusted_d_lat = calculate_adjusted_relative_distances(d_long, d_lat)
            v_rel_long, v_rel_lat = calculate_relative_velocity(
                ego_data['velocity'], ego_data['orientation'],
                obs_row['velocity'], obs_row['orientation']
            )
            a_rel_long, a_rel_lat = calculate_relative_acceleration(
                ego_data['acceleration'], ego_data['orientation'],
                obs_row['acceleration'], obs_row['orientation']
            )
            relative_direction = identify_relative_direction(d_long, d_lat)
            ttc_long, ttc_lat, motion_description = calculate_time_to_collision(adjusted_d_long, adjusted_d_lat, v_rel_long, v_rel_lat, relative_direction)

            # Append rounded results
            results.append({
                "timestep": timestep,
                "obstacle_id": obs_row['obstacle_id'],
                "relative_direction": relative_direction,
                "d_long": round(d_long, 2),
                "d_lat": round(d_lat, 2),
                "adjusted_d_long": round(adjusted_d_long, 2),
                "adjusted_d_lat": round(adjusted_d_lat, 2),
                "v_rel_long": round(v_rel_long, 2),
                "v_rel_lat": round(v_rel_lat, 2),
                "a_rel_long": round(a_rel_long, 2),
                "a_rel_lat": round(a_rel_lat, 2),
                "ttc_long": round(ttc_long, 2) if ttc_long != float('inf') else ttc_long,  # Handle infinity
                "ttc_lat": round(ttc_lat, 2) if ttc_lat != float('inf') else ttc_lat,  # Handle infinity
                "motion_description": motion_description  # Keep string as-is
            })
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    # Input and output file paths
    csv_file = f"{base_dir}/{scenario_name}/relative_metrics.csv"
    # Save DataFrame to CSV
    results_df.to_csv(csv_file, index=False)
    print(f"Results saved to: {csv_file}")


    txt_file = f"{base_dir}/{scenario_name}/output.txt"

    # Read the CSV and transform it into the required JSON structure
    output_data = {}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            timestep = float(row['timestep']) * 0.1  # Convert timestep to real time
            obstacle_id = int(float(row['obstacle_id']))
            relative_direction = row['relative_direction']
            adjusted_d_long  = abs(float(row['adjusted_d_long']))
            adjusted_d_lat = abs(float(row['adjusted_d_lat']))
            ttc_long = float(row['ttc_long']) if row['ttc_long'] != 'inf' else 'Infinity'
            ttc_lat = float(row['ttc_lat']) if row['ttc_lat'] != 'inf' else 'Infinity'
            motion_description = row['motion_description']  

            if f"At {timestep:.1f} seconds" not in output_data:
                output_data[f"At {timestep:.1f} seconds"] = {}

            output_data[f"At {timestep:.1f} seconds"][f"Obstacle {obstacle_id}"] = {
                "Relative Direction": relative_direction,
                "Distance to Collision": {
                    "Longitudinal": adjusted_d_long,
                    "Lateral": adjusted_d_lat
                },

                "Time to Collision": {
                    "Longitudinal": ttc_long,
                    "Lateral": ttc_lat
                },
                "Motion Description": motion_description
            }

    # Write the JSON structure to a txt file
    with open(txt_file, 'w') as file:
        json.dump(output_data, file, indent=4)

    print(f"Data successfully written to {txt_file}.")

# Main function
def main():
    CONFIG = {
        "base_dir": "/home/yuan/ITSC/Safety/output_validation_normal",
        "process_all": True,
        "scenarios": ["ESP_Barcelona-29_33_T-1"],
        "num_cpus": 12
    }

    base_dir = CONFIG["base_dir"]
    process_all = CONFIG["process_all"]
    scenarios = CONFIG["scenarios"]
    num_cpus = CONFIG["num_cpus"]

    if process_all:
        scenarios = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpus) as executor:
        executor.map(lambda scenario: process_single_scenario(base_dir, scenario), scenarios)

if __name__ == "__main__":
    main()