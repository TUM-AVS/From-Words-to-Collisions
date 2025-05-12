import pandas as pd
import numpy as np
import os

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def filter_close_obstacles(ego_df, obstacles_df, radius=30):
    """
    Filter obstacles based on proximity to the ego vehicle.
    
    Args:
        ego_df (pd.DataFrame): DataFrame containing ego vehicle trajectory
        obstacles_df (pd.DataFrame): DataFrame containing obstacle trajectories
        radius (float): Radius within which to consider obstacles (default: 30 units)
    
    Returns:
        pd.DataFrame: Filtered DataFrame containing only obstacles within radius of ego
    """
    # Create a list to store filtered data
    filtered_data = []
    
    # Get unique timesteps
    timesteps = sorted(ego_df['timestep'].unique())
    
    print(f"Processing {len(timesteps)} timesteps...")
    
    # Process each timestep
    for timestep in timesteps:
        # Get ego position for this timestep
        ego_data = ego_df[ego_df['timestep'] == timestep]
        if len(ego_data) == 0:
            continue
            
        ego_x = ego_data['x_position'].iloc[0]
        ego_y = ego_data['y_position'].iloc[0]
        
        # Get obstacles for this timestep
        timestep_obstacles = obstacles_df[obstacles_df['timestep'] == timestep]
        
        # Calculate distances for all obstacles
        timestep_obstacles['distance'] = timestep_obstacles.apply(
            lambda row: calculate_distance(ego_x, ego_y, row['x_position'], row['y_position']),
            axis=1
        )
        
        # Filter obstacles within radius
        close_obstacles = timestep_obstacles[timestep_obstacles['distance'] <= radius]
        
        # Add to filtered data
        filtered_data.append(close_obstacles)
        
        # Print progress
        if timestep % 10 == 0:
            print(f"Processed timestep {timestep}, found {len(close_obstacles)} close obstacles")
    
    # Combine all filtered data
    filtered_df = pd.concat(filtered_data, ignore_index=True)
    
    # Sort by timestep and obstacle_id
    filtered_df = filtered_df.sort_values(['timestep', 'obstacle_id'])
    
    return filtered_df

def main():
    # Configuration
    base_dir = "/home/yuan/ITSC/Safety/Generation/Metrices"
    radius = 30  # units
    
    # Get all scenario directories
    scenarios = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    print(f"Found {len(scenarios)} scenarios to process")
    
    for scenario in scenarios:
        print(f"\nProcessing scenario: {scenario}")
        
        # Construct file paths
        ego_path = os.path.join(base_dir, scenario, "ego_trajectory.csv")
        obstacles_path = os.path.join(base_dir, scenario, "dynamic_obstacles.csv")
        output_path = os.path.join(base_dir, scenario, "close_dynamic_obstacles.csv")
        
        # Check if files exist
        if not os.path.exists(ego_path) or not os.path.exists(obstacles_path):
            print(f"  Skipping {scenario}: Required files not found")
            continue
        
        try:
            # Read CSV files
            print(f"  Reading trajectory files...")
            ego_df = pd.read_csv(ego_path)
            obstacles_df = pd.read_csv(obstacles_path)
            
            # Filter obstacles
            print(f"  Filtering obstacles within {radius} units of ego vehicle...")
            filtered_df = filter_close_obstacles(ego_df, obstacles_df, radius)
            
            # Save filtered data
            print(f"  Saving filtered data...")
            filtered_df.to_csv(output_path, index=False)
            
            # Print statistics
            total_obstacles = len(obstacles_df['obstacle_id'].unique())
            filtered_obstacles = len(filtered_df['obstacle_id'].unique())
            print(f"  Original obstacles: {total_obstacles}")
            print(f"  Filtered obstacles: {filtered_obstacles}")
            print(f"  Reduction: {((total_obstacles - filtered_obstacles) / total_obstacles * 100):.1f}%")
            
        except Exception as e:
            print(f"  Error processing {scenario}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 