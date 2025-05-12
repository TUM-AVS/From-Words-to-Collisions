import pandas as pd
import os

def extract_close_metrics(scenario_dir):
    """
    Extract relative metrics for obstacles that are in close_dynamic_obstacles.csv.
    
    Args:
        scenario_dir (str): Path to the scenario directory
    """
    # Construct file paths
    close_obstacles_path = os.path.join(scenario_dir, "close_dynamic_obstacles.csv")
    relative_metrics_path = os.path.join(scenario_dir, "relative_metrics.csv")
    output_path = os.path.join(scenario_dir, "close_relative_metrics.csv")
    
    # Check if required files exist
    if not os.path.exists(close_obstacles_path):
        print(f"  Skipping: close_dynamic_obstacles.csv not found")
        return
    if not os.path.exists(relative_metrics_path):
        print(f"  Skipping: relative_metrics.csv not found")
        return
    
    try:
        # Read the CSV files
        print(f"  Reading CSV files...")
        close_obstacles_df = pd.read_csv(close_obstacles_path)
        relative_metrics_df = pd.read_csv(relative_metrics_path)
        
        # Print initial data info
        print(f"  Close obstacles shape: {close_obstacles_df.shape}")
        print(f"  Relative metrics shape: {relative_metrics_df.shape}")
        
        # Print sample data
        print(f"  Sample close obstacles (first 3 rows):")
        print(close_obstacles_df[['timestep', 'obstacle_id']].head(3))
        print(f"  Sample relative metrics (first 3 rows):")
        print(relative_metrics_df[['timestep', 'obstacle_id']].head(3))
        
        # Convert obstacle_id to integer in both dataframes
        close_obstacles_df['obstacle_id'] = close_obstacles_df['obstacle_id'].astype(int)
        relative_metrics_df['obstacle_id'] = relative_metrics_df['obstacle_id'].astype(int)
        
        # Convert timestep to integer in both dataframes
        close_obstacles_df['timestep'] = close_obstacles_df['timestep'].astype(int)
        relative_metrics_df['timestep'] = relative_metrics_df['timestep'].astype(int)
        
        # Get unique obstacle IDs and timesteps from close_obstacles.csv
        close_obstacles = close_obstacles_df[['timestep', 'obstacle_id']].drop_duplicates()
        print(f"  Unique close obstacles: {len(close_obstacles)}")
        
        # Create a key for merging
        close_obstacles['key'] = close_obstacles['timestep'].astype(str) + '_' + close_obstacles['obstacle_id'].astype(str)
        relative_metrics_df['key'] = relative_metrics_df['timestep'].astype(str) + '_' + relative_metrics_df['obstacle_id'].astype(str)
        
        # Print sample keys
        print(f"  Sample close obstacle keys (first 3):")
        print(close_obstacles['key'].head(3).tolist())
        print(f"  Sample relative metrics keys (first 3):")
        print(relative_metrics_df['key'].head(3).tolist())
        
        # Check if any keys match
        matching_keys = set(close_obstacles['key']).intersection(set(relative_metrics_df['key']))
        print(f"  Number of matching keys: {len(matching_keys)}")
        if len(matching_keys) > 0:
            print(f"  Sample matching keys: {list(matching_keys)[:3]}")
        
        # Filter relative metrics to only include close obstacles
        close_metrics = relative_metrics_df[relative_metrics_df['key'].isin(close_obstacles['key'])]
        
        # Remove the temporary key column
        close_metrics = close_metrics.drop('key', axis=1)
        
        # Save filtered data
        print(f"  Saving filtered metrics...")
        close_metrics.to_csv(output_path, index=False)
        
        # Print statistics
        total_metrics = len(relative_metrics_df)
        filtered_metrics = len(close_metrics)
        print(f"  Original metrics entries: {total_metrics}")
        print(f"  Filtered metrics entries: {filtered_metrics}")
        print(f"  Reduction: {((total_metrics - filtered_metrics) / total_metrics * 100):.1f}%")
        
    except Exception as e:
        print(f"  Error processing metrics: {str(e)}")

def main():
    # Configuration
    base_dir = "/home/yuan/ITSC/Safety/Generation/Metrices"
    
    # Get all scenario directories
    scenarios = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    print(f"Found {len(scenarios)} scenarios to process")
    
    for scenario in scenarios:
        print(f"\nProcessing scenario: {scenario}")
        scenario_dir = os.path.join(base_dir, scenario)
        extract_close_metrics(scenario_dir)

if __name__ == "__main__":
    main() 