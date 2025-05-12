import os
import pandas as pd
import numpy as np
import glob
import argparse


def get_dominant_weight(relative_direction):
    """
    Determine weight based on relative direction
    - Front/Back: wdominant = 1
    - Left/Right: wdominant = 0
    - Other directions: wdominant = 0.5
    """
    direction = relative_direction.lower()
    
    if direction == 'front' or direction == 'rear':
        return 1.0
    elif direction == 'left' or direction == 'right':
        return 0.0
    else:
        return 0.5


def calculate_distance_risk(d):
    """
    Calculate risk score based on absolute distance
    d: distance in meters (positive or negative)
    Returns: risk score (0-5)
    """
    d_abs = abs(d)
    if d_abs < 0.3:  # Collision risk within 0.3m
        return 0
    elif d_abs < 0.8:
        return 1
    elif d_abs < 1.3:
        return 2
    elif d_abs < 3:
        return 3
    elif d_abs < 5:
        return 4
    else:
        return 5  # Negligible risk beyond 2m


def calculate_ttc_risk(ttc):
    """
    Calculate risk score based on TTC using the correct 0-5 scale
    - Collision Risk (0): TTC = 0
    - Extreme Risk (1): 0 < TTC ≤ 0.5
    - High Risk (2): 0.5 < TTC ≤ 1
    - Medium Risk (3): 1 < TTC ≤ 3
    - Low Risk (4): 3 < TTC ≤ 5
    - Negligible Risk (5): TTC > 5 or TTC = inf
    """
    if ttc < 0.15:
        return 0  # Collision Risk
    elif pd.isna(ttc) or ttc == float('inf'):
        return 5  # Negligible Risk (no collision projected)
    elif ttc <= 0.65:
        return 1  # Extreme Risk
    elif ttc <= 1.15:
        return 2  # High Risk
    elif ttc <= 3:
        return 3  # Medium Risk
    elif ttc <= 5:
        return 4  # Low Risk
    else:
        return 5  # Negligible Risk


def calculate_risk_scores(df):
    """
    Calculate risk scores for each row in the dataframe
    """
    # Create new columns for scores
    df['wdominant'] = df['relative_direction'].apply(get_dominant_weight)
    
    # Calculate distance risk scores
    df['long_dsc'] = df['adjusted_d_long'].apply(calculate_distance_risk)
    df['lat_dsc'] = df['adjusted_d_lat'].apply(calculate_distance_risk)
    
    # Calculate weighted distance score
    df['dsc'] = df['long_dsc'] * df['wdominant'] + df['lat_dsc'] * (1 - df['wdominant'])
    
    # Calculate TTC risk scores
    df['long_tsc'] = df['ttc_long'].apply(calculate_ttc_risk)
    df['lat_tsc'] = df['ttc_lat'].apply(calculate_ttc_risk)
    
    # Calculate weighted TTC score
    df['tsc'] = df['long_tsc'] * df['wdominant'] + df['lat_tsc'] * (1 - df['wdominant'])
    
    # Calculate overall risk score using int() to truncate decimal values
    df['risk_score'] = df.apply(lambda row: int(calculate_overall_risk(row['dsc'], row['tsc'])), axis=1)
    
    return df


def process_file(csv_path, output_dir=None, scenario_name=None):
    """
    Process a single CSV file and calculate risk scores
    
    Parameters:
        csv_path (str): Path to the input CSV file
        output_dir (str): Directory to save results (optional)
        scenario_name (str): Custom scenario name to use (optional)
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Calculate risk scores
        df_with_scores = calculate_risk_scores(df)
        
        # Create output path
        if output_dir:
            filename = os.path.basename(csv_path)
            # Use custom scenario name if provided, otherwise extract from directory
            if scenario_name:
                out_dir = os.path.join(output_dir, scenario_name)
            else:
                scenario_dir = os.path.basename(os.path.dirname(csv_path))
                out_dir = os.path.join(output_dir, scenario_dir)
            
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, f"risk_scores_{filename}")
        else:
            output_path = csv_path.replace('.csv', '_risk_scores.csv')
        
        # Save results
        df_with_scores.to_csv(output_path, index=False)
        print(f"Processed {csv_path} -> {output_path}")
        
        return df_with_scores
    
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None


def process_directory(metrics_dir, output_dir=None, scenario_name=None):
    """
    Process all scenario directories in the metrics directory
    
    Parameters:
        metrics_dir (str): Directory containing metrics files
        output_dir (str): Directory to save results (optional)
        scenario_name (str): Custom scenario name to use for all files (optional)
    """
    # Find all close_relative_metrics.csv files
    pattern = os.path.join(metrics_dir, '*', 'close_relative_metrics.csv')
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No matching files found with pattern: {pattern}")
        return
    
    print(f"Found {len(csv_files)} files to process")
    
    # Process each file
    for csv_file in csv_files:
        process_file(csv_file, output_dir, scenario_name)


def calculate_scenario_risk(df_with_scores):
    """
    Calculate overall risk for a scenario based on maximum risk at each timestep
    """
    # Group by timestep and get minimum risk score for each (lower score = higher risk)
    timestep_min_risk = df_with_scores.groupby('timestep')['risk_score'].min().reset_index()
    
    # Overall scenario risk is the minimum risk score across all timesteps
    scenario_risk = timestep_min_risk['risk_score'].min()
    
    return timestep_min_risk, scenario_risk


def generate_summary_report(metrics_dir, output_dir):
    """
    Generate a summary report of risk scores for all scenarios
    """
    pattern = os.path.join(metrics_dir, '*', 'close_relative_metrics.csv')
    csv_files = glob.glob(pattern)
    
    summary_data = []
    
    for csv_file in csv_files:
        scenario = os.path.basename(os.path.dirname(csv_file))
        df = pd.read_csv(csv_file)
        df_with_scores = calculate_risk_scores(df)
        
        _, scenario_risk = calculate_scenario_risk(df_with_scores)
        
        summary_data.append({
            'scenario': scenario,
            'min_risk_score': scenario_risk,  # Lower score = higher risk
            'num_obstacles': df['obstacle_id'].nunique(),
            'num_timesteps': df['timestep'].nunique()
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.sort_values('min_risk_score', ascending=True, inplace=True)  # Sort by risk (ascending)
        
        summary_path = os.path.join(output_dir, 'risk_score_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary report saved to {summary_path}")


def calculate_overall_risk(dsc, tsc):
    """
    Calculate overall risk score based on distance and time risk scores
    
    Parameters:
        dsc (int): Distance-based safety score (0-5)
        tsc (int): Time-based safety score (0-5)
        
    Returns:
        float: Overall risk score (0-5)
    """
    # Calculate the average of distance and time risk scores
    return (dsc + tsc) / 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate risk scores from metrics files')
    parser.add_argument('--metrics_dir', type=str, default='/home/yuan/ITSC/Safety/Generation/Metrices',
                        help='Directory containing metrics files')
    parser.add_argument('--output_dir', type=str, default='/home/yuan/ITSC/Safety/Data_collection/Riskscore_calculation/results',
                        help='Directory to save output files')
    parser.add_argument('--scenario_name', type=str,
                        help='Custom scenario name to use in output path')
    parser.add_argument('--summary', action='store_true',
                        help='Generate summary report of all scenarios')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all files
    process_directory(args.metrics_dir, args.output_dir, args.scenario_name)
    
    # Generate summary if requested
    if args.summary:
        generate_summary_report(args.metrics_dir, args.output_dir) 