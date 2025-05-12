import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob


def find_risk_score_file(results_dir, scenario_name):
    """Find the risk score file for a given scenario"""
    scenario_dir = os.path.join(results_dir, scenario_name)
    if not os.path.exists(scenario_dir):
        # Try a case-insensitive search
        for dir_name in os.listdir(results_dir):
            if dir_name.lower() == scenario_name.lower():
                scenario_dir = os.path.join(results_dir, dir_name)
                break
    
    if not os.path.exists(scenario_dir):
        print(f"Scenario directory not found: {scenario_name}")
        return None
    
    risk_files = glob.glob(os.path.join(scenario_dir, "risk_scores_*.csv"))
    if not risk_files:
        print(f"No risk score files found in {scenario_dir}")
        return None
    
    return risk_files[0]


def list_available_scenarios(results_dir):
    """List all available scenarios in the results directory"""
    scenario_dirs = [d for d in os.listdir(results_dir) 
                    if os.path.isdir(os.path.join(results_dir, d))]
    return scenario_dirs


def list_obstacles_in_scenario(risk_file):
    """List all obstacle IDs in a scenario"""
    if not os.path.exists(risk_file):
        return []
    
    df = pd.read_csv(risk_file)
    obstacle_ids = df['obstacle_id'].unique()
    return obstacle_ids


def plot_risk_score(risk_file, obstacle_id, output_dir=None, show_plot=True, custom_scenario_name=None):
    """Plot risk score for a specific obstacle ID over time"""
    if not os.path.exists(risk_file):
        print(f"Risk score file not found: {risk_file}")
        return
    
    # Read the risk score data
    df = pd.read_csv(risk_file)
    
    # Filter for the specific obstacle ID
    obstacle_data = df[df['obstacle_id'] == obstacle_id]
    
    if obstacle_data.empty:
        print(f"No data found for obstacle ID {obstacle_id}")
        obstacle_ids = df['obstacle_id'].unique()
        print(f"Available obstacle IDs: {obstacle_ids}")
        return
    
    # Extract scenario name from file path or use custom name
    if custom_scenario_name:
        scenario_name = custom_scenario_name
    else:
        scenario_name = os.path.basename(os.path.dirname(risk_file))
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot only the overall risk score
    plt.plot(obstacle_data['timestep'], obstacle_data['risk_score'], 'o-', 
             linewidth=2, markersize=8, color='blue', label='Risk Score')
    
    # Add horizontal lines for risk levels
    plt.axhline(y=1, color='green', linestyle='-', alpha=0.3, label='Low Risk')
    plt.axhline(y=2, color='orange', linestyle='-', alpha=0.3, label='Medium Risk')
    plt.axhline(y=3, color='red', linestyle='-', alpha=0.3, label='High Risk')
    
    # Set the plot limits and labels
    plt.ylim(0.5, 3.5)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Risk Score', fontsize=12)
    plt.title(f'Risk Score Over Time - Scenario: {scenario_name}, Obstacle ID: {obstacle_id}', 
              fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Save the plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'risk_plot_{scenario_name}_obstacle_{obstacle_id}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    # Show the plot
    if show_plot:
        plt.show()
    
    plt.close()


def plot_all_obstacles(risk_file, output_dir=None, show_plot=False, custom_scenario_name=None):
    """Plot risk scores for all obstacles in a scenario"""
    if not os.path.exists(risk_file):
        print(f"Risk score file not found: {risk_file}")
        return
    
    # Read the risk score data
    df = pd.read_csv(risk_file)
    
    # Get unique obstacle IDs
    obstacle_ids = df['obstacle_id'].unique()
    
    # Extract scenario name from file path or use custom name
    if custom_scenario_name:
        scenario_name = custom_scenario_name
    else:
        scenario_name = os.path.basename(os.path.dirname(risk_file))
    
    # Create a combined plot for all obstacles
    plt.figure(figsize=(12, 8))
    
    # Plot risk score for each obstacle
    for obstacle_id in obstacle_ids:
        obstacle_data = df[df['obstacle_id'] == obstacle_id]
        plt.plot(obstacle_data['timestep'], obstacle_data['risk_score'], 'o-', 
                 linewidth=2, alpha=0.7, label=f'Obstacle {obstacle_id}')
    
    # Add horizontal lines for risk levels
    plt.axhline(y=1, color='green', linestyle='-', alpha=0.3, label='Low Risk')
    plt.axhline(y=2, color='orange', linestyle='-', alpha=0.3, label='Medium Risk')
    plt.axhline(y=3, color='red', linestyle='-', alpha=0.3, label='High Risk')
    
    # Set the plot limits and labels
    plt.ylim(0.5, 3.5)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Risk Score', fontsize=12)
    plt.title(f'Risk Scores For All Obstacles - Scenario: {scenario_name}', 
              fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Save the plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'risk_plot_{scenario_name}_all_obstacles.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    # Show the plot
    if show_plot:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot risk scores for a specific obstacle over time')
    parser.add_argument('--results_dir', type=str, 
                        default='/home/yuan/ITSC/Safety/Data_collection/Riskscore_calculation/results',
                        help='Directory containing risk score results')
    parser.add_argument('--scenario', type=str, help='Scenario name to plot')
    parser.add_argument('--obstacle_id', type=int, help='Obstacle ID to plot')
    parser.add_argument('--output_dir', type=str, 
                        default='/home/yuan/ITSC/Safety/Data_collection/Riskscore_calculation/plots',
                        help='Directory to save plot images')
    parser.add_argument('--list_scenarios', action='store_true', 
                        help='List available scenarios')
    parser.add_argument('--list_obstacles', action='store_true', 
                        help='List obstacles in the specified scenario')
    parser.add_argument('--all_obstacles', action='store_true', 
                        help='Plot all obstacles in the scenario')
    parser.add_argument('--no_show', action='store_true', 
                        help='Do not display the plot window (just save to file)')
    parser.add_argument('--custom_scenario_name', type=str,
                        help='Custom scenario name to use in plot titles and filenames')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # List available scenarios if requested
    if args.list_scenarios:
        scenarios = list_available_scenarios(args.results_dir)
        print("Available scenarios:")
        for scenario in scenarios:
            print(f"  - {scenario}")
        return
    
    # Check if scenario is provided
    if not args.scenario and not args.list_scenarios:
        print("Please specify a scenario name with --scenario or use --list_scenarios")
        return
    
    # Find the risk score file for the scenario
    if args.scenario:
        risk_file = find_risk_score_file(args.results_dir, args.scenario)
        if not risk_file:
            return
    
        # List obstacles in the scenario if requested
        if args.list_obstacles:
            obstacle_ids = list_obstacles_in_scenario(risk_file)
            print(f"Obstacles in scenario {args.scenario}:")
            for obstacle_id in obstacle_ids:
                print(f"  - {obstacle_id}")
            return
        
        # Plot all obstacles if requested
        if args.all_obstacles:
            plot_all_obstacles(risk_file, args.output_dir, not args.no_show, args.custom_scenario_name)
            return
        
        # Check if obstacle ID is provided
        if not args.obstacle_id and not args.list_obstacles and not args.all_obstacles:
            print("Please specify an obstacle ID with --obstacle_id, use --list_obstacles, or use --all_obstacles")
            return
        
        # Plot the risk score for the specified obstacle
        if args.obstacle_id:
            plot_risk_score(risk_file, args.obstacle_id, args.output_dir, not args.no_show, args.custom_scenario_name)


if __name__ == "__main__":
    main() 