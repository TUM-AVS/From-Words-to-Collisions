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


def plot_compare_obstacles(scenario1, scenario2, obstacle_id, max_timestep=None, min_timestep=None,
                          output_dir=None, save_plot=True, show_plot=True,
                          results_dir="/home/yuan/ITSC/Safety/Data_collection/Riskscore_calculation/results"):
    """
    Plot and compare risk scores for the same obstacle ID from two different scenarios
    """
    risk_file1 = find_risk_score_file(results_dir, scenario1)
    risk_file2 = find_risk_score_file(results_dir, scenario2)
    
    if not risk_file1:
        print(f"Could not find risk scores for scenario: {scenario1}")
        return
    if not risk_file2:
        print(f"Could not find risk scores for scenario: {scenario2}")
        return
    
    df1 = pd.read_csv(risk_file1)
    df2 = pd.read_csv(risk_file2)
    
    obstacle_data1 = df1[df1['obstacle_id'] == obstacle_id]
    obstacle_data2 = df2[df2['obstacle_id'] == obstacle_id]
    
    if obstacle_data1.empty:
        print(f"No data found for obstacle ID {obstacle_id} in {scenario1}")
        print(f"Available obstacle IDs: {df1['obstacle_id'].unique()}")
        return
    if obstacle_data2.empty:
        print(f"No data found for obstacle ID {obstacle_id} in {scenario2}")
        print(f"Available obstacle IDs: {df2['obstacle_id'].unique()}")
        return
    
    if max_timestep is not None:
        obstacle_data1 = obstacle_data1[obstacle_data1['timestep'] <= max_timestep]
        obstacle_data2 = obstacle_data2[obstacle_data2['timestep'] <= max_timestep]
    if min_timestep is not None:
        obstacle_data1 = obstacle_data1[obstacle_data1['timestep'] >= min_timestep]
        obstacle_data2 = obstacle_data2[obstacle_data2['timestep'] >= min_timestep]
    
    max_timestep1 = obstacle_data1['timestep'].max() if not obstacle_data1.empty else 0
    max_timestep2 = obstacle_data2['timestep'].max() if not obstacle_data2.empty else 0
    overall_max_timestep = max(max_timestep1, max_timestep2)
    
    plt.figure(figsize=(9.2, 6))

    # ✅ Updated colors and simplified legend
    plt.plot(obstacle_data1['timestep'], obstacle_data1['risk_score'], 'o-', 
             linewidth=2, markersize=8, color='green', 
             label=f'{scenario1} (Original safe scenario)')
    
    plt.plot(obstacle_data2['timestep'], obstacle_data2['risk_score'], 's-', 
             linewidth=2, markersize=8, color='red', 
             label=f'{scenario1}_New (Generated safety-critical scenario)')
    
    plt.ylim(-0.5, 5.5)
    plt.xlim(0, overall_max_timestep + 0)

    x_ticks = np.arange(0, overall_max_timestep + 1)
    plt.xticks(x_ticks, fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.xlabel('Timestep', fontsize=16)
    plt.ylabel('Risk Score (0-5)', fontsize=16)
    plt.title('Ego attacker Risk Score Comparison', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=14)
    
    if save_plot and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'risk_comparison_obstacle_{obstacle_id}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    if show_plot:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare risk scores for the same obstacle ID from two scenarios')
    parser.add_argument('--results_dir', type=str, 
                        default='/home/yuan/ITSC/Safety/Data_collection/Riskscore_calculation/results',
                        help='Directory containing risk score results')
    parser.add_argument('--scenario1', type=str, 
                        default='BEL_Antwerp-1_14_T-1',
                        help='First scenario name')
    parser.add_argument('--scenario2', type=str, 
                        default='BEL_Antwerp-1_14_T-1n',
                        help='Second scenario name')
    parser.add_argument('--obstacle_id', type=int, 
                        default=30762,
                        help='Obstacle ID to plot')
    parser.add_argument('--max_timestep', type=int, default=16,
                        help='Maximum timestep to plot')
    parser.add_argument('--min_timestep', type=int, default=None,
                        help='Minimum timestep to plot')
    parser.add_argument('--output_dir', type=str, 
                        default='/home/yuan/ITSC/Safety/Data_collection/Riskscore_calculation/plots',
                        help='Directory to save plot images')
    parser.add_argument('--no_save', action='store_true', 
                        help='Do not save the plot to a file')
    parser.add_argument('--no_show', action='store_true', 
                        help='Do not display the plot window')
    
    args = parser.parse_args()
    
    plot_compare_obstacles(
        args.scenario1, 
        args.scenario2,
        args.obstacle_id,
        args.max_timestep,
        args.min_timestep,
        args.output_dir,
        not args.no_save,
        not args.no_show,
        args.results_dir
    )


if __name__ == "__main__":
    main()
