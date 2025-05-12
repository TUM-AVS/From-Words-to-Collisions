from xml2json_all import convert_single_xml_to_json
from layermodel import extract_important_information, assign_layers
from plot import lanelets_to_polygons, visualize_dynamic_obstacles_with_time, visualize_and_save_ego_trajectory_with_lanelets
from extract_trajectories import dynamic_obstacles_with_lanelets, extract_every_nth_timestep, extract_ego_every_nth_timestep
import os

def main(scenario_name):
    # Define the source and destination directories
    source_dir = '/home/yuan/ITSC/Safety/Generation/BEL_Antwerp-1_14_T-1n'
    destination_dir = '/home/yuan/ITSC/Safety/json_scenarios'
    log_dir= '/home/yuan/ITSC/Safety/Generation'
    output_dir = '/home/yuan/ITSC/Safety/Generation/Metrices'

    # Define the path to the XML file and the destination JSON file
    xml_file_path = f'{source_dir}/{scenario_name}.xml'

    # Convert the single XML file to JSON
    convert_single_xml_to_json(xml_file_path, destination_dir)

    # Load the JSON scenario file
    json_file_path = f'{destination_dir}/{scenario_name}.json'

    # Extract important information from the JSON file
    important_info= extract_important_information(json_file_path)

    # Assuming assign_layers is a function that processes the important_info
    # and returns a dictionary with the layers
    layers = assign_layers(important_info)

    # Extract individual layers
    L1 = layers["L1_RoadLevel"]
    L2 = layers["L2_TrafficInfrastructure"]
    L3 = layers["L3_TemporalModifications"]
    L4 = layers["L4_MovableObjects"]
    L5 = layers["L5_EnvironmentalConditions"]
    L6 = layers["L6_DigitalInformation"]
    L7 = layers["L7_PlanningProblem"]

    # Convert lanelets to polygons
    polygons = lanelets_to_polygons(L1)

    # Visualize dynamic obstacles with time
    visualize_dynamic_obstacles_with_time(L4,show_plot=False)

    # File paths
    input_csv_path = f'{log_dir}/{scenario_name}/logs.csv'

    # Ensure the output directory exists
    scenario_output_dir = f'{output_dir}/{scenario_name}'
    os.makedirs(scenario_output_dir, exist_ok=True)
    output_csv_path = f'{output_dir}/{scenario_name}/ego_trajectory_positions_with_lanelets.csv'

    # Execute the function
    trajectory_data_with_lanelets = visualize_and_save_ego_trajectory_with_lanelets(input_csv_path, output_csv_path, polygons, L7,show_plot=False)
    # Assuming the first element of the tuple is the DataFrame

    obstacle_csv_path = f'{output_dir}/{scenario_name}/dynamic_obstacles_with_lanelets.csv'
    dynamic_obstacles_df = dynamic_obstacles_with_lanelets(L4, polygons, obstacle_csv_path)

    # Example usage
    obstacles_path = f'{output_dir}/{scenario_name}/dynamic_obstacles.csv'

    filtered_data = extract_every_nth_timestep(obstacle_csv_path, obstacles_path,n=1)

    # File paths
    ego_path = f'{output_dir}/{scenario_name}/ego_trajectory.csv'

    # Execute the function
    filtered_ego_data = extract_ego_every_nth_timestep(output_csv_path, ego_path,n=1)

if __name__ == "__main__":
    scenario_name = 'BEL_Antwerp-1_14_T-1n'
    main(scenario_name)