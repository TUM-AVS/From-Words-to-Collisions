import os
import shutil
import multiprocessing
import csv
import logging
from tqdm import tqdm
from xml2json_all import convert_single_xml_to_json
from layermodel import extract_important_information, assign_layers
from plot import lanelets_to_polygons, visualize_dynamic_obstacles_with_time, visualize_and_save_ego_trajectory_with_lanelets
from extract_trajectories import dynamic_obstacles_with_lanelets, extract_every_nth_timestep, extract_ego_every_nth_timestep
import cProfile

# Set up logging to output to a file and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.StreamHandler(),
    logging.FileHandler("process.log")
])

def clean_output_directory(output_dir):
    """Clean the output directory by removing its contents."""
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        logging.info(f"Cleaned output directory: {output_dir}")

def process_scenario(scenario_name, source_dir, destination_dir, log_dir, output_dir):
    """Process a single scenario, extracting data and performing visualizations"""
    result = [scenario_name, 'skipped']  # Default result is skipped
    try:
        # Check if the log file exists before proceeding
        input_csv_path = os.path.join(log_dir, scenario_name, 'logs.csv')
        if not os.path.exists(input_csv_path):
            logging.warning(f"Skipping scenario {scenario_name} due to missing log file.")
            result[1] = 'skipped'  # Mark as skipped
            return result

        # Proceed with processing only if log file exists
        xml_file_path = os.path.join(source_dir, f'{scenario_name}.xml')
        convert_single_xml_to_json(xml_file_path, destination_dir)

        json_file_path = os.path.join(destination_dir, f'{scenario_name}.json')
        important_info = extract_important_information(json_file_path)
        layers = assign_layers(important_info)

        L1 = layers["L1_RoadLevel"]
        L2 = layers["L2_TrafficInfrastructure"]
        L3 = layers["L3_TemporalModifications"]
        L4 = layers["L4_MovableObjects"]
        L5 = layers["L5_EnvironmentalConditions"]
        L6 = layers["L6_DigitalInformation"]
        L7 = layers["L7_PlanningProblem"]

        polygons = lanelets_to_polygons(L1)
        visualize_dynamic_obstacles_with_time(L4, show_plot=False)

        scenario_output_dir = os.path.join(output_dir, scenario_name)
        os.makedirs(scenario_output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, scenario_name, 'ego_trajectory_positions_with_lanelets.csv')

        visualize_and_save_ego_trajectory_with_lanelets(input_csv_path, output_csv_path, polygons, L7, show_plot=False)
        obstacle_csv_path = os.path.join(output_dir, scenario_name, 'dynamic_obstacles_with_lanelets.csv')
        dynamic_obstacles_df = dynamic_obstacles_with_lanelets(L4, polygons, obstacle_csv_path)

        obstacles_path = os.path.join(output_dir, scenario_name, 'dynamic_obstacles.csv')
        filtered_data = extract_every_nth_timestep(obstacle_csv_path, obstacles_path, n=1)

        ego_path = os.path.join(output_dir, scenario_name, 'ego_trajectory.csv')
        filtered_ego_data = extract_ego_every_nth_timestep(output_csv_path, ego_path, n=1)

        # Update the result to 'done' after processing successfully
        result[1] = 'done'  # Mark as done

    except Exception as e:
        logging.error(f"Error processing scenario {scenario_name}: {e}")
        result[1] = 'error'  # Mark as error

    logging.info(f"Processed: {scenario_name} with status: {result[1]}")  # Log the output
    return result  # Return result

def main(source_dir, destination_dir, log_dir, output_dir, single_scenario, scenario_name=None, num_cpus=1, clean_output=False):
    """Main function for processing scenarios"""
    # Ensure output and destination directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(destination_dir, exist_ok=True)
    
    if clean_output:
        clean_output_directory(output_dir)

    # Prepare the report file
    report_path = os.path.join(output_dir, 'report.csv')
    with open(report_path, mode='w', newline='') as report_file:
        report_writer = csv.writer(report_file)
        report_writer.writerow(['Scenario Name', 'Status'])  # Write header row

        if single_scenario:
            if scenario_name is None:
                raise ValueError("Please provide a 'scenario_name' for single scenario processing.")
            logging.info(f"Processing single scenario: {scenario_name}")
            result = process_scenario(scenario_name, source_dir, destination_dir, log_dir, output_dir)
            report_writer.writerow(result)  # Write result to report
        else:
            # Get a list of all XML files in the source directory
            xml_files = [f for f in os.listdir(source_dir) if f.endswith('.xml')]

            # Use tqdm to display a progress bar
            if num_cpus > 1:
                # Create a manager for the shared progress counter
                with multiprocessing.Manager() as manager:
                    progress_counter = manager.Value('i', 0)  # Shared integer counter

                    # Create a shared tqdm progress bar to update
                    with tqdm(total=len(xml_files), desc="Processing scenarios") as pbar:
                        with multiprocessing.Pool(processes=num_cpus) as pool:
                            # Pass progress_counter to all workers
                            args = [(os.path.splitext(xml_file)[0], source_dir, destination_dir, log_dir, output_dir) for xml_file in xml_files]
                            results = pool.starmap(process_scenario, args)

                        # Manually update the progress bar
                        for result in results:
                            report_writer.writerow(result)  # Write result to report file
                            pbar.update(1)  # Update progress bar

            else:
                # Sequential processing with tqdm
                with tqdm(xml_files, desc="Processing scenarios") as pbar:
                    for xml_file in pbar:
                        scenario_name = os.path.splitext(xml_file)[0]  # Extract scenario name without extension
                        logging.info(f"Processing scenario: {scenario_name}")
                        result = process_scenario(scenario_name, source_dir, destination_dir, log_dir, output_dir)
                        report_writer.writerow(result)  # Write result to report
                        pbar.update(1)  # Update progress bar in sequential mode

if __name__ == "__main__":
    source_dir = '/home/yuan/ITSC/Safety/success_scenarios'
    destination_dir = '/home/yuan/ITSC/Safety/json_scenarios'
    log_dir = '/home/yuan/ITSC/Safety/validation_scenarios_normal'
    output_dir = '/home/yuan/ITSC/Safety/output_validation_normal'

    single_scenario = False  # Set to True for processing a single scenario
    scenario_name = 'ESP_Barcelona-29_33_T-1'  # Provide the scenario name if single_scenario is True
    num_cpus = 25  # Example: use 30 CPUs for batch processing
    clean_output = True  # Set this to True if you want to clean the output directory

    cProfile.run('main(source_dir, destination_dir, log_dir, output_dir, single_scenario, scenario_name, num_cpus, clean_output)')
