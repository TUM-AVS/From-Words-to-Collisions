
def generate_timestep_report(ego_csv_path, obstacles_csv_path, output_txt_path):
    """
    Generate a text report for each 5th timestep, including the positions and lanelet IDs
    of the ego vehicle and dynamic obstacles.

    Parameters:
    - ego_csv_path: str, path to the ego trajectory CSV file
    - obstacles_csv_path: str, path to the dynamic obstacles CSV file
    - output_txt_path: str, path to save the generated text file
    """
    # Load CSV files
    ego_df = pd.read_csv(ego_csv_path)
    obstacles_df = pd.read_csv(obstacles_csv_path)

    # Open the output text file
    with open(output_txt_path, 'w') as file:
        # Get unique timesteps from ego data (every 5th timestep)
        unique_timesteps = sorted(ego_df['timestep'].unique())
        
        for timestep in unique_timesteps:
            # Get ego data for the current timestep
            ego_data = ego_df[ego_df['timestep'] == timestep].iloc[0]
            ego_info = (
                f"The position of Ego: {ego_data['ego_id']} is "
                f"({ego_data['x_position']}, {ego_data['y_position']}) and "
                f"current lanelet is {ego_data['lanelet_id']}"
            )

            # Get obstacle data for the current timestep
            obstacle_data = obstacles_df[obstacles_df['timestep'] == timestep]
            obstacle_info = "\n".join(
                f"The position of obstacle: {row['obstacle_id']} is "
                f"({row['x_position']}, {row['y_position']}) and "
                f"current lanelet is {row['lanelet_id']}"
                for _, row in obstacle_data.iterrows()
            )

            # Write to the file
            file.write(f"At {timestep} seconds:\n")
            file.write(ego_info + "\n")
            if not obstacle_data.empty:
                file.write(obstacle_info + "\n")
            file.write("\n")

    print(f"Report saved to {output_txt_path}")


# Generate a text report for each 5th timestep
def generate_timestep_report(ego_csv_path, obstacles_csv_path, output_txt_path, timestep_interval=5):
    """
    Generate a text report for each specified timestep interval, including the positions
    and lanelet IDs of the ego vehicle and dynamic obstacles.
    """
    # Load CSV files
    ego_df = pd.read_csv(ego_csv_path)
    obstacles_df = pd.read_csv(obstacles_csv_path)

    # Initialize the report list
    report = []

    # Get unique timesteps and filter by the specified interval
    unique_timesteps = sorted(ego_df['timestep'].unique())
    filtered_timesteps = [t for t in unique_timesteps if t % timestep_interval == 0]

    for timestep in filtered_timesteps:
        # Get ego data for the current timestep
        ego_data = ego_df[ego_df['timestep'] == timestep].iloc[0]
        ego_info = (
            f"The position of Ego: {ego_data['ego_id']} is "
            f"({ego_data['x_position']}, {ego_data['y_position']}) and "
            f"current lanelet is {ego_data['lanelet_id']}"
        )

        # Get obstacle data for the current timestep
        obstacle_data = obstacles_df[obstacles_df['timestep'] == timestep]
        obstacle_info = "\n".join(
            f"The position of obstacle: {row['obstacle_id']} is "
            f"({row['x_position']}, {row['y_position']}) and "
            f"current lanelet is {row['lanelet_id']}"
            for _, row in obstacle_data.iterrows()
        )

        # Compile the timestep report
        timestep_report = f"At {timestep} seconds:\n{ego_info}\n"
        if not obstacle_data.empty:
            timestep_report += obstacle_info + "\n"
        timestep_report += "\n"

        # Add to the report list
        report.append(timestep_report)

    # Save the report to the output text file
    with open(output_txt_path, 'w') as file:
        file.writelines(report)
    
    print(f"Report saved to {output_txt_path}")
    return report

