import os
import openai
import json
import pandas as pd
import re
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY3")  # Using key3 for Cartesian analysis
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

# Gemini API setup
gemini_client = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# DeepSeek API setup
deepseek_client = OpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com"
)

# Configuration dictionary for paths and parameters
CONFIG = {
    "base_dir": "/home/yuan/ITSC/Safety/output_validation_normal",
    "model_params": {
        "openai": {
            "model": "gpt-4o",
            "temperature": 1,  # Reduced temperature for more deterministic outputs
            "max_tokens": 16383,
            "top_p": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        },
        "gemini": {
            "model": "gemini-1.5-pro",
            "temperature": 1,  # Reduced temperature for more deterministic outputs
            "max_tokens": 32768,
            "top_p": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        },
        "deepseek": {
            "model": "deepseek-chat",
            "temperature": 1,  # Reduced temperature for more deterministic outputs
            "max_tokens": 8192,
            "top_p": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
    }
}

def set_params(engine="openai", **kwargs):
    """Set model parameters with defaults overridden by provided kwargs."""
    params = CONFIG["model_params"][engine].copy()
    params.update(kwargs)
    return params

def get_completion(engine, params, messages, max_retries=3, retry_delay=10):
    """Attempts to get a response from the model, with retries for transient errors."""
    retries = 0
    while retries < max_retries:
        try:
            if engine == "openai":
                response = openai.chat.completions.create(
                    model=params["model"],
                    messages=messages,
                    temperature=params["temperature"],
                    max_tokens=params["max_tokens"],
                    top_p=params["top_p"],
                    frequency_penalty=params["frequency_penalty"],
                    presence_penalty=params["presence_penalty"],
                )
                return response.choices[0].message.content

            elif engine == "gemini":
                response = gemini_client.chat.completions.create(
                    model=params["model"],
                    messages=messages,
                    n=1
                )
                return response.choices[0].message.content

            elif engine == "deepseek":
                response = deepseek_client.chat.completions.create(
                    model=params["model"],
                    messages=messages,
                    temperature=params["temperature"],
                    max_tokens=params["max_tokens"],
                    top_p=params["top_p"],
                    frequency_penalty=params["frequency_penalty"],
                    presence_penalty=params["presence_penalty"],
                )
                return response.choices[0].message.content

        except Exception as e:
            error_message = str(e)
            if "overloaded" in error_message.lower() or "500" in error_message:
                print(f"API is overloaded. Retrying {retries+1}/{max_retries} in {retry_delay} seconds...")
                time.sleep(retry_delay)  # Wait before retrying
                retries += 1
            else:
                print(f"Error occurred while processing {engine} completion: {e}")
                return None

    print(f"Failed to get a response from {engine} after {max_retries} retries.")
    return None



def generate_cart_analysis_icl(context):
    """Generate analysis using Cartesian coordinate system with improved chain of thought reasoning and in-context learning examples."""
    # Escape curly braces in the context using regex
    escaped_context = context.replace("{", "{{").replace("}", "}}")  # Double the braces

    return [
                {
            "role": "system",
            "content": """You are an expert in collision analysis for autonomous driving scenarios using Cartesian coordinates.
                        Your role is to evaluate the provided scenario based on the following safety metrics with scores ranging from 0 to 5 for each metric, where 0 indicates collision and 5 indicates no risk of collision:

                        Risk Levels:
                        0: Collision - Physical contact occurred or unavoidable
                        1: Extreme Risk - Immediate collision likely
                        2: High Risk - Close to collision, needs quick action
                        3: Medium Risk - Manageable with proper reaction
                        4: Low Risk - Minimal risk, sufficient time to react
                        5: Negligible Risk - No significant risk

                        You MUST end your analysis with properly formatted JSON that summarizes your findings.
                        **Metrics Considered in Cartesian Coordinates:**
                        - **Position**: X and Y coordinates in the global reference frame
                        - **Orientation**: Angle of the vehicle relative to the global reference frame
                        - **Velocity**: Speed and direction of movement
                        - **Acceleration**: Rate of change of velocity

                        Provide a detailed evaluation and summarize your findings."""
        },
        {
            "role": "user",
            "content": f"""Analyze the provided scenario for potential collisions:
                        Context: {escaped_context}

                        ### Instructions:
                        1. Focus primarily on the **most recent timestep**, using earlier steps to understand motion trends.
                        2. Evaluate each obstacle's trajectory and relative motion compared to the ego vehicle.
                        3. Use **TTC** (time to collision) and **MDC** (minimum distance to collision) to support your assessment.
                        4. If an obstacle has **Overall Risk Score = 0 or 1**, include a **brief explanation**.
                        5. Always include a valid JSON output at the end **this is essential**.
                        
                        ---

                            ### In-Context Examples:

                            **TTC Calculation Example**  
                            Obstacle 341 (at 3.1s):  
                            - Relative longitudinal position: -4.72m (behind ego)  
                            - Relative longitudinal velocity: 7.24 m/s (approaching)  
                            - TTC = -4.72 / 7.24 = -0.65s  overtaking from behind  
                            - Lateral distance = 0.27m  **High risk**

                            Obstacle 357 (at 3.1s):  
                            - Ahead by 0.77m, lateral distance 4.13m  **No immediate risk**

                            **Risk Assessment Example**  
                            Obstacle 30295:  
                            - Distance decreased from 26.17m to 21.16m  
                            - TTC 3.4s, same lanelet, closing in fast  
                            - Risk: **Extreme (1)**

                            Obstacle 30233:  
                            - Ego approaching stationary obstacle  
                            - Distance 0m, TTC = 0  
                            - Risk: **Collision (0)**

                            ---


                        ### Output Format:

                            Brief analysis for risky obstacles (0 or 1 score).  
                            Then, output this exact JSON format:

                                ```json
                                {{
                                    "CollisionObstacles": [
                                        {{
                                            "ObstacleID": "<ID>",
                                            "OverallRiskScore": 0
                                        }}
                                    ],
                                    "ExtremeRiskObstacle": {{
                                        "ObstacleID": "<ID>",
                                        "OverallRiskScore": 1
                                    }}
                                    "HighRiskObstacle": {{
                                        "ObstacleID": "<ID>",
                                        "OverallRiskScore": 2
                                    }}
                                }}
                                ```"""
        }
    ]

def process_safety_analysis(
    engine,
    base_dir,
    scenario_name,
    api_delay=5,
):
    """Process safety analysis for a scenario's first 10 timesteps in a single batch."""
    
    # Paths to CSV files
    ego_path = os.path.join(base_dir, scenario_name, "ego_trajectory.csv")
    obstacles_path = os.path.join(base_dir, scenario_name, "close_dynamic_obstacles.csv")  # Use the full obstacles file
    
    if not os.path.exists(ego_path) or not os.path.exists(obstacles_path):
        print(f"Trajectory files not found for scenario {scenario_name}")
        return
    
    # Read the CSV files
    ego_df = pd.read_csv(ego_path)
    obstacles_df = pd.read_csv(obstacles_path)
    
    # Get all timesteps and the first 10
    all_timesteps = sorted(ego_df['timestep'].unique())
    total_timesteps = len(all_timesteps)
    first_10_timesteps = all_timesteps[:10]
    
    print(f"  Total timesteps in scenario: {total_timesteps}")
    print(f"  Processing first {len(first_10_timesteps)} timesteps in a single batch")
    
    # Create output directories if they don't exist
    context_dir = os.path.join(base_dir, scenario_name, "agent_first", "cart_icl", engine, "context")
    analysis_dir = os.path.join(base_dir, scenario_name, "agent_first", "cart_icl", engine, "analysis")
    os.makedirs(context_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Generate combined context for all timesteps
    combined_context = ""
    
    for timestep in first_10_timesteps:
        try:
            # Generate context for this timestep
            ego_data = ego_df[ego_df['timestep'] == timestep].iloc[0]
            obstacle_data = obstacles_df[obstacles_df['timestep'] == timestep]
            
            combined_context += f"At {timestep * 0.1:.1f} seconds:\n"
            combined_context += (f"The position of Ego: {ego_data['ego_id']} is "
                               f"({ego_data['x_position']}, {ego_data['y_position']}), "
                               f"the orientation is {ego_data['orientation']}, "
                               f"the velocity is {ego_data['velocity']} and "
                               f"the acceleration is {ego_data['acceleration']}, "
                               f"current located in lanelet {ego_data['lanelet_id']}\n")
            
            # Add all obstacle positions
            for _, row in obstacle_data.iterrows():
                combined_context += (f"The position of obstacle: {row['obstacle_id']} is "
                                   f"({row['x_position']}, {row['y_position']}), "
                                   f"the orientation is {row['orientation']}, "
                                   f"the velocity is {row['velocity']} and "
                                   f"the acceleration is {row['acceleration']}, "
                                   f"current located in lanelet {row['lanelet_id']}\n")
            
            combined_context += "\n"  # Add a separator between timesteps
            
        except Exception as e:
            print(f"    Error generating context for timestep {timestep}: {str(e)}")
            continue
    
    # Save combined context
    context_file = os.path.join(context_dir, "context_all_timesteps.txt")
    with open(context_file, 'w') as f:
        f.write(combined_context)
    
    print(f"  Generated combined context with {len(first_10_timesteps)} timesteps")
    
    # Generate messages for LLM analysis with improved chain of thought and in-context learning
    messages = generate_cart_analysis_icl(combined_context)
    
    # Get model completion with retries
    params = set_params(engine)
    response = get_completion(engine, params, messages)
    
    if response:
        try:
            # Extract the JSON part from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                # Validate JSON
                json.loads(json_str)
            
            # Save the analysis
            output_file = os.path.join(analysis_dir, f"cart_analysis_all_timesteps.txt")
            with open(output_file, 'w') as f:
                f.write(response)
            
            print(f"  Completed analysis for all timesteps")
            
        except Exception as e:
            print(f"  Error processing response: {str(e)}")
    
    # Add delay to avoid API rate limits
    time.sleep(api_delay)

def process_scenarios(
    engine,
    base_dir,
    single_scenario=False,
    scenario_name=None,
    api_delay=30,
):
    """Process scenarios focusing on the first 10 timesteps."""
    
    if single_scenario and scenario_name:
        scenarios = [scenario_name]
    else:
        # Get all scenario directories
        scenarios = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    total_scenarios = len(scenarios)
    print(f"\nTotal number of scenarios to process: {total_scenarios}")
    
    # Calculate estimated time
    estimated_time_per_scenario = (api_delay + 2)  # 2 seconds buffer for processing
    total_estimated_time = total_scenarios * estimated_time_per_scenario
    hours = total_estimated_time // 3600
    minutes = (total_estimated_time % 3600) // 60
    seconds = total_estimated_time % 60
    
    print(f"Estimated total time: {hours}h {minutes}m {seconds}s")
    print(f"Estimated completion time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + total_estimated_time))}")
    print("\nList of scenarios:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario}")
    print("\n" + "="*50 + "\n")
    
    start_time = time.time()
    scenarios_completed = 0
    
    for idx, scenario in enumerate(scenarios, 1):
        try:
            print(f"\nProcessing scenario [{idx}/{total_scenarios}]: {scenario}")
            scenario_start_time = time.time()
            
            # Process safety analysis for the scenario
            process_safety_analysis(
                engine,
                base_dir,
                scenario,
                api_delay=api_delay
            )
            
            scenarios_completed += 1
            scenario_time = time.time() - scenario_start_time
            elapsed_time = time.time() - start_time
            
            # Calculate remaining time based on average time per scenario
            avg_time_per_scenario = elapsed_time / scenarios_completed
            remaining_scenarios = total_scenarios - scenarios_completed
            remaining_time = remaining_scenarios * avg_time_per_scenario
            
            # Format times
            elapsed_hours = int(elapsed_time // 3600)
            elapsed_minutes = int((elapsed_time % 3600) // 60)
            elapsed_seconds = int(elapsed_time % 60)
            
            remaining_hours = int(remaining_time // 3600)
            remaining_minutes = int((remaining_time % 3600) // 60)
            remaining_seconds = int(remaining_time % 60)
            
            print(f"Successfully completed scenario: {scenario} in {scenario_time:.1f}s")
            print(f"Progress: {scenarios_completed}/{total_scenarios} scenarios")
            print(f"Elapsed time: {elapsed_hours}h {elapsed_minutes}m {elapsed_seconds}s")
            print(f"Estimated remaining time: {remaining_hours}h {remaining_minutes}m {remaining_seconds}s")
            print(f"Expected completion: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + remaining_time))}")
            print("-"*50)
            
        except Exception as e:
            print(f"Error processing scenario {scenario}: {e}")
            continue

def main():
    # Global flags
    ENGINE = "deepseek"  # or "openai" or "deepseek"
    
    # Use 5 second delay for all engines
    API_DELAY = 5  # seconds
    
    # Process all scenarios with enhanced chain of thought and in-context learning approach
    process_scenarios(
        engine=ENGINE,
        base_dir=CONFIG["base_dir"],
        single_scenario=False,  # Process all scenarios
        scenario_name=None,     # Not used when processing all scenarios
        api_delay=API_DELAY
    )

if __name__ == "__main__":
    main() 