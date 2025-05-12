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
gemini_api_key = os.getenv("GEMINI_API_KEY3")  # Using key1 for Cartesian analysis
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
    "base_dir": "/home/yuan/ITSC/Safety/output_validation",
    "model_params": {
        "openai": {
            "model": "gpt-4o",
            "temperature": 1.0,
            "max_tokens": 16383,
            "top_p": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        },
        "gemini": {
            "model": "gemini-1.5-pro",
            "temperature": 1.0,
            "max_tokens": 32768,
            "top_p": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        },
        "deepseek": {
            "model": "deepseek-chat",
            "temperature": 1.0,
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

def generate_cart_analysis(context):
    """Generate CoT-based analysis using Cartesian coordinates with clearer structure and streamlined reasoning."""
    escaped_context = context.replace("{", "{{").replace("}", "}}")

    return [
        {
            "role": "system",
            "content": """You are an expert in collision analysis for autonomous driving scenarios using Cartesian coordinates.

            Evaluate the scenario using these **risk levels**:
            - 0: Collision Physical contact occurred or unavoidable
            - 1: Extreme Risk Collision likely and imminent
            - 2: High Risk Close to collision, needs urgent action
            - 3: Medium Risk Manageable with driver/system response
            - 4: Low Risk No immediate danger
            - 5: Negligible Risk No meaningful risk

            You will reason over position, velocity, acceleration, and heading changes across 10 timesteps. Your final task is to output structured JSON summarizing obstacles with the highest risks.

            You MUST ALWAYS end with a correctly formatted JSON object. This is the highest priority. Make sure all JSON syntax is valid, with proper commas between objects and no trailing commas."""
        },
        {
            "role": "user",
            "content": f"""Analyze the following scenario using Cartesian coordinates.
                        Context: {escaped_context}

                        ---

                        ### Instructions:
                        1. Focus primarily on the **most recent timestep**, using earlier steps to understand motion trends.
                        2. Evaluate each obstacle's trajectory and relative motion compared to the ego vehicle.
                        3. Use **TTC** (time to collision) and **MDC** (minimum distance to collision) to support your assessment.
                        4. If an obstacle has **Overall Risk Score = 0 or 1**, include a **brief explanation**.
                        5. Always include a valid JSON output at the end **this is essential**.

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
                                    }},
                                    "HighRiskObstacle": {{
                                        "ObstacleID": "<ID>",
                                        "OverallRiskScore": 2
                                    }}
                                }}
                                ```
                                
                            IMPORTANT: Ensure your JSON is properly formatted with required commas between objects and no trailing commas. The JSON must be valid and parseable.
"""
        }
    ]

def process_safety_analysis(
    engine,
    base_dir,
    scenario_name,
    api_delay=5,
):
    """Process safety analysis for a scenario's last 10 timesteps in a single batch."""
    
    # Paths to CSV files
    ego_path = os.path.join(base_dir, scenario_name, "ego_trajectory.csv")
    obstacles_path = os.path.join(base_dir, scenario_name, "close_dynamic_obstacles.csv")  # Use the full obstacles file
    
    if not os.path.exists(ego_path) or not os.path.exists(obstacles_path):
        print(f"Trajectory files not found for scenario {scenario_name}")
        return
    
    # Read the CSV files
    ego_df = pd.read_csv(ego_path)
    obstacles_df = pd.read_csv(obstacles_path)
    
    # Get all timesteps and the last 10
    all_timesteps = sorted(ego_df['timestep'].unique())
    total_timesteps = len(all_timesteps)
    last_10_timesteps = all_timesteps[-12:-2]
    
    print(f"  Total timesteps in scenario: {total_timesteps}")
    print(f"  Processing last {len(last_10_timesteps)} timesteps in a single batch")
    
    # Create output directories if they don't exist
    context_dir = os.path.join(base_dir, scenario_name, "agent", "cart_cot", engine, "context")
    analysis_dir = os.path.join(base_dir, scenario_name, "agent", "cart_cot", engine, "analysis")
    os.makedirs(context_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Generate combined context for all timesteps
    combined_context = ""
    
    for timestep in last_10_timesteps:
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
    
    print(f"  Generated combined context with {len(last_10_timesteps)} timesteps")
    
    # Generate messages for LLM analysis
    messages = generate_cart_analysis(combined_context)
    
    # Get model completion with retries
    params = set_params(engine)
    response = get_completion(engine, params, messages)
    
    if response:
        try:
            # Save the full response first
            output_file = os.path.join(analysis_dir, f"cart_analysis_all_timesteps.txt")
            with open(output_file, 'w') as f:
                f.write(response)
            
            # Extract the JSON part from the response using a more robust approach
            try:
                # Try to find JSON between ```json and ``` markers
                json_pattern = r'```json\s*([\s\S]*?)\s*```'
                json_matches = re.findall(json_pattern, response)
                
                if json_matches:
                    json_str = json_matches[0]
                    # Clean up any potential formatting issues
                    json_str = json_str.strip()
                    # Validate and format JSON
                    parsed_json = json.loads(json_str)
                    formatted_json = json.dumps(parsed_json, indent=4)
                    
                    # Save just the formatted JSON
                    json_file = os.path.join(analysis_dir, f"cart_analysis_json.txt")
                    with open(json_file, 'w') as f:
                        f.write(formatted_json)
                    
                    print(f"  Extracted and saved valid JSON")
                else:
                    # Fallback to the old method if no code blocks found
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        # Try to fix common JSON formatting issues
                        json_str = json_str.strip()
                        # Validate and format JSON
                        parsed_json = json.loads(json_str)
                        formatted_json = json.dumps(parsed_json, indent=4)
                        
                        # Save just the formatted JSON
                        json_file = os.path.join(analysis_dir, f"cart_analysis_json.txt")
                        with open(json_file, 'w') as f:
                            f.write(formatted_json)
                        
                        print(f"  Extracted and saved valid JSON using fallback method")
                    else:
                        print(f"  Could not find JSON in the response")
            except json.JSONDecodeError as je:
                print(f"  Invalid JSON in response: {str(je)}")
                # Save the problematic JSON string for debugging
                error_file = os.path.join(analysis_dir, f"invalid_json_error.txt")
                with open(error_file, 'w') as f:
                    f.write(f"Error: {str(je)}\n\nAttempted JSON:\n{json_str if 'json_str' in locals() else 'Not extracted'}")
            
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
    """Process scenarios focusing on the last 10 timesteps."""
    
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
    ENGINE = "gemini"  # or "openai" or "deepseek"
    
    # Use 5 second delay for all engines
    API_DELAY = 5  # seconds
    
    # Process all scenarios with enhanced chain of thought approach
    process_scenarios(
        engine=ENGINE,
        base_dir=CONFIG["base_dir"],
        single_scenario=False,  # Process all scenarios
        scenario_name=None,     # Not used when processing all scenarios
        api_delay=API_DELAY
    )

if __name__ == "__main__":
    main() 