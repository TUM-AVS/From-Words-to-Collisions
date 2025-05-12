import os
import openai
import json
import pandas as pd
import re
import time
from dotenv import load_dotenv
from openai import OpenAI
import argparse

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY2")  # Using key3 for ego analysis
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
            "temperature": 1,
            "max_tokens": 32768,
            "top_p": 1,
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

def generate_ego_analysis(scenario_name, all_data, model_params):
    # Extract context from all_data and escape curly braces to avoid f-string formatting issues
    context = all_data  # Assuming all_data is the context string
    escaped_context = context.replace("{", "{{").replace("}", "}}")  # Double the braces
    
    # Define JSON example format
    json_example = '''{{
                "CollisionObstacle": {{
                    "ObstacleID": "OBSTACLE_ID",
                    "OverallRiskScore": "0"
                }},
                "ExtremeRiskObstacle": {{
                    "ObstacleID": "OBSTACLE_ID",
                    "OverallRiskScore": "1"
                }}
                "HighRiskObstacle": {{
                    "ObstacleID": "OBSTACLE_ID",
                    "OverallRiskScore": "2"
                }}  
            }}'''
    
    system_message = {"role": "system", "content": f"""You are a vehicle safety analysis AI that evaluates collision risks in autonomous driving scenarios. 
    
You analyze the risk of collision between the ego vehicle and multiple obstacles.

Risk Levels:
0: Collision - Physical contact occurred or unavoidable
1: Extreme Risk - Immediate collision likely
2: High Risk - Close to collision, needs quick action
3: Medium Risk - Manageable with proper reaction
4: Low Risk - Minimal risk, sufficient time to react
5: Negligible Risk - No significant risk
You MUST end your analysis with properly formatted JSON that summarizes your findings.

Metrics to assess:
1. Proximity - How close obstacles are to the ego vehicle
2. Minimum Distance to Collision (MDC) - Smallest projected distance between vehicles
   - MDC = 0 meters: Collision Risk (0)
   - 0 < MDC ≤ 0.5 meters: Extreme Risk (1)
   - 0.5 < MDC ≤ 1.0 meters: High Risk (2)
   - 1.0 < MDC ≤ 3.0 meters: Medium Risk (3)
   - 3.0 < MDC ≤ 5.0 meters: Low Risk (4)
   - MDC > 5.0 meters: No Risk (5)
3. Time-to-Collision (TTC) - Time until collision if trajectories maintained
   - TTC = 0 seconds: Collision Risk (0)
   - TTC ≤ 0.5 seconds: Extreme Risk (1)
   - 0.5 < TTC ≤ 1.0 seconds: High Risk (2)
   - 1.0 < TTC ≤ 3.0 seconds: Medium Risk (3)
   - 3.0 < TTC ≤ 5.0 seconds: Low Risk (4)
   - TTC > 5.0 seconds: No Risk (5)

IMPORTANT: Interpreting Alignment Cases
- Exact alignment (lateral or longitudinal distance = 0.00m) does NOT automatically indicate a collision. 
- Alignment simply means vehicles are on the same line in one dimension, but collision depends on both dimensions, trajectories, and velocities.
- When assessing risk based on obstacle direction:
   a) For obstacles in front/behind ego: Focus primarily on longitudinal distance, TTC, and closing velocity
   b) For obstacles to left/right of ego: Focus primarily on lateral distance, TTC, and closing velocity
   c) For diagonal positions: Focus on the smaller of the two distances, especially if both are closing

MDC calculation:
- For obstacles in the same lane (front/behind): Use primarily the lateral distance, but confirm with longitudinal trajectory
- For side-by-side obstacles (left/right): Use primarily the longitudinal distance, but confirm with lateral trajectory
- For crossing obstacles: MDC is the minimum distance when projected paths intersect

Examples:
1. Obstacle directly in front (lateral = 0.00m) at 10m:
   - Not a collision if longitudinal distance stable/increasing
   - High risk if longitudinal distance rapidly decreasing
   
2. Obstacle on same lateral position (lateral = 0.00m) but 5m behind ego:
   - Not a collision if obstacle speed ≤ ego speed (not catching up)
   - High/Medium risk if obstacle rapidly approaching from behind

3. Obstacle to right (longitudinal = 0.00m) at 1.5m laterally:
   - Medium risk if lateral distance stable/increasing
   - High risk if lateral distance rapidly decreasing

If TTC is negative, it typically means the obstacle has already passed the ego vehicle or their paths won't intersect. However, if MDC is very small and TTC is negative, this could indicate an imminent or recent collision.

When both TTC and MDC indicate different risk levels, prioritize the higher risk assessment (lower numerical score).

TTC calculation: 
- TTC = distance / relative_velocity (when velocities are approaching)
- If relative_velocity is zero or negative (moving apart), TTC is not applicable for collision

Your output MUST be in a specific format with detailed reasoning and a final JSON summary that explicitly identifies if there is any collision or extreme risk obstacle."""}

    example_analysis = """
For this scenario, I'll analyze the collision risk at the most recent timestep (2.4 seconds).

### Obstacle Analysis: Obstacle 344

#### Trajectory Analysis Across Timesteps:
At 1.5s: Behind (longitudinal: -8.67m, lateral: -1.52m), approaching at 4.16 m/s longitudinally
At 1.6s: Behind (longitudinal: -8.30m, lateral: -1.21m), approaching at 4.35 m/s longitudinally
At 1.7s: Rear-right (longitudinal: -7.73m, lateral: -1.86m), approaching at 4.41 m/s longitudinally
At 1.8s: Rear-right (longitudinal: -7.13m, lateral: -2.36m), approaching at 4.48 m/s longitudinally
At 1.9s: Rear-right (longitudinal: -6.78m, lateral: -1.90m), approaching at 4.71 m/s longitudinally
At 2.0s: Rear-right (longitudinal: -6.13m, lateral: -2.31m), approaching at 4.70 m/s longitudinally
At 2.1s: Rear-right (longitudinal: -5.51m, lateral: -2.50m), approaching at 4.66 m/s longitudinally
At 2.2s: Rear-right (longitudinal: -5.19m, lateral: -1.90m), approaching at 4.99 m/s longitudinally
At 2.3s: Rear-right (longitudinal: -4.62m, lateral: -1.89m), approaching at 5.08 m/s longitudinally
At 2.4s: Right (longitudinal: -4.06m, lateral: -1.78m), approaching at 5.26 m/s longitudinally

#### Most Recent Timestep (2.4s) Analysis:
- Position: Right of ego vehicle (previously Rear-right)
- Longitudinal distance: -4.06m (negative means behind ego)
- Lateral distance: -1.78m (negative means to the right of ego)
- Relative velocity: longitudinal 5.26 m/s, lateral 2.25 m/s
- Relative acceleration: longitudinal 1.61 m/s², lateral -1.51 m/s²
- Motion description: "Obstacle is driving toward the ego car from behind. Obstacle is driving toward the ego car laterally from the right."

#### Distance Risk Analysis:
- Current separation: 4.06m longitudinally (behind ego), 1.78m laterally (to the right)
- Trajectories: Obstacle is approaching rapidly from behind and right
- The obstacle has been consistently gaining on the ego vehicle
- Longitudinal distance decreased from -8.67m to -4.06m over 0.9 seconds
- Lateral distance is also closing (obstacle moving toward ego from right)
- Minimum Distance to Collision (MDC) calculation: 
  - With relative velocities of 5.26 m/s longitudinally and 2.25 m/s laterally
  - Projecting forward, the trajectories will intersect
  - Given current acceleration trends, the obstacle will overtake the ego vehicle
  - MDC is projected to be approximately 0.8m when the obstacle passes the ego
- MDC threshold check: 0.8m < 1.0m, indicating High Risk (2) for distance

#### Time Risk Analysis:
- Time-to-Collision (TTC) calculation:
  - Longitudinal TTC = 4.06m ÷ 5.26m/s = 0.77 seconds
  - Lateral TTC = 1.78m ÷ 2.25m/s = 0.79 seconds
  - Combined TTC = min(0.77s, 0.79s) = 0.77 seconds
  - Accounting for vehicle dimensions: approximately 0.5-0.6 seconds
- TTC is accelerating (longitudinal acceleration is 1.61 m/s²)
- TTC threshold check: 0.77s < 1.0s, indicating High Risk (2) for time

#### Combined Risk Assessment:
- Distance Risk: 2 (High Risk)
- Time Risk: 2 (High Risk)
- Overall Risk score: 2 (High Risk)
- Based on the rapid approach from behind and steadily decreasing distance
- Notable concerns: Accelerating approach (longitudinal acceleration 1.61 m/s²)

```
"""

    example_analysis = clean_example(example_analysis)

    return [
               {
                    "role": "system",
                    "content": system_message["content"]
                },
                    {
                "role": "user",
                "content": f"""Based on the given scenario context:
                ```{escaped_context}```
                which shows information about obstacles across multiple timesteps (usually 10 timesteps with 0.1s intervals).

                ### CRITICAL INSTRUCTION:
                1. Focus your analysis on the MOST RECENT TIMESTEP (the last one in the context)
                2. Use the previous timesteps to understand obstacle trajectories and predict immediate future positions
                3. IMPORTANT: When doing your analysis, use the ADJUSTED distances (adjusted_d_long and adjusted_d_lat) rather than the raw distances. These adjusted values account for vehicle size and provide more accurate risk assessments.

                For each obstacle, the data includes:
                - Obstacle ID
                - Relative Direction to ego vehicle (front, rear, left, right, or combinations)
                - Raw longitudinal and lateral distances (m)
                - Adjusted longitudinal and lateral distances (m) - USE THESE FOR YOUR ANALYSIS
                - Relative velocity components (m/s) positive means obstacle drive to ego, negative means obstacle drive away from ego
                - Relative acceleration components (m/s²)
                - Motion description explaining trajectory

                ### IMPORTANT: Pay close attention to the Motion Description!
                The motion description (e.g., "Exact longitudinal alignment" or "Obstacle is driving toward ego car laterally") provides critical context for interpreting:
                - Near-zero distances (vehicles are aligned in that dimension, NOT necessarily colliding)
                - Negative distances (behind/right of ego)
                - Positive velocities (approaching in that dimension)
                - Direction of movement relative to ego
                
                ### Direction-Based Analysis:
                1. For obstacles mainly in front/behind (Front, Rear):
                   - Focus primarily on LONGITUDINAL distances and velocities
                   - A lateral distance of 0.00m indicates same-lane alignment, but NOT collision unless longitudinal distance is also near zero with closing velocity
                
                2. For obstacles mainly to sides (Left, Right):
                   - Focus primarily on LATERAL distances and velocities
                   - A longitudinal distance of 0.00m indicates side-by-side alignment, but NOT collision unless lateral distance is also near zero with closing velocity
                
                3. For diagonal positions (Front-Left, Rear-Right, etc.):
                   - Consider both dimensions with emphasis on the smaller distance if both are closing

                Use these descriptions to verify your mathematical calculations of TTC and collision risk, especially when values are near zero or negative.

                ### Risk Assessment Process:
                1. Track position changes across timesteps to establish trajectory
                2. For each obstacle at the MOST RECENT timestep:
                   - Calculate Time-to-Collision (TTC) = Distance/|Relative Velocity| for both directions
                   - Calculate Minimum Distance to Collision (MDC)
                   - Assess both longitudinal and lateral components
                   - Consider acceleration trends when predicting imminent collisions
                   - Confirm your calculations align with the motion description
                3. Assign risk scores (0-5):
                   - 0: Collision occurring (distance ≈ 0m in BOTH dimensions with closing/zero velocity)
                   - 1: Extreme Risk (TTC < 0.5s or MDC < 0.5m with converging trajectories)
                   - 2: High Risk (TTC < 1.0s or MDC < 1.0m)
                   - 3: Medium Risk (TTC < 3.0s or MDC < 3.0m)
                   - 4: Low Risk (TTC < 5.0s or MDC < 5.0m)
                   - 5: No Risk (TTC > 5.0s and MDC > 5.0m or diverging paths)
                4. Focus on the LOWEST score (highest risk) for each obstacle
                5. Predict the collision obstacle with risk score 0

                ### Output Format:
                1. Begin with obstacle-by-obstacle analysis
                2. Include trajectory analysis across timesteps
                3. Show TTC and MDC calculations and reasoning
                4. End with JSON summary in this exact format:

                {json_example}
                
                You MUST follow this exact risk scoring and format to ensure consistent analysis."""
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
    relative_metrics_path = os.path.join(base_dir, scenario_name, "close_relative_metrics.csv")
    
    if not os.path.exists(ego_path) or not os.path.exists(relative_metrics_path):
        print(f"  Required trajectory files not found for scenario {scenario_name}")
        return {}
    
    # Create output directories if they don't exist
    context_dir = os.path.join(base_dir, scenario_name, "agent", "ego_icl", engine, "context")
    analysis_dir = os.path.join(base_dir, scenario_name, "agent", "ego_icl", engine, "analysis")
    os.makedirs(context_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Path to the combined context file
    context_file = os.path.join(context_dir, "context_all_timesteps.txt")
    
    # Check if context file already exists, load it if it does
    if os.path.exists(context_file):
        print(f"  Loading existing context from {context_file}")
        with open(context_file, 'r') as f:
            context_data = f.read()
    else:
        # If context doesn't exist, we need to generate it
        print(f"  Generating context for {scenario_name}")
        
        # Read the CSV files
        ego_df = pd.read_csv(ego_path)
        relative_metrics_df = pd.read_csv(relative_metrics_path)
        
        # Get all timesteps and the last 10
        all_timesteps = sorted(ego_df['timestep'].unique())
        total_timesteps = len(all_timesteps)
        last_10_timesteps = all_timesteps[-12:-2]
        
        print(f"  Total timesteps in scenario: {total_timesteps}")
        print(f"  Processing last {len(last_10_timesteps)} timesteps in a single batch")
        
        # Generate combined context for all timesteps
        context_data = ""
        
        for timestep in last_10_timesteps:
            try:
                # Generate context for this timestep
                context_data += f"At {timestep * 0.1:.1f} seconds:\n"
                
                # Add relative metrics information if available
                timestep_metrics = relative_metrics_df[relative_metrics_df['timestep'] == timestep]
                if not timestep_metrics.empty:
                    for _, metric in timestep_metrics.iterrows():

                        # Use motion description from CSV if available
                        motion_description = metric['motion_description'] if 'motion_description' in metric else ""
                        
                        # Use adjusted distances for context with clear indication these are adjusted values
                        context_data += f"  Obstacle {metric['obstacle_id']} is in the {metric['relative_direction']} of the ego car. " \
                                      f"The real distance is longitudinal {metric['adjusted_d_long']:.2f} m and lateral {metric['adjusted_d_lat'] :.2f} m. " \
                                      f"Relative velocity: longitudinal {metric['v_rel_long']:.2f} m/s, lateral {metric['v_rel_lat']:.2f} m/s. " \
                                      f"Relative acceleration: longitudinal {metric['a_rel_long']:.2f} m/s², lateral {metric['a_rel_lat']:.2f} m/s². " \
                                      f"Motion: {motion_description}\n"
                
                context_data += "\n"  # Add a separator between timesteps
                
            except Exception as e:
                print(f"    Error generating context for timestep {timestep}: {str(e)}")
                continue
        
        # Save the context
        with open(context_file, 'w') as f:
            f.write(context_data)
        
        print(f"  Generated and saved context to {context_file}")
    
    # Generate messages for LLM analysis
    print(f"  Generating analysis using {engine}")
    model_params = set_params(engine)
    messages = generate_ego_analysis(scenario_name, context_data, model_params)
    
    # Get model completion with retries
    response = get_completion(engine, model_params, messages)
    
    if response:
        try:
            # Extract the JSON part from the response
            json_part = extract_json_from_text(response)
            
            if json_part:
                try:
                    # Validate JSON
                    json.loads(json_part)
                    
                    # Save the analysis
                    output_file = os.path.join(analysis_dir, f"ego_analysis_all_timesteps.txt")
                    with open(output_file, 'w') as f:
                        f.write(response)
                    
                    print(f"  Successfully saved analysis to {output_file}")
                    
                    # Process the results
                    result = process_safety_analysis_text(scenario_name, response, engine)
                    
                    return result
                    
                except json.JSONDecodeError:
                    print(f"  Warning: Invalid JSON in response for {scenario_name}")
            else:
                print(f"  Warning: No JSON found in response for {scenario_name}")
            
            # Save the response anyway for debugging
            output_file = os.path.join(analysis_dir, f"ego_analysis_all_timesteps.txt")
            with open(output_file, 'w') as f:
                f.write(response)
            
        except Exception as e:
            print(f"  Error processing response: {str(e)}")
    else:
        print(f"  Error: No response from {engine} for {scenario_name}")
    
    # Add delay to avoid API rate limits
    time.sleep(api_delay)
    
    return {}

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

def clean_example(example_text):
    """Clean and format the example text."""
    lines = example_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Remove excessive whitespace but keep the structure
        cleaned_line = line.rstrip()
        cleaned_lines.append(cleaned_line)
    
    return '\n'.join(cleaned_lines)

def extract_json_from_text(text):
    """Extract JSON part from the provided text."""
    # Try to find a JSON section enclosed in ```json and ``` markers
    json_pattern = r'```json(.*?)```'
    json_match = re.search(json_pattern, text, re.DOTALL)
    
    if json_match:
        return json_match.group(1).strip()
    
    # If no JSON block found, look for JSON object start/end
    json_start = text.find('{')
    json_end = text.rfind('}') + 1
    
    if json_start != -1 and json_end > json_start:
        return text[json_start:json_end]
    
    return None

def extract_field_value(data, field_names):
    """Extract a field value from a dictionary using a list of possible field names."""
    if data is None:
        return None
    
    for field in field_names:
        if field in data:
            return data[field]
    
    return None

def process_safety_analysis_text(scenario_name, safety_analysis, engine):
    """Process the safety analysis text and extract collision obstacles."""
    
    # Extract JSON part from the response
    json_part = extract_json_from_text(safety_analysis)
    
    if not json_part:
        return {}
    
    try:
        json_data = json.loads(json_part)
    except json.JSONDecodeError:
        return {}
    
    # Initialize result dictionary
    result = {
        "scenario_name": scenario_name,
        "is_collision_scenario": False,
        "collision_obstacle_ids": [],
        "extreme_risk_obstacle_id": None,
        "reasoning": "",
        "engine": engine
    }
    
    # Extract collision obstacle ID if it exists
    collision_obstacle = extract_field_value(json_data, ["CollisionObstacle", "CollisionObstacles"])
    if collision_obstacle:
        # Handle both single object and array formats
        if isinstance(collision_obstacle, list):
            for obstacle in collision_obstacle:
                obstacle_id = extract_field_value(obstacle, ["ObstacleID"])
                if obstacle_id is not None:
                    result["collision_obstacle_ids"].append(str(obstacle_id))
        else:
            obstacle_id = extract_field_value(collision_obstacle, ["ObstacleID"])
            if obstacle_id is not None:
                result["collision_obstacle_ids"].append(str(obstacle_id))
    
    # Extract IsCollisionScenario flag
    is_collision = extract_field_value(json_data, ["IsCollisionScenario"])
    if is_collision is not None:
        result["is_collision_scenario"] = is_collision
    
    # Extract justification/reasoning if it exists
    reasoning = extract_field_value(json_data, ["Reasoning", "Justification"])
    if reasoning:
        result["reasoning"] = reasoning
    
    # Extract extreme risk obstacle ID if it exists
    extreme_risk = extract_field_value(json_data, ["ExtremeRiskObstacle", "ExtremeRiskObstacles"])
    if extreme_risk:
        # Handle both single object and array formats
        if isinstance(extreme_risk, list):
            if len(extreme_risk) > 0:
                obstacle_id = extract_field_value(extreme_risk[0], ["ObstacleID"])
                if obstacle_id is not None:
                    result["extreme_risk_obstacle_id"] = str(obstacle_id)
        else:
            obstacle_id = extract_field_value(extreme_risk, ["ObstacleID"])
            if obstacle_id is not None:
                result["extreme_risk_obstacle_id"] = str(obstacle_id)
    
    # Check if this is a collision scenario based on the data
    if not result["is_collision_scenario"] and result["collision_obstacle_ids"]:
        # The collision data is already extracted above, so we just need to set the flag if we have collision obstacles
        result["is_collision_scenario"] = True
    
    return result

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Process safety analysis for scenarios')
    parser.add_argument('--engine', type=str, default="openai", help='LLM engine to use (openai, gemini, or deepseek)')
    parser.add_argument('--scenario', type=str, help='Single scenario to process')
    parser.add_argument('--api-delay', type=int, default=25, help='Delay between API calls in seconds')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Engine selection
    ENGINE = args.engine
    
    # API delay
    API_DELAY = args.api_delay
    
    # Process scenarios
    process_scenarios(
        engine=ENGINE,
        base_dir=CONFIG["base_dir"],
        single_scenario=False,  # Process all scenarios
        scenario_name=args.scenario,
        api_delay=API_DELAY
    )

if __name__ == "__main__":
    main() 