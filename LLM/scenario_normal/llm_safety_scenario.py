import os
import openai
import json
import pandas as pd
import re
import time
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY3")  # Using key3 for safety metrics analysis
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
            "max_tokens": 32768,
            "top_p": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
    },
    "proximity_radius": 30  # Only include obstacles within this radius of the ego vehicle
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

def generate_safety_analysis(context):
    """Generate analysis using safety metrics by constructing messages for LLM reasoning."""
    # Escape curly braces in the context using regex
    escaped_context = context.replace("{", "{{").replace("}", "}}")  # Double the braces

    return [
        {
            "role": "system",
            "content": """
                    You are an expert in collision analysis for autonomous driving scenarios.
                    Your role is to evaluate the provided scenario based on the following safety metrics with scores ranging from 0 to 5 for each metric, where 0 indicates collision and 5 indicates no risk of collision:

                    **Risk Levels and Definitions:**
                    <Collision Risk (Score: 0)>: Actual collision occurring.
                    <Extreme Risk (Score: 1)>: Immediate very high likelihood of collision. Urgent action is required.
                    <High Risk (Score: 2)>: Close to collision or highly probable collision path. Needs prompt attention and quick manoeuvring.
                    <Medium Risk (Score: 3)>: Moderate collision risk but manageable with timely reactions and proper strategies.
                    <Low Risk (Score: 4)>: Minimal collision risk. The situation is controllable with sufficient time to react.
                    <Negligible Risk (Score: 5)>: No significant collision risk. Obstacles are either moving away or far enough not to interfere.

                    **Metrics Considered:**

                    1. **Distance to Collision (DTC):**
                        - DTClong: Longitudinal Distance to Collision.
                        - DTClat: Lateral Distance to Collision.
                        - LongDSC: Longitudinal Distance Safety Score.
                        - LatDSC: Lateral Distance Safety Score.
                        - Risk Levels Based on DTC:
                            - **Collision Risk (LongDSC = 0 or LatDSC = 0):** DTClong = 0 or DTClat = 0.
                            - **Extreme Risk (LongDSC = 1 or LatDSC = 1):** 0 <DTClong <= 0.5 or 0 <DTClat <= 0.5.
                            - **High Risk (LongDSC = 2 or LatDSC = 2):** 0.5 < DTClong <= 1 or 0.5 < DTClat <= 1.
                            - **Medium Risk (LongDSC = 3 or LatDSC = 3 ):** 1 < DTClong <= 3  or 1 < DTClat <= 3.
                            - **Low Risk (LongDSC = 4 or LatDSC = 4):**  3 < DTClong <= 5  or  3 < DTClat <= 5.
                            - **Negligible Risk (LongDSC = 5 or LatDSC = 5):** DTClong > 5 or DTClat > 5.

                        - **Weighting and Direction Adjustment:** 
                            - Overall Risk Score: DSC = LongDSC * wdominant + LatDSC * (1-wdominant),
                              where wdominant is determined by the relative direction:
                                - Front/Back: wdominant = 1.
                                - Left/Right: wdominant = 0.
                                - Other directions: wdominant = 0.5.

                    2. **Time to Collision (TTC):**
                        - TTClong: Longitudinal Time to Collision.
                        - TTClat: Lateral Time to Collision.
                        - LongTSC: Longitudinal Time Safety Score.
                        - LatTSC: Lateral Time Safety Score.
                        - Risk Levels Based on TTC:
                            - **Collision Risk (LongTSC = 0 or LatTSC = 0):** TTClong = 0 or TTClat = 0.
                            - **Extreme Risk (LongTSC = 1 or LatTSC = 1): ** TTClong <= 0.5 or TTClat <= 0.5.
                            - **High Risk (LongTSC = 2 or LatTSC = 2):** 0.5 < TTClong <= 1 or 0.5 < TTClat <= 1.
                            - **Medium Risk (LongTSC = 3 or LatTSC = 3):** 1 < TTClong <= 3 or 1 < TTClat <= 3.
                            - **Low Risk (LongTSC = 4 or LatTSC = 4):** 3 < TTClong <= 5 or 3 < TTClat <= 5.
                            - **Negligible Risk (LongTSC = 5 or LatTSC = 5):** TTClong > 5 or TTClat > 5.
                            If both are 0, the risk level should be 0 which means collision.

                        - **Weighting and Direction Adjustment:** 
                            - Overall Risk Score: TSC = LongTSC * wdominant + LatTSC * (1-wdominant),
                              where wdominant is determined by the relative direction:
                                - Front/Back: wdominant = 1.
                                - Left/Right: wdominant = 0.
                                - Other directions: wdominant = 0.5.

                    **Determining Overall Risk:**
                    The overall risk score combines DTC and TTC metrics:
                    Risk Score = 0.5 * DSC + 0.5 * TSC
                    The final risk score should be rounded to the nearest integer.

                    **Identifying Potential Ego Attackers:**
                    An obstacle can be identified as a potential ego attacker based on the following comprehensive criteria:

                    1. Position and Direction:
                       - Must be in front, front-left, or front-right of the ego vehicle
                       - Must be within a critical distance (DTClong < 5.0m and DTClat < 3.0m)
                       - Must have a trajectory that could intersect with the ego vehicle's path

                    2. Risk Assessment:
                       - Risk Score of 0-3 (Collision to Medium Risk)
                       - For Risk Score 0-1: Immediate threat, high probability of collision
                       - For Risk Score 2-3: Potential threat if the obstacle changes behavior

                    3. Motion Characteristics:
                       - Sudden or unexpected changes in speed or direction
                       - Aggressive acceleration or deceleration
                       - Erratic or unpredictable movement patterns
                       - Intentional path changes towards the ego vehicle

                    4. Attack Potential:
                       - High: Risk Score 0-1, close proximity, aggressive motion
                       - Medium: Risk Score 2-3, moderate proximity, suspicious motion
                       - Low: Risk Score 4-5, but showing concerning behavior patterns

                    5. Contextual Factors:
                       - Multiple obstacles coordinating movement
                       - Obstacles maintaining dangerous proximity
                       - Repeated aggressive maneuvers
                       - History of risk-increasing behavior across timesteps

                    **IMPORTANT: FOLLOW THESE STEPS CAREFULLY FOR EACH OBSTACLE:**
                    
                    1. Identify the relative direction of the obstacle (Front, Back, Left, Right, Front-left, etc.)
                    
                    2. Determine wdominant based on the direction:
                       - For Front or Back/Behind: wdominant = 1.0 (longitudinal is dominant)
                       - For Left or Right: wdominant = 0.0 (lateral is dominant)
                       - For Front-left, Front-right, Rear-left, or Rear-right: wdominant = 0.5 (both directions equally)
                    
                    3. For Distance Safety Score (DSC): 
                       - Calculate LongDSC based on DTClong
                       - Calculate LatDSC based on DTClat
                       - Apply the formula: DSC = LongDSC * wdominant + LatDSC * (1-wdominant)
                    
                    4. For Time Safety Score (TSC):
                       - Calculate LongTSC based on TTClong
                       - Calculate LatTSC based on TTClat  
                       - Apply the formula: TSC = LongTSC * wdominant + LatTSC * (1-wdominant)

                    5. Calculate Overall Risk Score = 0.5 * DSC + 0.5 * TSC (round to nearest integer)
                    """
        },
        {
            "role": "user",
            "content": f"""
            Based on the given scenario context:
            ```{escaped_context}```
            which describes information about obstacles across multiple timesteps (usually 10 timesteps with 0.1s intervals).

            ### CRITICAL INSTRUCTION:
            1. Focus your analysis on the MOST RECENT TIMESTEP (the last one in the context)
            2. Use the previous timesteps to understand obstacle trajectories and predict immediate future positions

            For each obstacle, the data includes:
            - Obstacle ID  
            - Relative Direction, which can be front, back, left, right, front-left, front-right, back-left, or back-right.
            - Real Distance in both longitudinal and lateral directions (treat these as DTClong and DTClat).
            - Longitudinal and Lateral TTC values.
            - Motion Description providing additional context about the obstacle's movement.

            Steps to Follow:
            1. Use the provided relative direction to determine the dominant direction for weighting.
                - Front/Back: wdominant = 1.
                - Left/Right: wdominant = 0.
                - Other directions: wdominant = 0.5.
            2. Evaluate the DTC metrics for both longitudinal and lateral distances to determine the risk levels: LongDSC and LatDSC.
            3. Using the dominant direction weighting, calculate the overall DTC score - DSC based on the weighted combination of longitudinal and lateral risks:
            DTC Score: DSC = LongDSC * wdominant +  LatDSC * (1-wdominant).
            4. Evaluate the TTC metrics for both longitudinal and lateral times to collision to determine the risk levels: LongTSC and LongTSC.
            5. Using the dominant direction weighting, calculate the overall TTC socre - TSC based on the weighted combination of longitudinal and lateral risks:
            TTC Score: TSC =  LongTSC * wdominant + LatTSC * (1-wdominant).
            6. Calculate the overall risk score by combining the DSC and TSC.
            Risk Score = 0.5 * DSC + 0.5 * TSC, rounded to the nearest integer.

            No matter how many obstacles are present, ensure all obstacles are included in the output with the following format:

            ### Safety analysis for the most recent timestep: Here's the evaluation of each obstacle according to the provided metrics and calculations.
            ### Obstacle Analysis:
                - Obstacle ID: <numeric ID>
                - Relative Direction: <Front/Back/Left/Right/front-left/front-right/back-left/back-right>
                - Distance Risk Reason: <description in context of DTClong and DTClat values and relative direction>
                - Longitudinal Distance Safety Score: <LongDSC>  
                - Lateral Distance Safety Score: <LatDSC>
                - Overall Distance Safety Score: <DSC>
                - Time Risk Reason: <description in context of TTClong and TTClat values and relative direction>
                - Longitudinal Time Safety Score: <LongTSC>
                - Lateral Time Safety Score: <LatTSC>
                - Overall Time Safety Score: <TSC>
                - Overall Risk Score: <Risk Score>

            ### Summary in JSON Format:  
            Summarize all obstacles with collision risk (Overall Risk Score is 0) and all obstacles with extreme risk (Overall Risk Score is 1) in the following JSON format. Make sure if they don't exist, set them as `null`:
            
            {{
                "CollisionObstacle": {{
                    "ObstacleID": "<Obstacle ID>",
                    "OverallRiskScore": "0"
                }},
                "ExtremeRiskObstacle": {{
                    "ObstacleID": "<Obstacle ID>",
                    "OverallRiskScore": "1"
                }},
                "PotentialAttackers": [
                    {{
                        "ObstacleID": "<Obstacle ID>",
                        "AttackPotential": "<High/Medium/Low>",
                        "RiskScore": "<Overall Risk Score>",
                        "AttackReason": "<Detailed explanation of why this obstacle could be an attacker>",
                        "PositionMetrics": {{
                            "Direction": "<relative direction>",
                            "DTClong": "<value>",
                            "DTClat": "<value>"
                        }},
                        "TimeMetrics": {{
                            "TTClong": "<value>",
                            "TTClat": "<value>"
                        }},
                        "MotionAnalysis": {{
                            "Pattern": "<Description of motion pattern>",
                            "Behavior": "<Description of concerning behavior>",
                            "Intent": "<Analysis of potential malicious intent>"
                        }},
                        "AttackFeasibility": {{
                            "EaseOfAttack": "<High/Medium/Low>",
                            "RequiredManeuvers": "<Description of maneuvers needed to attack>",
                            "SuccessProbability": "<High/Medium/Low>"
                        }}
                    }}
                ]
            }}
            """
        }
    ]

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_distance_risk(d):
    """Calculate risk score based on ABSOLUTE distance"""
    d_abs = abs(d)
    if d_abs < 0.3:
        return 0
    elif d_abs < 0.5:
        return 1
    elif d_abs < 1.0:
        return 2
    elif d_abs < 1.5:
        return 3
    elif d_abs < 2.0:
        return 4
    else:
        return 5

def process_safety_analysis(
    engine,
    base_dir,
    scenario_name,
    api_delay=5,
    max_timesteps=10  # Added parameter to control max number of timesteps to analyze
):
    """Process safety analysis for a scenario's last N timesteps in a single batch."""
    
    # Paths to CSV files
    ego_path = os.path.join(base_dir, scenario_name, "ego_trajectory.csv")
    relative_metrics_path = os.path.join(base_dir, scenario_name, "close_relative_metrics.csv")
    
    # Check if files exist before trying to read them
    if not os.path.exists(ego_path):
        print(f"  Warning: ego_trajectory.csv not found for scenario {scenario_name}")
        return
        
    if not os.path.exists(relative_metrics_path):
        print(f"  Warning: close_relative_metrics.csv not found for scenario {scenario_name}")
        return
    
    # Read the CSV files
    try:
        ego_df = pd.read_csv(ego_path)
        relative_metrics_df = pd.read_csv(relative_metrics_path)
        print(f"  Using filtered relative metrics with {len(relative_metrics_df)} entries")
        
        # Print column names to debug
        print(f"  Available columns in relative_metrics_df: {relative_metrics_df.columns.tolist()}")
    except Exception as e:
        print(f"  Error reading CSV files: {e}")
        return
    
    # Get all timesteps and the last N (up to max_timesteps)
    all_timesteps = sorted(ego_df['timestep'].unique())
    total_timesteps = len(all_timesteps)
    
    # Take the last N timesteps, but no more than what's available
    available_timesteps = min(max_timesteps, total_timesteps)
    last_timesteps = all_timesteps[-available_timesteps:]
    
    print(f"  Total timesteps in scenario: {total_timesteps}")
    print(f"  Processing last {len(last_timesteps)} timesteps in a single batch")
    
    # Create output directories with a more generic name
    output_subdir = f"last{len(last_timesteps)}"
    context_dir = os.path.join(base_dir, scenario_name, output_subdir, "safety", engine, "context")
    analysis_dir = os.path.join(base_dir, scenario_name, output_subdir, "safety", engine, "analysis")
    os.makedirs(context_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Generate combined context for all timesteps
    combined_context = ""
    included_obstacles = set()
    
    for timestep in last_timesteps:
        try:
            combined_context += f"At {timestep * 0.1:.1f} seconds:\n"
            
            # Only include relative metrics information
            timestep_metrics = relative_metrics_df[relative_metrics_df['timestep'] == timestep]
            if not timestep_metrics.empty:
                combined_context += "Relative metrics for this timestep:\n"
                for _, metric in timestep_metrics.iterrows():
                    obstacle_id = metric['obstacle_id']
                    included_obstacles.add(obstacle_id)
                    
                    # Use the correct column names (adjusted_d_long instead of d_long)
                    combined_context += (f"Obstacle {obstacle_id} is in the {metric['relative_direction']} of the ego car. "\
                                        f"The real distance is longitudinal {metric['adjusted_d_long']:.2f} m and lateral {metric['adjusted_d_lat']:.2f} m. " \
                                       f"The Longitudinal TTC is {metric['ttc_long']:.2f} and Lateral TTC is {metric['ttc_lat']:.2f}. " \
                                       f"Motion: {metric['motion_description']}\n")
            
            combined_context += "\n"  # Add a separator between timesteps
            
        except Exception as e:
            print(f"    Error generating context for timestep {timestep}: {str(e)}")
            continue
    
    # Save combined context
    context_file = os.path.join(context_dir, f"context_timesteps.txt")
    with open(context_file, 'w') as f:
        f.write(combined_context)
    
    print(f"  Generated combined context with {len(last_timesteps)} timesteps")
    print(f"  Included {len(included_obstacles)} unique obstacles")
    
    # Generate messages for LLM analysis
    messages = generate_safety_analysis(combined_context)
    
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
            output_file = os.path.join(analysis_dir, f"safety_analysis.txt")
            with open(output_file, 'w') as f:
                f.write(response)
            
            print(f"  Completed analysis for {len(last_timesteps)} timesteps")
            
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
    max_timesteps=10  # Add parameter to control max timesteps
):
    """Process scenarios focusing on the last N timesteps."""
    
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
                api_delay=api_delay,
                max_timesteps=max_timesteps  # Pass through max_timesteps
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
    API_DELAY = 15  # seconds
    
    # Maximum timesteps to process (change to 8 for 8-timestep scenarios)
    MAX_TIMESTEPS = 10  # Default value
    
    # Process a batch of 5 scenarios
    process_scenarios(
        engine=ENGINE,
        base_dir=CONFIG["base_dir"],
        single_scenario=True,  # Process multiple scenarios
        scenario_name="ARG_Carcarana-1_8_T-1",
        api_delay=API_DELAY,
        max_timesteps=MAX_TIMESTEPS  # Pass max_timesteps parameter
    )

if __name__ == "__main__":
    main() 