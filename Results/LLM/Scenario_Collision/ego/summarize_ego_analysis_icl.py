#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path

def generate_icl_summary(output_dir):
    """Generate the ego_analysis_summary.json from icl analysis outputs."""
    # Base directory for analysis files
    base_dir = "/home/yuan/ITSC/Safety/output_validation"
    
    # Find all analysis files - updated path to match where files are actually stored
    analysis_files = glob.glob(os.path.join(base_dir, "*", "all10", "ego_icl", "openai", "analysis", "ego_analysis_all_timesteps.txt"))
    
    # Process all files
    summary_data = []
    scenarios_missing_json = []  # Track scenarios missing JSON output
    
    for file_path in analysis_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Extract scenario name from file path
                parts = file_path.split(os.sep)
                validation_idx = parts.index("output_validation")
                scenario_name = parts[validation_idx + 1]
                
                # Try to parse as JSON first
                try:
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    
                    if json_start != -1 and json_end != -1:
                        json_content = content[json_start:json_end]
                        json_content = json_content.strip()
                        
                        # Check if it's valid JSON
                        try:
                            data = json.loads(json_content)
                            
                            # Create summary entry from JSON - handle different JSON formats
                            # Check different possible keys for collision obstacles
                            collision_obstacles = None
                            if "CollisionObstacles" in data:
                                collision_obstacles = data.get("CollisionObstacles", [])
                            elif "CollisionObstacle" in data:
                                # CollisionObstacle might be null or a different format
                                collision_obstacle = data.get("CollisionObstacle", None)
                                if collision_obstacle and isinstance(collision_obstacle, dict) and "ObstacleID" in collision_obstacle:
                                    collision_obstacles = [collision_obstacle]
                                elif collision_obstacle and isinstance(collision_obstacle, list):
                                    collision_obstacles = collision_obstacle
                            
                            # Check different possible sources for IsCollisionScenario
                            is_collision = False
                            if "ScenarioAnalysis" in data and "IsCollisionScenario" in data["ScenarioAnalysis"]:
                                is_collision = data["ScenarioAnalysis"]["IsCollisionScenario"]
                            elif "IsCollisionScenario" in data:
                                is_collision = data["IsCollisionScenario"]
                            
                            # Check different possible sources for reasoning/justification
                            reasoning = ""
                            if "ScenarioAnalysis" in data and "Reasoning" in data["ScenarioAnalysis"]:
                                reasoning = data["ScenarioAnalysis"]["Reasoning"]
                            elif "Justification" in data:
                                reasoning = data["Justification"]
                            elif "Reasoning" in data:
                                reasoning = data["Reasoning"]
                            
                            # Handle different formats for extreme risk obstacle
                            extreme_risk = None
                            if "ExtremeRiskObstacle" in data:
                                extreme_risk = data["ExtremeRiskObstacle"]
                                # If ExtremeRiskObstacle has ObstacleID as an array, take the first one
                                if extreme_risk and isinstance(extreme_risk, dict) and "ObstacleID" in extreme_risk and isinstance(extreme_risk["ObstacleID"], list):
                                    extreme_risk = {"ObstacleID": extreme_risk["ObstacleID"][0], "OverallRiskScore": extreme_risk.get("OverallRiskScore", 1)}
                            
                            summary_entry = {
                                "Scenario": scenario_name,
                                "IsCollisionScenario": is_collision,
                                "CollisionObstacleIDs": collision_obstacles,
                                "ExtremeRiskObstacleID": extreme_risk,
                                "Reasoning": reasoning,
                                "JsonFormat": True  # Flag to indicate JSON was found
                            }
                            
                            summary_data.append(summary_entry)
                            
                        except json.JSONDecodeError:
                            # Found text that looks like JSON but isn't valid
                            scenarios_missing_json.append({
                                "Scenario": scenario_name, 
                                "Reason": "Invalid JSON format",
                                "Content": json_content[:100] + "..." if len(json_content) > 100 else json_content,
                                "FilePath": file_path
                            })
                    else:
                        # No JSON-like structure found, try to parse the text
                        scenarios_missing_json.append({
                            "Scenario": scenario_name, 
                            "Reason": "No JSON structure found",
                            "FilePath": file_path
                        })
                        
                        # Attempt to parse text format
                        is_collision = False
                        collision_ids = []
                        extreme_risk_id = None
                        reasoning = ""
                        
                        # Parse the text content
                        lines = content.split('\n')
                        current_obstacle = None
                        current_risk_score = None
                        
                        for line in lines:
                            line = line.strip()
                            
                            # Check for obstacle ID
                            if line.startswith('- Obstacle ID:'):
                                current_obstacle = line.split(':')[1].strip()
                            
                            # Check for overall risk score
                            elif line.startswith('- Overall Risk score:'):
                                current_risk_score = int(line.split(':')[1].strip())
                                
                                # If risk score is 0 or 1, it's a collision or extreme risk
                                if current_risk_score <= 1:
                                    if current_risk_score == 0:
                                        is_collision = True
                                        collision_ids.append({"ObstacleID": current_obstacle, "OverallRiskScore": current_risk_score})
                                    else:
                                        extreme_risk_id = {"ObstacleID": current_obstacle, "OverallRiskScore": current_risk_score}
                            
                            # Collect reasoning
                            elif line.startswith('* Conclusion:'):
                                reasoning += line[13:].strip() + " "
                        
                        # Create summary entry from text analysis
                        summary_entry = {
                            "Scenario": scenario_name,
                            "IsCollisionScenario": is_collision,
                            "CollisionObstacleIDs": collision_ids,
                            "ExtremeRiskObstacleID": extreme_risk_id,
                            "Reasoning": reasoning.strip(),
                            "JsonFormat": False  # Flag to indicate JSON was NOT found
                        }
                        
                        summary_data.append(summary_entry)
                    
                except Exception as e:
                    scenarios_missing_json.append({
                        "Scenario": scenario_name, 
                        "Reason": f"Error parsing: {str(e)}",
                        "FilePath": file_path
                    })
                
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            continue
    
    # Sort by scenario name for consistency
    summary_data.sort(key=lambda x: x["Scenario"])
    
    # Save the summary data
    output_path = os.path.join(output_dir, "ego_analysis_summary.json")
    with open(output_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Save the list of scenarios missing JSON
    missing_json_path = os.path.join(output_dir, "scenarios_missing_json.json")
    with open(missing_json_path, 'w') as f:
        json.dump(scenarios_missing_json, f, indent=2)
    
    # Print summary statistics
    total_scenarios = len(summary_data)
    if total_scenarios > 0:
        json_scenarios = sum(1 for entry in summary_data if entry.get("JsonFormat", False))
        text_scenarios = total_scenarios - json_scenarios
        
        print(f"Summary statistics:")
        print(f"- Total scenarios processed: {total_scenarios}")
        print(f"- Scenarios with valid JSON: {json_scenarios} ({json_scenarios/total_scenarios*100:.2f}%)")
        print(f"- Scenarios without JSON (text parsed): {text_scenarios} ({text_scenarios/total_scenarios*100:.2f}%)")
        print(f"- Details of scenarios without JSON saved to: {missing_json_path}")
    else:
        print("No scenarios were processed.")
    
    return output_path

def analyze_discrepancy(annotations_path, icl_json_path, output_dir):
    """Analyze the discrepancy between the icl analysis and the ground truth annotations."""
    # Load annotations
    annotations_df = pd.read_csv(annotations_path)
    annotations_dict = dict(zip(annotations_df['Scenario_Name'], annotations_df['Obstacle_ID']))
    
    # Load icl analysis
    with open(icl_json_path, 'r') as f:
        icl_data = json.load(f)
    
    # Initialize results storage
    results = []
    processed_scenarios = set()
    
    # Process each scenario
    for scenario_data in icl_data:
        scenario_name = scenario_data['Scenario']
        processed_scenarios.add(scenario_name)
        
        # Skip if scenario not in annotations
        if scenario_name not in annotations_dict:
            continue
        
        # Get annotated obstacle ID
        annotated_obstacle_id = annotations_dict[scenario_name]
        
        # Get model predictions
        is_collision_scenario = scenario_data['IsCollisionScenario']
        collision_obstacle_ids = scenario_data.get('CollisionObstacleIDs', []) or []  # Handle None case
        extreme_risk_obstacle_id = scenario_data['ExtremeRiskObstacleID']
        
        # Extract the obstacle ID from the extreme risk object if it's a dictionary
        if isinstance(extreme_risk_obstacle_id, dict) and 'ObstacleID' in extreme_risk_obstacle_id:
            extreme_risk_obstacle_id = extreme_risk_obstacle_id['ObstacleID']
        
        # Clean up obstacle IDs (remove decimal points)
        cleaned_collision_ids = []
        for id_obj in collision_obstacle_ids:
            try:
                if isinstance(id_obj, dict) and 'ObstacleID' in id_obj:
                    id_str = id_obj['ObstacleID']
                else:
                    id_str = id_obj
                # Try to convert to integer
                cleaned_id = int(float(id_str))
                cleaned_collision_ids.append(cleaned_id)
            except:
                # If conversion fails, keep the original string
                if isinstance(id_obj, dict) and 'ObstacleID' in id_obj:
                    cleaned_collision_ids.append(id_obj['ObstacleID'])
                else:
                    cleaned_collision_ids.append(id_obj)
        
        # Check if annotated obstacle ID is in collision IDs
        annotated_id_in_collision_ids = annotated_obstacle_id in cleaned_collision_ids
        
        # Check if annotated obstacle ID is the ONLY collision ID (exactly one match)
        annotated_id_is_only_collision_id = (len(cleaned_collision_ids) == 1 and 
                                             annotated_obstacle_id in cleaned_collision_ids)
        
        # Check if annotated obstacle ID is the extreme risk obstacle
        try:
            extreme_risk_id = int(float(extreme_risk_obstacle_id)) if extreme_risk_obstacle_id else None
            annotated_id_is_extreme_risk = extreme_risk_id == annotated_obstacle_id
        except:
            annotated_id_is_extreme_risk = False
        
        # Check if the model correctly identified the collision scenario
        correct_collision_identification = is_collision_scenario and annotated_id_is_only_collision_id
        
        # Create result entry
        result = {
            "Scenario": scenario_name,
            "AnnotatedObstacleID": annotated_obstacle_id,
            "IsCollisionScenario": is_collision_scenario,
            "CollisionObstacleIDs": collision_obstacle_ids,
            "CleanedCollisionIDs": cleaned_collision_ids,
            "ExtremeRiskObstacleID": extreme_risk_obstacle_id,
            "AnnotatedIDInCollisionIDs": annotated_id_in_collision_ids,
            "AnnotatedIDIsOnlyCollisionID": annotated_id_is_only_collision_id,
            "AnnotatedIDIsExtremeRisk": annotated_id_is_extreme_risk,
            "CorrectlyIdentified": annotated_id_is_only_collision_id or annotated_id_is_extreme_risk,
            "CorrectCollisionIdentification": correct_collision_identification,
            "JsonFormat": scenario_data.get("JsonFormat", True)  # Include JSON format status
        }
        
        results.append(result)
    
    # Check for missing scenarios
    missing_scenarios = set(annotations_dict.keys()) - processed_scenarios
    if missing_scenarios:
        print(f"Warning: {len(missing_scenarios)} scenarios in annotations are missing from the icl data:")
        for scenario in sorted(missing_scenarios):
            print(f"  - {scenario}")
    
    # Create a DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Calculate metrics
        total_scenarios = len(annotations_dict)  # Total number of scenarios in annotations
        processed_scenarios_count = len(results)
        collision_scenarios = sum(df['IsCollisionScenario'])
        extreme_risk_scenarios = sum(df['ExtremeRiskObstacleID'].notna())
        
        # Calculate correct identifications separately for collision and extreme risk
        correct_collision_only = sum(df['AnnotatedIDIsOnlyCollisionID'])
        correct_extreme_risk_only = sum(df['AnnotatedIDIsExtremeRisk'])
        correct_either = sum(df['CorrectlyIdentified'])
        
        # Calculate metrics based on JSON format
        json_format_scenarios = sum(df['JsonFormat'])
        text_format_scenarios = processed_scenarios_count - json_format_scenarios
        
        # Save to CSV
        csv_path = os.path.join(output_dir, "discrepancy_analysis.csv")
        df.to_csv(csv_path, index=False)
        
        # Create a summary text file
        summary_path = os.path.join(output_dir, "discrepancy_analysis_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"# Discrepancy Analysis Summary\n\n")
            f.write(f"## Dataset Statistics\n")
            f.write(f"Total scenarios in annotations: {total_scenarios}\n")
            f.write(f"Scenarios processed: {processed_scenarios_count}\n")
            f.write(f"Collision scenarios (model prediction): {collision_scenarios} ({collision_scenarios/processed_scenarios_count*100:.2f}%)\n")
            f.write(f"Extreme risk scenarios (model prediction): {extreme_risk_scenarios} ({extreme_risk_scenarios/processed_scenarios_count*100:.2f}%)\n")
            f.write(f"Scenarios with JSON format: {json_format_scenarios} ({json_format_scenarios/processed_scenarios_count*100:.2f}%)\n")
            f.write(f"Scenarios with text-only format: {text_format_scenarios} ({text_format_scenarios/processed_scenarios_count*100:.2f}%)\n\n")
            
            f.write(f"## Identification Accuracy\n")
            f.write(f"Correctly identified as ONLY collision: {correct_collision_only} ({correct_collision_only/processed_scenarios_count*100:.2f}%)\n")
            f.write(f"Correctly identified as extreme risk: {correct_extreme_risk_only} ({correct_extreme_risk_only/processed_scenarios_count*100:.2f}%)\n")
            f.write(f"Correctly identified as either collision or extreme risk: {correct_either} ({correct_either/processed_scenarios_count*100:.2f}%)\n\n")
            
            f.write(f"## Detailed Analysis\n")
            f.write(f"Scenarios where annotated obstacle is the ONLY collision ID: {sum(df['AnnotatedIDIsOnlyCollisionID'])} ({sum(df['AnnotatedIDIsOnlyCollisionID'])/processed_scenarios_count*100:.2f}%)\n")
            f.write(f"Scenarios where annotated obstacle is in collision IDs (may be multiple): {sum(df['AnnotatedIDInCollisionIDs'])} ({sum(df['AnnotatedIDInCollisionIDs'])/processed_scenarios_count*100:.2f}%)\n")
            f.write(f"Scenarios where annotated obstacle is the extreme risk obstacle: {sum(df['AnnotatedIDIsExtremeRisk'])} ({sum(df['AnnotatedIDIsExtremeRisk'])/processed_scenarios_count*100:.2f}%)\n")
            f.write(f"Scenarios where annotated obstacle is either the ONLY collision ID or is the extreme risk obstacle: {correct_either} ({correct_either/processed_scenarios_count*100:.2f}%)\n\n")
            
            # Add analysis of JSON vs text format accuracy
            if json_format_scenarios > 0:
                json_format_df = df[df['JsonFormat']]
                json_correct = sum(json_format_df['CorrectlyIdentified'])
                f.write(f"## JSON Format Performance\n")
                f.write(f"JSON format scenarios: {json_format_scenarios}\n")
                f.write(f"Correctly identified in JSON format: {json_correct} ({json_correct/json_format_scenarios*100:.2f}%)\n\n")
            
            if text_format_scenarios > 0:
                text_format_df = df[~df['JsonFormat']]
                text_correct = sum(text_format_df['CorrectlyIdentified'])
                f.write(f"## Text Format Performance\n")
                f.write(f"Text format scenarios: {text_format_scenarios}\n")
                f.write(f"Correctly identified in text format: {text_correct} ({text_correct/text_format_scenarios*100:.2f}%)\n\n")
            
            if missing_scenarios:
                f.write(f"## Missing Scenarios\n")
                for scenario in sorted(missing_scenarios):
                    f.write(f"- {scenario}\n")
                f.write("\n")
            
            f.write(f"## Scenarios with Correct Collision Identification (Only One Collision ID)\n")
            for _, row in df[df['CorrectCollisionIdentification']].iterrows():
                f.write(f"- {row['Scenario']}: Annotated ID {row['AnnotatedObstacleID']}, Collision IDs {row['CollisionObstacleIDs']}\n")
            
            f.write(f"\n## Scenarios with Annotated Obstacle as ONLY Collision ID\n")
            for _, row in df[df['AnnotatedIDIsOnlyCollisionID']].iterrows():
                f.write(f"- {row['Scenario']}: Annotated ID {row['AnnotatedObstacleID']}, Collision IDs {row['CollisionObstacleIDs']}\n")
            
            f.write(f"\n## Scenarios with Annotated Obstacle in Multiple Collision IDs\n")
            for _, row in df[df['AnnotatedIDInCollisionIDs'] & ~df['AnnotatedIDIsOnlyCollisionID']].iterrows():
                f.write(f"- {row['Scenario']}: Annotated ID {row['AnnotatedObstacleID']}, Collision IDs {row['CollisionObstacleIDs']}\n")
            
            f.write(f"\n## Scenarios with Annotated Obstacle as Extreme Risk\n")
            for _, row in df[df['AnnotatedIDIsExtremeRisk']].iterrows():
                f.write(f"- {row['Scenario']}: Annotated ID {row['AnnotatedObstacleID']}, Extreme Risk ID {row['ExtremeRiskObstacleID']}\n")
            
            f.write(f"\n## Scenarios with Incorrect Collision Identification\n")
            for _, row in df[df['IsCollisionScenario'] & ~df['AnnotatedIDInCollisionIDs']].iterrows():
                f.write(f"- {row['Scenario']}: Annotated ID {row['AnnotatedObstacleID']}, Collision IDs {row['CollisionObstacleIDs']}\n")
            
            f.write(f"\n## Scenarios with Missed Collision Identification\n")
            for _, row in df[~df['IsCollisionScenario'] & df['AnnotatedIDInCollisionIDs']].iterrows():
                f.write(f"- {row['Scenario']}: Annotated ID {row['AnnotatedObstacleID']}, Collision IDs {row['CollisionObstacleIDs']}\n")
            
            # Add section for scenarios missing JSON
            f.write(f"\n## Scenarios without JSON Format\n")
            for _, row in df[~df['JsonFormat']].iterrows():
                f.write(f"- {row['Scenario']}\n")
        
        print(f"Analysis complete. Results saved to {output_dir}")
        print(f"Summary: {total_scenarios} total scenarios, {processed_scenarios_count} processed, {collision_scenarios} collisions, {extreme_risk_scenarios} extreme risks, {correct_either} correctly identified")
        print(f"Format analysis: {json_format_scenarios} JSON format, {text_format_scenarios} text format")
    else:
        print("No results found to analyze.")

if __name__ == "__main__":
    # Set the paths specific to ego
    annotations_path = "/home/yuan/ITSC/Safety/llm/all10_analysis/ego/annotations.csv"
    output_dir = "/home/yuan/ITSC/Safety/llm/all10_analysis/ego/icl_openai"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # First generate the icl summary
    print("Generating icl summary...")
    icl_json_path = generate_icl_summary(output_dir)
    
    # Then analyze the discrepancy
    print("\nAnalyzing discrepancies...")
    analyze_discrepancy(annotations_path, icl_json_path, output_dir) 