#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
import glob
from pathlib import Path

def generate_cot_summary(output_dir):
    """Generate the ego_analysis_summary.json from CoT analysis outputs."""
    # Base directory for analysis files
    base_dir = "/home/yuan/ITSC/Safety/output_validation_normal"
    
    # Find all analysis files
    analysis_files = glob.glob(os.path.join(base_dir, "*", "agent_first", "ego_cot", "gemini", "analysis", "ego_analysis_all_timesteps.txt"))
    
    # Process all files
    summary_data = []
    scenarios_missing_json = []  # Track scenarios missing JSON output
    
    for file_path in analysis_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Extract scenario name from file path
                parts = file_path.split(os.sep)
                validation_idx = parts.index("output_validation_normal")
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
                            
                            # Create summary entry from JSON
                            summary_entry = {
                                "Scenario": scenario_name,
                                "CollisionObstacleIDs": data.get("CollisionObstacles", []),
                                "ExtremeRiskObstacleID": data.get("ExtremeRiskObstacle", None),
                                "PredictedCollisionObstacle": data.get("PredictedCollisionObstacle", None),
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
                                        collision_ids.append({"ObstacleID": current_obstacle, "OverallRiskScore": current_risk_score})
                                    else:
                                        extreme_risk_id = {"ObstacleID": current_obstacle, "OverallRiskScore": current_risk_score}
                            
                            # Collect reasoning
                            elif line.startswith('* Conclusion:'):
                                reasoning += line[13:].strip() + " "
                        
                        # Create summary entry from text analysis
                        summary_entry = {
                            "Scenario": scenario_name,
                            "CollisionObstacleIDs": collision_ids,
                            "ExtremeRiskObstacleID": extreme_risk_id,
                            "PredictedCollisionObstacle": None,  # Set to None for text-parsed data
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

def analyze_discrepancy(annotations_path, cot_json_path, output_dir):
    """Analyze false positives in normal scenarios where obstacles are incorrectly identified."""
    # Load annotations
    annotations_df = pd.read_csv(annotations_path)
    annotations_dict = dict(zip(annotations_df['Scenario_Name'], annotations_df['Obstacle_ID']))
    
    # Load CoT analysis
    with open(cot_json_path, 'r') as f:
        cot_data = json.load(f)
    
    # Initialize results storage
    results = []
    processed_scenarios = set()
    
    # Counters for false positives
    total_false_positive_collision_obstacles = 0
    total_false_positive_extreme_risk_obstacles = 0
    total_false_positive_predicted_collision_obstacles = 0
    
    # Process each scenario
    for scenario_data in cot_data:
        scenario_name = scenario_data['Scenario']
        processed_scenarios.add(scenario_name)
        
        # Skip if scenario not in annotations
        if scenario_name not in annotations_dict:
            continue
        
        # Get model predictions
        collision_obstacle_ids = scenario_data.get('CollisionObstacleIDs', []) or []  # Handle None case
        extreme_risk_obstacle_id = scenario_data['ExtremeRiskObstacleID']
        predicted_collision_obstacle = scenario_data.get('PredictedCollisionObstacle', None)
        
        # Clean up collision obstacle IDs
        cleaned_collision_ids = []
        for id_obj in collision_obstacle_ids:
            if isinstance(id_obj, dict) and 'ObstacleID' in id_obj:
                cleaned_collision_ids.append(id_obj['ObstacleID'])
            else:
                cleaned_collision_ids.append(id_obj)
        
        # Check if extreme risk obstacle has actual content (not just empty dictionary)
        has_extreme_risk_obstacle = bool(extreme_risk_obstacle_id is not None and (
            (isinstance(extreme_risk_obstacle_id, dict) and 
             extreme_risk_obstacle_id and  # Check if dict is not empty
             'ObstacleID' in extreme_risk_obstacle_id and
             extreme_risk_obstacle_id['ObstacleID'])  # Check if ObstacleID has value
            or (not isinstance(extreme_risk_obstacle_id, dict) and extreme_risk_obstacle_id)  # If not dict, check if has value
        ))
        
        # Count false positives (any identified obstacle is a false positive in normal scenarios)
        has_collision_obstacles = len(cleaned_collision_ids) > 0
        has_predicted_collision_obstacle = bool(predicted_collision_obstacle is not None and predicted_collision_obstacle)
        
        # Count total false positives
        num_collision_obstacles = len(cleaned_collision_ids)
        total_false_positive_collision_obstacles += num_collision_obstacles
        total_false_positive_extreme_risk_obstacles += (1 if has_extreme_risk_obstacle else 0)
        total_false_positive_predicted_collision_obstacles += (1 if has_predicted_collision_obstacle else 0)
        
        # Create result entry
        result = {
            "Scenario": scenario_name,
            "CollisionObstacleIDs": collision_obstacle_ids,
            "CleanedCollisionIDs": cleaned_collision_ids,
            "ExtremeRiskObstacleID": extreme_risk_obstacle_id,
            "PredictedCollisionObstacle": predicted_collision_obstacle,
            "HasCollisionObstacles": has_collision_obstacles,
            "HasExtremeRiskObstacle": has_extreme_risk_obstacle,
            "HasPredictedCollisionObstacle": has_predicted_collision_obstacle,
            "NumCollisionObstacles": num_collision_obstacles,
            "IncorrectIdentification": has_collision_obstacles or has_extreme_risk_obstacle or has_predicted_collision_obstacle,
            "JsonFormat": scenario_data.get("JsonFormat", True)
        }
        
        results.append(result)
    
    # Check for missing scenarios
    missing_scenarios = set(annotations_dict.keys()) - processed_scenarios
    if missing_scenarios:
        print(f"Warning: {len(missing_scenarios)} scenarios in annotations are missing from the CoT data:")
        for scenario in sorted(missing_scenarios):
            print(f"  - {scenario}")
    
    # Create a DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Calculate metrics
        total_scenarios = len(df)
        
        # False positive scenarios
        scenarios_with_collision_obstacles = sum(df['HasCollisionObstacles'])
        scenarios_with_extreme_risk_obstacle = sum(df['HasExtremeRiskObstacle'])
        scenarios_with_predicted_collision = sum(df['HasPredictedCollisionObstacle'])
        scenarios_with_any_incorrect = sum(df['IncorrectIdentification'])
        
        # Calculate metrics based on JSON format
        json_format_scenarios = sum(df['JsonFormat'])
        text_format_scenarios = total_scenarios - json_format_scenarios
        
        # Save to CSV
        csv_path = os.path.join(output_dir, "false_positive_analysis.csv")
        df.to_csv(csv_path, index=False)
        
        # Create a summary text file
        summary_path = os.path.join(output_dir, "false_positive_analysis_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"# False Positive Analysis Summary for Normal Scenarios\n\n")
            f.write(f"## Dataset Statistics\n")
            f.write(f"Total scenarios processed: {total_scenarios}\n")
            f.write(f"Scenarios with JSON format: {json_format_scenarios} ({json_format_scenarios/total_scenarios*100:.2f}%)\n")
            f.write(f"Scenarios with text-only format: {text_format_scenarios} ({text_format_scenarios/total_scenarios*100:.2f}%)\n\n")
            
            f.write(f"## False Positive Statistics\n")
            f.write(f"Total false positive collision obstacles identified: {total_false_positive_collision_obstacles}\n")
            f.write(f"Total false positive extreme risk obstacles identified: {total_false_positive_extreme_risk_obstacles}\n")
            f.write(f"Total false positive predicted collision obstacles identified: {total_false_positive_predicted_collision_obstacles}\n\n")
            
            f.write(f"Scenarios with false positive collision obstacles: {scenarios_with_collision_obstacles} ({scenarios_with_collision_obstacles/total_scenarios*100:.2f}%)\n")
            f.write(f"Scenarios with false positive extreme risk obstacle: {scenarios_with_extreme_risk_obstacle} ({scenarios_with_extreme_risk_obstacle/total_scenarios*100:.2f}%)\n")
            f.write(f"Scenarios with false positive predicted collision: {scenarios_with_predicted_collision} ({scenarios_with_predicted_collision/total_scenarios*100:.2f}%)\n")
            f.write(f"Scenarios with any false positive identification: {scenarios_with_any_incorrect} ({scenarios_with_any_incorrect/total_scenarios*100:.2f}%)\n\n")
            
            # Add analysis of JSON vs text format accuracy
            if json_format_scenarios > 0:
                json_format_df = df[df['JsonFormat']]
                json_incorrect = sum(json_format_df['IncorrectIdentification'])
                json_fp_collision = sum(json_format_df['HasCollisionObstacles'])
                json_fp_extreme_risk = sum(json_format_df['HasExtremeRiskObstacle'])
                json_fp_predicted = sum(json_format_df['HasPredictedCollisionObstacle'])
                
                f.write(f"## JSON Format Performance\n")
                f.write(f"JSON format scenarios: {json_format_scenarios}\n")
                f.write(f"JSON format scenarios with false positives: {json_incorrect} ({json_incorrect/json_format_scenarios*100:.2f}%)\n")
                f.write(f"JSON format scenarios with collision obstacles: {json_fp_collision} ({json_fp_collision/json_format_scenarios*100:.2f}%)\n")
                f.write(f"JSON format scenarios with extreme risk obstacle: {json_fp_extreme_risk} ({json_fp_extreme_risk/json_format_scenarios*100:.2f}%)\n")
                f.write(f"JSON format scenarios with predicted collision: {json_fp_predicted} ({json_fp_predicted/json_format_scenarios*100:.2f}%)\n\n")
            
            if text_format_scenarios > 0:
                text_format_df = df[~df['JsonFormat']]
                text_incorrect = sum(text_format_df['IncorrectIdentification'])
                text_fp_collision = sum(text_format_df['HasCollisionObstacles'])
                text_fp_extreme_risk = sum(text_format_df['HasExtremeRiskObstacle'])
                text_fp_predicted = sum(text_format_df['HasPredictedCollisionObstacle'])
                
                f.write(f"## Text Format Performance\n")
                f.write(f"Text format scenarios: {text_format_scenarios}\n")
                f.write(f"Text format scenarios with false positives: {text_incorrect} ({text_incorrect/text_format_scenarios*100:.2f}%)\n")
                f.write(f"Text format scenarios with collision obstacles: {text_fp_collision} ({text_fp_collision/text_format_scenarios*100:.2f}%)\n")
                f.write(f"Text format scenarios with extreme risk obstacle: {text_fp_extreme_risk} ({text_fp_extreme_risk/text_format_scenarios*100:.2f}%)\n")
                f.write(f"Text format scenarios with predicted collision: {text_fp_predicted} ({text_fp_predicted/text_format_scenarios*100:.2f}%)\n\n")
            
            if missing_scenarios:
                f.write(f"## Missing Scenarios\n")
                for scenario in sorted(missing_scenarios):
                    f.write(f"- {scenario}\n")
                f.write("\n")
            
            f.write(f"\n## Scenarios with False Positive Collision Obstacles\n")
            for _, row in df[df['HasCollisionObstacles']].iterrows():
                f.write(f"- {row['Scenario']}: Collision IDs: {row['CollisionObstacleIDs']}\n")
            
            f.write(f"\n## Scenarios with False Positive Extreme Risk Obstacle\n")
            for _, row in df[df['HasExtremeRiskObstacle']].iterrows():
                f.write(f"- {row['Scenario']}: Extreme Risk ID: {row['ExtremeRiskObstacleID']}\n")
            
            f.write(f"\n## Scenarios with False Positive Predicted Collision\n")
            for _, row in df[df['HasPredictedCollisionObstacle']].iterrows():
                f.write(f"- {row['Scenario']}: Predicted Collision: {row['PredictedCollisionObstacle']}\n")
        
        # Print summary
        print(f"Analysis complete. Results saved to {output_dir}")
        print(f"False positive statistics:")
        print(f"- Total scenarios: {total_scenarios}")
        print(f"- Scenarios with false positive collision obstacles: {scenarios_with_collision_obstacles} ({scenarios_with_collision_obstacles/total_scenarios*100:.2f}%)")
        print(f"- Scenarios with false positive extreme risk obstacle: {scenarios_with_extreme_risk_obstacle} ({scenarios_with_extreme_risk_obstacle/total_scenarios*100:.2f}%)")
        print(f"- Scenarios with false positive predicted collision: {scenarios_with_predicted_collision} ({scenarios_with_predicted_collision/total_scenarios*100:.2f}%)")
        print(f"- Scenarios with any false positive: {scenarios_with_any_incorrect} ({scenarios_with_any_incorrect/total_scenarios*100:.2f}%)")
        print(f"- Total false positive collision obstacles: {total_false_positive_collision_obstacles}")
        print(f"- Total false positive extreme risk obstacles: {total_false_positive_extreme_risk_obstacles}")
        print(f"- Total false positive predicted collision obstacles: {total_false_positive_predicted_collision_obstacles}")
    else:
        print("No results found to analyze.")

if __name__ == "__main__":
    annotations_path = "/home/yuan/ITSC/Safety/llm/agent_first/ego/annotation_normal.csv"
    output_dir = "/home/yuan/ITSC/Safety/llm/agent_first/ego/cot"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)  
    
    # First generate the CoT summary
    print("Generating CoT summary...")
    cot_json_path = generate_cot_summary(output_dir)
    
    # Then analyze the false positives in normal scenarios
    print("\nAnalyzing false positives in normal scenarios...")
    analyze_discrepancy(annotations_path, cot_json_path, output_dir) 