import os
import json

def extract_important_information(json_file_path):
    # Read the JSON file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Extract important information
    important_info = {
        "timeStepSize": data.get("timeStepSize"),
        "location": {
            "geoNameId": data.get("location", {}).get("geoNameId", {}).get("text", "Unknown"),  # Provide default "Unknown"
            "gpsLatitude": data.get("location", {}).get("gpsLatitude", {}).get("text", "Unknown"),
            "gpsLongitude": data.get("location", {}).get("gpsLongitude", {}).get("text", "Unknown")
        },
        "scenarioTags": list(data.get("scenarioTags", {}).keys()),
        "lanelet": [],
        "trafficSign": [],
        "dynamicObstacle": [],
        "planningProblem": {}
    }

    # Extract lanelet details
    # Loop over lanelets
    for lanelet in data.get("lanelet", []):
        # Prepare lanelet information
        lanelet_info = {
            "id": lanelet.get("id"),
            "leftBound": [
                {
                    "x": point.get("x", {}).get("text", 0.0),
                    "y": point.get("y", {}).get("text", 0.0)
                }
                for point in lanelet.get("leftBound", {}).get("point", [])
            ],
            "rightBound": [
                {
                    "x": point.get("x", {}).get("text", 0.0),
                    "y": point.get("y", {}).get("text", 0.0)
                }
                for point in lanelet.get("rightBound", {}).get("point", [])
            ],
            "trafficSignRef": []  # Initialize
        }

        # Handle trafficSignRef
        traffic_sign_ref = lanelet.get("trafficSignRef")
        if isinstance(traffic_sign_ref, dict):
            lanelet_info["trafficSignRef"] = [traffic_sign_ref.get("ref", "None")]
        elif isinstance(traffic_sign_ref, list):
            lanelet_info["trafficSignRef"] = [
                ref.get("ref", "None") for ref in traffic_sign_ref if isinstance(ref, dict)
            ]
        else:
            lanelet_info["trafficSignRef"] = ["None"]

        # Append to important_info
        important_info["lanelet"].append(lanelet_info)


    # Extract traffic sign details
    for traffic_sign in data.get("trafficSign", []):
        traffic_sign_info = {
            "id": traffic_sign.get("id"),  # Extract the traffic sign ID

            # Extract traffic sign element details
            "trafficSignElement": {
                "trafficSignID": traffic_sign.get("trafficSignElement", {}).get("trafficSignID", {}).get("text", "Unknown"),
                "additionalValue": traffic_sign.get("trafficSignElement", {}).get("additionalValue", {}).get("text", "Unknown")
            },

        # Extract position details
        "position": {
            "x": traffic_sign.get("position", {}).get("point", {}).get("x", {}).get("text", 0.0),  # Default to 0.0 if x is missing
            "y": traffic_sign.get("position", {}).get("point", {}).get("y", {}).get("text", 0.0)   # Default to 0.0 if y is missing
        },

            # Extract virtual flag
            "virtual": traffic_sign.get("virtual", "false")  # Default to "false" if virtual is missing
        }

        # Append the extracted traffic sign info to the list in important_info
        important_info["trafficSign"].append(traffic_sign_info)

   # Extract dynamic obstacle details
    for obstacle in data.get("dynamicObstacle", []):
        obstacle_info = {
            "id": obstacle.get("id"),  # Extract the obstacle ID
            "type": obstacle.get("type", {}).get("text"),  # Extract the obstacle type

            # Extract the shape details (length and width)
            "shape": {
                "length": obstacle.get("shape", {}).get("rectangle", {}).get("length", {}).get("text"),
                "width": obstacle.get("shape", {}).get("rectangle", {}).get("width", {}).get("text")
            },

            # Extract the initial state details
            "initialState": {
                "position": {
                    "x": obstacle.get("initialState", {}).get("position", {}).get("point", {}).get("x", {}).get("text"),
                    "y": obstacle.get("initialState", {}).get("position", {}).get("point", {}).get("y", {}).get("text")
                },
                "orientation": obstacle.get("initialState", {}).get("orientation", {}).get("exact", {}).get("text"),
                "time": obstacle.get("initialState", {}).get("time", {}).get("exact", {}).get("text"),
                "velocity": obstacle.get("initialState", {}).get("velocity", {}).get("exact", {}).get("text"),
                "acceleration": obstacle.get("initialState", {}).get("acceleration", {}).get("exact", {}).get("text")
            },

            # Initialize an empty list for trajectory
            "trajectory": []
        }

        # Handle the trajectory data
        trajectory = obstacle.get("trajectory", {}).get("state", [])
        if isinstance(trajectory, dict):  # If the trajectory is a single object, convert to a list
            trajectory = [trajectory]

        # Iterate over trajectory states and extract details
        for state in trajectory:
            trajectory_info = {
                "position": {
                    "x": state.get("position", {}).get("point", {}).get("x", {}).get("text"),
                    "y": state.get("position", {}).get("point", {}).get("y", {}).get("text")
                },
                "orientation": state.get("orientation", {}).get("exact", {}).get("text"),
                "time": state.get("time", {}).get("exact", {}).get("text"),
                "velocity": state.get("velocity", {}).get("exact", {}).get("text"),
                "acceleration": state.get("acceleration", {}).get("exact", {}).get("text")

            }
            obstacle_info["trajectory"].append(trajectory_info)

        # Append the processed obstacle information to the list in important_info
        important_info["dynamicObstacle"].append(obstacle_info)

    # Extract planning problem details
    planning_problem = data.get("planningProblem", {})
    important_info["planningProblem"] = {
        "id": planning_problem.get("id"),
        "initialState": {
            "position": {
                "x": planning_problem.get("initialState", {}).get("position", {}).get("point", {}).get("x", {}).get("text"),
                "y": planning_problem.get("initialState", {}).get("position", {}).get("point", {}).get("y", {}).get("text")
            },
            "orientation": planning_problem.get("initialState", {}).get("orientation", {}).get("exact", {}).get("text"),
            "time": planning_problem.get("initialState", {}).get("time", {}).get("exact", {}).get("text"),
            "velocity": planning_problem.get("initialState", {}).get("velocity", {}).get("exact", {}).get("text"),
            "acceleration": planning_problem.get("initialState", {}).get("acceleration", {}).get("exact", {}).get("text"),
            "yawRate": planning_problem.get("initialState", {}).get("yawRate", {}).get("exact", {}).get("text"),
            "slipAngle": planning_problem.get("initialState", {}).get("slipAngle", {}).get("exact", {}).get("text")
        },
        "goalState": {
            "position": {
                "lanelet": planning_problem.get("goalState", {}).get("position", {}).get("lanelet", {}).get("ref")
            },
            "time": {
                "intervalStart": planning_problem.get("goalState", {}).get("time", {}).get("intervalStart", {}).get("text"),
                "intervalEnd": planning_problem.get("goalState", {}).get("time", {}).get("intervalEnd", {}).get("text")
            },
            "velocity": {
                "intervalStart": planning_problem.get("goalState", {}).get("velocity", {}).get("intervalStart", {}).get("text"),
                "intervalEnd": planning_problem.get("goalState", {}).get("velocity", {}).get("intervalEnd", {}).get("text")
            }
        }
    }

    return important_info

def assign_layers(important_info):
    # Layer 1: Road-Level
    L1 = {
        "lanelet": important_info.get("lanelet", [])
    }
    # Layer 2: Traffic Infrastructure
    L2 = {
        "trafficSign": important_info.get("trafficSign", [])
            # Generate traffic sign associations
    }
    # Generate traffic sign associations
    # Layer 3: Temporal Modifications
    # Extend this layer to include temporal modifications like roadwork, traffic changes, etc.
    L3 = {
        "roadwork": [],  # Placeholder: Add data here if available
        "temporaryChanges": []  # Placeholder: Add data here if available
    }

    # Layer 4: Movable Objects
    L4 = {
        "dynamicObstacle": important_info.get("dynamicObstacle", [])
    }

    # Layer 5: Environmental Conditions
    # Update this layer to dynamically extract weather and environmental conditions if the data is present.
    location_info = important_info.get("location", {})
    L5 = {
        "weather": location_info.get("weather", "Unknown"),  # Placeholder: Add weather if available
        "timeOfDay": location_info.get("timeOfDay", "Unknown"),  # Placeholder: Add time of day if available
        "season": location_info.get("season", "Unknown"),  # Placeholder: Add season if available
        "visibility": location_info.get("visibility", "Unknown")  # Placeholder: Add visibility info
    }

    # Layer 6: Digital Information
    # Includes location metadata and scenario tags.
    L6 = {
        "location": {
            "geoNameId": location_info.get("geoNameId", "Unknown"),
            "gpsLatitude": location_info.get("gpsLatitude", "Unknown"),
            "gpsLongitude": location_info.get("gpsLongitude", "Unknown")
        },
        "scenarioTags": important_info.get("scenarioTags", [])
    }

    # Layer 7: Planning Problem
    L7 = {
        "planningProblem": important_info.get("planningProblem", {})
    }

    # Combine all layers
    layers = {
        "L1_RoadLevel": L1,
        "L2_TrafficInfrastructure": L2,
        "L3_TemporalModifications": L3,
        "L4_MovableObjects": L4,
        "L5_EnvironmentalConditions": L5,
        "L6_DigitalInformation": L6,
        "L7_PlanningProblem": L7
    }
    
    return layers
