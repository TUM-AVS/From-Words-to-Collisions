import os
import glob
import json
import xml.etree.ElementTree as ET

def xml_to_dict(element):
    # Convert an XML element and its children to a dictionary
    node = {}
    if element.attrib:
        node.update(element.attrib)
    for child in element:
        child_dict = xml_to_dict(child)
        if child.tag not in node:
            node[child.tag] = child_dict
        else:
            if not isinstance(node[child.tag], list):
                node[child.tag] = [node[child.tag]]
            node[child.tag].append(child_dict)
    if element.text and element.text.strip():
        node['text'] = element.text.strip()
    return node

def xml_file_to_json(xml_file_path, json_file_path):
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Convert the XML tree to a dictionary
    xml_dict = xml_to_dict(root)

    # Convert the dictionary to a JSON string
    json_data = json.dumps(xml_dict, indent=4)

    # Write the JSON string to a file
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_data)

def convert_single_xml_to_json(xml_file_path, destination_dir):
    # Create the destination directory if it does not exist
    os.makedirs(destination_dir, exist_ok=True)

    # Generate the corresponding JSON file path
    json_file_name = os.path.splitext(os.path.basename(xml_file_path))[0] + '.json'
    json_file_path = os.path.join(destination_dir, json_file_name)

    # Convert XML to JSON
    xml_file_to_json(xml_file_path, json_file_path)
    print(f"Converted {xml_file_path} to {json_file_path}")

def convert_all_xml_to_json(source_dir, destination_dir):
    # Create the destination directory if it does not exist
    os.makedirs(destination_dir, exist_ok=True)

    # Get all XML files in the source directory
    xml_files = glob.glob(os.path.join(source_dir, "*.xml"))

    # Convert each XML file to JSON
    for xml_file_path in xml_files:
        # Generate the corresponding JSON file path
        json_file_name = os.path.splitext(os.path.basename(xml_file_path))[0] + '.json'
        json_file_path = os.path.join(destination_dir, json_file_name)

        # Convert XML to JSON
        xml_file_to_json(xml_file_path, json_file_path)
        print(f"Converted {xml_file_path} to {json_file_path}")
