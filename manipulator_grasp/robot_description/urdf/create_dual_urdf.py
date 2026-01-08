import xml.etree.ElementTree as ET
import copy
import os

def create_dual_urdf():
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_path, "ur3e_ag95.urdf")
    output_file = os.path.join(base_path, "ur3e_ag95_dual.urdf")

    tree = ET.parse(input_file)
    root = tree.getroot()

    # Create new root
    new_root = ET.Element("robot", name="dual_ur3e_ag95")
    
    # Copy ros2_control (optional, might need prefixing too if used by planner/ros)
    # We will skip ros2_control for pure planning usage as it's for simulation control
    
    # 1. Create World Link
    world_link = ET.SubElement(new_root, "link", name="world")

    # Helper to process a robot copy
    def process_arm(prefix, y_offset):
        # We need to map old names to new names
        name_map = {"world": "world"} # World stays world
        
        # First pass sets up the name map
        for element in root:
            if element.tag in ["link", "joint"]:
                name = element.attrib.get("name")
                if name and name != "world":
                    name_map[name] = f"{prefix}_{name}"

        # Second pass copies and renames
        for element in root:
            if element.tag in ["link", "joint"]:
                # specific handling for base_joint (world connection)
                if element.attrib.get("name") == "base_joint":
                    continue # We create our own base joints
                # Skip duplicate world link
                if element.attrib.get("name") == "world":
                    continue

                new_elem = copy.deepcopy(element)
                
                # Update attributes
                if "name" in new_elem.attrib:
                    old_name = new_elem.attrib["name"]
                    if old_name in name_map:
                        new_elem.attrib["name"] = name_map[old_name]
                
                # Resolve absolute mesh paths in all descendants
                for child in new_elem.iter():
                    if "filename" in child.attrib:
                        fn = child.attrib["filename"]
                        if fn.startswith("../"):
                            abs_fn = os.path.normpath(os.path.join(base_path, fn))
                            child.attrib["filename"] = abs_fn
                
                # Update inner tags (parent, child, mimic)
                for child in new_elem:
                    if child.tag == "parent":
                        p = child.attrib.get("link")
                        if p in name_map:
                            child.attrib["link"] = name_map[p]
                    elif child.tag == "child":
                        c = child.attrib.get("link")
                        if c in name_map:
                            child.attrib["link"] = name_map[c]
                    elif child.tag == "mimic":
                        j = child.attrib.get("joint")
                        if j in name_map:
                            child.attrib["joint"] = name_map[j]
                            
                new_root.append(new_elem)
        
        # Add connection to world for this arm
        # <joint name="prefix_base_joint" type="fixed">
        #   <parent link="world"/>
        #   <child link="prefix_base_link"/>
        #   <origin xyz="0.8 y_offset 0.745" rpy="0 0 0"/>
        # </joint>
        
        # Note: In original URDF, base_joint connects world -> base_link with origin 1.0 0.6 0.745
        # We want our custom origin.
        # Left: 0.8 0.3 0.745
        # Right: 0.8 -0.3 0.745
        
        join_elem = ET.SubElement(new_root, "joint", name=f"{prefix}_base_joint", type="fixed")
        ET.SubElement(join_elem, "parent", link="world")
        ET.SubElement(join_elem, "child", link=f"{prefix}_base_link")
        # Quat/RPY: Original had quat via MuJoCo, here URDF uses RPY.
        # MuJoCo xml had quat="1 0 0 0" (identity).
        # We assume RPY 0 0 0.
        ET.SubElement(join_elem, "origin", xyz=f"0.8 {y_offset} 0.745", rpy="0 0 0")

    # Process Left Arm
    process_arm("left", 0.3)
    
    # Process Right Arm (rotated 180? No, in phase 1 they were same orientation)
    # Checking ur3e_ag95_dual.xml from previous turn:
    # <body name="left_ur3e_base" pos="0.8 0.3 0.745" ...>
    # <body name="right_ur3e_base" pos="0.8 -0.3 0.745" ...>
    # Both had quat="1 0 0 0" (Base orientation).
    process_arm("right", -0.3)

    # Write output
    tree = ET.ElementTree(new_root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Created {output_file}")

if __name__ == "__main__":
    create_dual_urdf()
