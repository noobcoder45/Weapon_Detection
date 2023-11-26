import os
import xml.etree.ElementTree as ET

def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name 
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
            
        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))
            
            info_dict['image_size'] = tuple(image_size)
        
        # Get details of the bounding box 
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                    
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)            
            info_dict['bboxes'].append(bbox)
    
    return info_dict

def process_directory(directory_path, create_path):
    if not os.path.exists(create_path):
        os.makedirs(create_path)

    for filename in os.listdir(directory_path):
        if filename.endswith(".xml"):
            xml_path = os.path.join(directory_path, filename)
            create_and_write_file(xml_path, create_path)

def create_and_write_file(xml_filename, final_path):
    info_dict = extract_info_from_xml(xml_filename)
    output_filename = os.path.join(final_path, os.path.basename(xml_filename).replace(".xml", ".txt"))

    with open(output_filename, 'w') as output_file:    
        for b  in info_dict['bboxes']:
            try:
                class_id = class_name_to_id_mapping[b["class"]]
            except KeyError:
                print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())

            b_center_x = (b["xmin"] + b["xmax"]) / 2 
            b_center_y = (b["ymin"] + b["ymax"]) / 2
            b_width    = (b["xmax"] - b["xmin"])
            b_height   = (b["ymax"] - b["ymin"])

            # Normalise the co-ordinates by the dimensions of the image
            image_w, image_h, image_c = info_dict["image_size"]  
            b_center_x /= image_w 
            b_center_y /= image_h 
            b_width    /= image_w 
            b_height   /= image_h 

            # Append the YOLO-formatted string to the print buffer

            # Write the YOLO-formatted string to the output file
            output_file.write("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
            if b != info_dict["bboxes"][-1]:
                output_file.write("\n")
           

# Dictionary that maps class names to IDs
class_name_to_id_mapping = {"pistol": 0,
                           "stop": 1,
                           "speedlimit": 2,
                           "crosswalk": 3}


# Replace 'your_directory_path' and 'your_output_directory' with the actual paths
directory_path = '.'
output_directory = 'yolo'

process_directory(directory_path, output_directory)