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

types =[]

# no = ['Knife', 'Grenade']

def create_and_write_file(xml_filename, final_path):
    info_dict = extract_info_from_xml(xml_filename)
    output_filename = os.path.join(final_path, os.path.basename(xml_filename).replace(".xml", ".txt"))

    with open(output_filename, 'w') as output_file:
        
        boxes = info_dict['bboxes']
        output_file.write(str(len(boxes)))

        for box in boxes:
            output_file.write('\n')
            if box["class"] not in types:
                types.append(box["class"])
            
            # if box['class'] in no:
            #     continue
            output_file.write(
                f"{box['xmin']} {box['ymin']} {box['xmax']} {box['ymax']}   "
            )

# Replace 'your_directory_path' and 'your_output_directory' with the actual paths
directory_path = './Y1.v2i.voc/valid'
output_directory = './Y1.v2i.voc/rcnn_labels_valid'

process_directory(directory_path, output_directory)

print(types)