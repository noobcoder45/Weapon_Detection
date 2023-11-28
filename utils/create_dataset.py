import os
import shutil
import random
#   valid valid valid
#a  600    80   30
#b  100    30    20
#c  1000   200  60
#d  100    1    10

def copy_files_by_names(source_folder, destination_folder, name_list, file_type):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Loop through the list of strings
    for name in name_list:
        # Construct the source file path
        source_file_path = os.path.join(source_folder, f"{name}.{file_type}")

        # Check if the source file exists
        if os.path.exists(source_file_path):
            # Construct the destination file path
            destination_file_path = os.path.join(destination_folder, f"{name}.{file_type}")

            # Copy the file to the destination folder
            shutil.copy2(source_file_path, destination_file_path)
        else:
            print(f"File '{name}.{file_type}' not found in '{source_folder}'")


def merge_folders(source_folders, destination_folders, n,file_type):
    # Create the destination folder if it doesn't exist
    # List to store all eligible files for copying
    # Loop through each source folder
    source_folder = source_folders[0]
    eligible_files = []
    # Loop through each file in the source folder
    for filename in os.listdir(source_folder):   
        # Check if the file is a JPG file
        if filename.lower().endswith(".jpg"):
            eligible_files.append(filename.split(".jpg")[0])
    # Check if there are enough eligible files to copy
    print(len(eligible_files))
    if len(eligible_files) < n[0]:
        print(f"Not enough eligible files. Found {len(eligible_files)} files, but requested {n[0]}.")
        return
    # Randomly select n files from the eligible files
    selected_files = random.sample(eligible_files, n[0])
    print(len(selected_files))
        # Copy the selected files to the destination folder
    print(file_type)
    for i in range(3):
        copy_files_by_names(source_folders[i],destination_folders[i],selected_files,file_type[i])


def x(fpath):
    files = os.listdir(fpath)
    return all(file==files[0] for file in files)

if __name__ == "__main__":
    result = x("/home/glutamicacid/Music/WEAPON_DETECTION/dataset creation/x")
    print(result)
    # file_type=["jpg","txt","txt"]
    # n=[1000]
    # source_folders = ["/home/glutamicacid/Music/WEAPON_DETECTION/dataset creation/OD/images"
    #                   ,"/home/glutamicacid/Music/WEAPON_DETECTION/dataset creation/OD/yolo_labels"
    #                   ,"/home/glutamicacid/Music/WEAPON_DETECTION/dataset creation/OD/rcnn_labels"]
    # destination_folder= ["/home/glutamicacid/Music/WEAPON_DETECTION/dataset creation/dataset/final_valid"
    #                         ,"/home/glutamicacid/Music/WEAPON_DETECTION/dataset creation/dataset/final_yolo_labels_valid"
    #                         ,"/home/glutamicacid/Music/WEAPON_DETECTION/dataset creation/dataset/final_rcnn_labels_valid"]
    # merge_folders(source_folders,destination_folder,n,file_type)
    # n=[30]

    # source_folders = ["/home/glutamicacid/Music/WEAPON_DETECTION/dataset creation/a/valid"
    #                   ,"/home/glutamicacid/Music/WEAPON_DETECTION/dataset creation/a/yolo_labels_valid"
    #                   ,"/home/glutamicacid/Music/WEAPON_DETECTION/dataset creation/a/rcnn_labels_valid"]
    # merge_folders(source_folders,destination_folder,n,file_type)
    # n=[20]
    # source_folders = ["/home/glutamicacid/Music/WEAPON_DETECTION/dataset creation/b/valid"
    #                   ,"/home/glutamicacid/Music/WEAPON_DETECTION/dataset creation/b/yolo_labels_valid"
    #                   ,"/home/glutamicacid/Music/WEAPON_DETECTION/dataset creation/b/rcnn_labels_valid"]
    # merge_folders(source_folders,destination_folder,n,file_type)
    # n=[60]
    
    # source_folders = ["/home/glutamicacid/Music/WEAPON_DETECTION/dataset creation/c/valid"
    #                   ,"/home/glutamicacid/Music/WEAPON_DETECTION/dataset creation/c/yolo_labels_valid"
    #                   ,"/home/glutamicacid/Music/WEAPON_DETECTION/dataset creation/c/rcnn_labels_valid"]
    # merge_folders(source_folders,destination_folder,n,file_type)
    # n=[10]

    # source_folders = ["/home/glutamicacid/Music/WEAPON_DETECTION/dataset creation/d/valid"
    #                     ,"/home/glutamicacid/Music/WEAPON_DETECTION/dataset creation/d/yolo_labels_valid"
    #                     ,"/home/glutamicacid/Music/WEAPON_DETECTION/dataset creation/d/rcnn_labels_valid"]

    # merge_folders(source_folders,destination_folder,n,file_type)