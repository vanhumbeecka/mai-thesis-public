import os
import shutil


def move_files_to_nas(source_dir, nas_dir):
    # Check if both source and NAS directories exist
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return

    if not os.path.exists(nas_dir):
        print(f"NAS directory '{nas_dir}' does not exist.")
        return

    # Iterate through files in the source directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            source_file_path = os.path.join(root, file)
            destination_file_path = os.path.join(
                nas_dir, os.path.relpath(source_file_path, source_dir)
            )

            # Move files to NAS, merging if files with identical names exist
            try:
                shutil.move(source_file_path, destination_file_path)
                print(f"Moved '{source_file_path}' to '{destination_file_path}'")
            except shutil.Error as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    # Replace these paths with your source and NAS directories
    source_directory = "/Users/avanhum/helioviewer"
    nas_directory = "/Volumes/home/ml/datasets/helioviewer"

    move_files_to_nas(source_directory, nas_directory)
