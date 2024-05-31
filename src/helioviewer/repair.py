import os

def rename_files(folder_path):
    # List all files in the specified directory
    files = os.listdir(folder_path)

    for filename in files:
        if '-' in filename:
            # Replace '/' with an empty string in the filename
            new_filename = filename.replace('-', '')

            # Construct the old and new file paths
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed '{filename}' to '{new_filename}'")

def rename_content(file_path):
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()

    # Replace '/' with an empty string in the file content
    new_content = content.replace(':', '')

    # Write the new file content
    with open(file_path, 'w') as file:
        file.write(new_content)


if __name__ == "__main__":
    # Replace 'path_to_your_folder' with the path to your specified folder
    folder_path = '/Users/avanhum/helioviewer/'

    # Call the function to rename files in the specified folder
    # rename_files(folder_path)

    rename_content(os.path.join(folder_path, "downloaded_files_2011.txt"))
    rename_content(os.path.join(folder_path, "downloaded_files_2010.txt"))