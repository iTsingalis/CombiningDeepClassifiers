import os
import requests
from tqdm import tqdm

root_download = '/media/blue/tsingalis/IARIADevIDFusion/datasets'


def download_and_organize(urls):
    pbar = tqdm(urls)

    for url in pbar:
        pbar.set_description("Processing %s" % url)

        # Extract the path from the URL
        path_parts = url.split('/')
        folder_path = os.path.join(*path_parts[3:-1])  # Ignore the domain and filename

        # Create the folder if it doesn't exist
        os.makedirs(os.path.join(root_download, folder_path), exist_ok=True)

        # Get the filename from the URL
        filename = path_parts[-1]
        if os.path.exists(os.path.join(root_download, folder_path, filename)):
            pbar.set_description("Already processed %s" % url)
            continue
        # Download the file
        response = requests.get(url)
        if response.status_code == 200:
            file_path = os.path.join(root_download, folder_path, filename)
            with open(file_path, 'wb') as file:
                file.write(response.content)
            # print(f"Downloaded: {url} -> {file_path}")
        else:
            print(f"Failed to download: {url}")


if __name__ == "__main__":

    # VISION_statistics_files = 'VISION_statistics_files.txt'
    VISION_files = 'metadataVISION/VISION_base_files.txt'

    with open(VISION_files, 'r') as file:
        url_list = [line.strip() for line in file]

    download_and_organize(url_list)
