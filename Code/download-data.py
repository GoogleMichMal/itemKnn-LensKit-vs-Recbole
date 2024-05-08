import os
import zipfile
import requests
from tqdm import tqdm


def download_and_extract(url, target_dir):
    # Download the ZIP file
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(
            f"Failed to download ZIP file from {url}. Status code: {response.status_code}"
        )
        return

    # Get the total file size in bytes
    total_size = int(response.headers.get("content-length", 0))

    # Initialize progress bar
    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)

    # Write the downloaded content to a file
    zip_file_path = os.path.join(target_dir, "Data.zip")
    with open(zip_file_path, "wb") as zip_file:
        for data in response.iter_content(chunk_size=1024):
            progress_bar.update(len(data))
            zip_file.write(data)
    progress_bar.close()

    # Extract the contents of the ZIP file
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    # Remove the downloaded ZIP file
    os.remove(zip_file_path)


def main():
    target_dir = "./" 
    data_folder = os.path.join(target_dir, "Data")

    # Check if the "Data" folder exists
    if not os.path.isdir(data_folder):
        # If "Data" folder doesn't exist, download and extract the ZIP file
        url = "https://github.com/GoogleMichMal/itemKnn-LensKit-vs-Recbole/releases/latest/download/Data.zip"
        download_and_extract(url, target_dir)
        print("Data folder downloaded and extracted successfully.")
    else:
        print("Data folder already exists.")


if __name__ == "__main__":
    main()
