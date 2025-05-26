import os
import requests
import tarfile
from tqdm import tqdm

# Define the directory to save the dataset
SAVE_DIR = "./datasets/ms_marco"

# Define the URLs for the MS MARCO passage ranking dataset files
DATASET_FILES = {
    "collection.tar.gz": "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz",
    "queries.tar.gz": "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz",
    "qrels.dev.tsv": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv",
    "qrels.train.tsv": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv",
}

def download_file(url, destination_path):
    """Downloads a file from a URL to a destination path with a progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination_path, 'wb') as f, tqdm(
            desc=destination_path.split('/')[-1],
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                progress_bar.update(len(data))
        print(f"Successfully downloaded {destination_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while downloading {url}: {e}")
        return False

def extract_tar_gz(file_path, extract_to_dir):
    """Extracts a .tar.gz file."""
    try:
        print(f"Extracting {file_path}...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_to_dir)
        print(f"Successfully extracted {file_path} to {extract_to_dir}")
        # Optionally, remove the .tar.gz file after extraction
        # os.remove(file_path)
        # print(f"Removed {file_path}")
        return True
    except tarfile.ReadError as e:
        print(f"Error reading tar file {file_path}: {e}. It might be corrupted or not a tar.gz file.")
        return False
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return False

def main():
    # Create the save directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")
    else:
        print(f"Directory already exists: {SAVE_DIR}")

    # Download and extract each file
    for filename, url in DATASET_FILES.items():
        file_path = os.path.join(SAVE_DIR, filename)
        
        if os.path.exists(file_path):
            print(f"File {filename} already exists in {SAVE_DIR}. Skipping download.")
        elif filename.endswith(".tsv") and os.path.exists(file_path[:-4]): # Check if extracted .tsv exists
             print(f"Extracted file for {filename} likely already exists. Skipping download.")
        else:
            if not download_file(url, file_path):
                print(f"Skipping extraction for {filename} due to download error.")
                continue # Skip to next file if download failed

        # Extract if it's a .tar.gz file and was successfully downloaded or already exists
        if filename.endswith(".tar.gz"):
            # Check if the expected output of tar.gz already exists
            # For collection.tar.gz -> collection.tsv
            # For queries.tar.gz -> queries.tsv (assuming it contains queries.tsv or similar)
            extracted_file_name = filename.replace(".tar.gz", ".tsv") # A common pattern
            if filename == "collection.tar.gz": # collection.tar.gz contains collection.tsv
                 extracted_file_name = "collection.tsv"
            elif filename == "queries.tar.gz": # queries.tar.gz contains train.tsv, dev.tsv etc or just queries.tsv
                # We'll assume for now it extracts to a file like queries.tsv or similar
                # A more robust check would be to inspect the tar contents or know the exact structure
                 pass # Handled by checking if file_path (the .tar.gz) exists for extraction logic


            # More specific check for 'collection.tar.gz' which extracts to 'collection.tsv'
            if filename == "collection.tar.gz" and os.path.exists(os.path.join(SAVE_DIR, "collection.tsv")):
                print(f"Extracted file collection.tsv already exists. Skipping extraction of {filename}.")
            # More specific check for 'queries.tar.gz' which might extract to various .tsv files (e.g., queries.train.tsv)
            # For simplicity, we'll check if 'queries.tsv' exists as a proxy.
            # A truly robust solution would require knowing the exact contents of queries.tar.gz
            elif filename == "queries.tar.gz" and (os.path.exists(os.path.join(SAVE_DIR, "queries.tsv")) or os.path.exists(os.path.join(SAVE_DIR, "queries.train.tsv"))):
                 print(f"Extracted files from {filename} appear to exist. Skipping extraction.")
            elif os.path.exists(file_path): # Ensure the tarball exists before trying to extract
                extract_tar_gz(file_path, SAVE_DIR)
            else:
                print(f"Tarball {file_path} not found, cannot extract.")


    print("\nDataset download and extraction process finished.")
    print(f"Files are located in: {os.path.abspath(SAVE_DIR)}")

if __name__ == "__main__":
    main() 