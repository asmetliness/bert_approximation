import os
import nltk
from nltk.corpus import reuters

# Define the directory to save the dataset information or a marker file
SAVE_DIR = "./datasets/reuters"
DOWNLOAD_MARKER = os.path.join(SAVE_DIR, ".reuters_downloaded")

def download_reuters_dataset():
    """Downloads the Reuters dataset using nltk if not already downloaded."""
    try:
        # Create the save directory if it doesn't exist
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
            print(f"Created directory: {SAVE_DIR}")
        else:
            print(f"Directory already exists: {SAVE_DIR}")

        # Check if a marker file exists, indicating dataset was likely processed
        if os.path.exists(DOWNLOAD_MARKER):
            print(f"Reuters dataset already appears to be downloaded and processed (marker file found: {DOWNLOAD_MARKER}).")
            print(f"NLTK typically stores data in: {nltk.data.path}")
            # You can access files using reuters.fileids()
            # and content using reuters.raw(fileid)
            return

        print("Attempting to download Reuters dataset via NLTK...")
        # NLTK's download function for specific corpora
        nltk.download('reuters')
        # Check if the download was successful by trying to access it
        _ = reuters.fileids() # This will raise an error if 'reuters' is not found
        print("Reuters dataset downloaded successfully via NLTK.")
        print(f"NLTK typically stores data in one of the following paths: {nltk.data.path}")
        print("You can access the corpus using `from nltk.corpus import reuters`.")
        
        # Create a marker file to indicate successful download and processing by this script
        with open(DOWNLOAD_MARKER, 'w') as f:
            f.write("NLTK Reuters corpus downloaded.")
        print(f"Created marker file: {DOWNLOAD_MARKER}")

    except Exception as e:
        print(f"Error downloading or accessing Reuters dataset via NLTK: {e}")
        print("Please ensure you have an internet connection and NLTK is properly configured.")
        print("You might need to run `nltk.download('popular')` or `nltk.download('all')` in a Python interpreter once.")

def main():
    download_reuters_dataset()
    print("\nReuters dataset download process finished.")
    print(f"The dataset is managed by NLTK. You can find NLTK's data path(s) by checking nltk.data.path.")
    print(f"A marker file has been placed in: {os.path.abspath(SAVE_DIR)}")

if __name__ == "__main__":
    main() 