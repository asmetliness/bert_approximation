import os
from tqdm import tqdm

# Define paths and parameters
MS_MARCO_DIR = "./datasets/ms_marco"
MS_MARCO_FULL_COLLECTION = os.path.join(MS_MARCO_DIR, "collection.tsv")

TINY_DATASET_DIR = "./datasets/ms_marco_tiny"
TINY_COLLECTION_FILE = os.path.join(TINY_DATASET_DIR, "collection_tiny.tsv")

NUM_DOCUMENTS_TO_SELECT = 20000

def create_subset():
    """Creates a subset of the MS MARCO collection.tsv file."""
    # Check if the source file exists
    if not os.path.exists(MS_MARCO_FULL_COLLECTION):
        print(f"Error: Source file not found: {MS_MARCO_FULL_COLLECTION}")
        print(f"Please ensure you have downloaded and extracted the MS MARCO dataset first,")
        print(f"specifically the 'collection.tsv' file should be in '{MS_MARCO_DIR}'.")
        print("You can use the 'ms_marco_download.py' script for this.")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(TINY_DATASET_DIR):
        os.makedirs(TINY_DATASET_DIR)
        print(f"Created directory: {TINY_DATASET_DIR}")
    else:
        print(f"Output directory already exists: {TINY_DATASET_DIR}")

    print(f"Creating a subset of {NUM_DOCUMENTS_TO_SELECT} documents from {MS_MARCO_FULL_COLLECTION}...")
    print(f"Saving to: {TINY_COLLECTION_FILE}")

    try:
        with open(MS_MARCO_FULL_COLLECTION, 'r', encoding='utf-8') as infile, open(TINY_COLLECTION_FILE, 'w', encoding='utf-8') as outfile:
            
            # Using tqdm for a progress bar
            for i, line in enumerate(tqdm(infile, total=NUM_DOCUMENTS_TO_SELECT, desc="Processing documents")):
                if i >= NUM_DOCUMENTS_TO_SELECT:
                    break
                outfile.write(line)
        
        print(f"Successfully created subset: {TINY_COLLECTION_FILE}")
        # Verify the number of lines written
        with open(TINY_COLLECTION_FILE, 'r', encoding='utf-8') as verify_file:
            lines_written = sum(1 for _ in verify_file)
        print(f"The subset file contains {lines_written} documents.")
        if lines_written < NUM_DOCUMENTS_TO_SELECT and os.path.getsize(MS_MARCO_FULL_COLLECTION) > 0:
            # Get total lines in source if subset is smaller than expected and source is not empty
            print("Warning: The number of documents in the subset is less than requested.")
            # Counting lines in a large file can be slow, so this is a simplified check.
            # For a more accurate check of source lines, one would need to iterate through MS_MARCO_FULL_COLLECTION fully.
            print(f"This might happen if the source file ({MS_MARCO_FULL_COLLECTION}) has fewer than {NUM_DOCUMENTS_TO_SELECT} documents.")


    except FileNotFoundError:
        # This case should be caught by the initial check, but as a fallback:
        print(f"Error: Source file not found during processing: {MS_MARCO_FULL_COLLECTION}")
    except Exception as e:
        print(f"An error occurred during subset creation: {e}")

def main():
    create_subset()
    print("\nSubset creation process finished.")

if __name__ == "__main__":
    main() 