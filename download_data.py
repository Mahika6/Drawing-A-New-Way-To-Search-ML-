# download_data.py

import requests
from pathlib import Path
from tqdm import tqdm
import urllib.parse

# The base URL where the Quick, Draw! .npy files are stored
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

# The 50 class names your project uses
CLASSES = [
    "airplane", "alarm clock", "ambulance", "angel", "ant", "apple", "axe", "banana", 
    "bandage", "barn", "baseball", "baseball bat", "basket", "basketball", "bathtub", 
    "beach", "bear", "bed", "bee", "belt", "bench", "bicycle", "binoculars", "bird", 
    "birthday cake", "blackberry", "blueberry", "book", "boomerang", "bottlecap", 
    "bowtie", "bracelet", "brain", "bread", "bridge", "broccoli", "broom", "bucket", 
    "bulldozer", "bus", "bush", "butterfly", "cactus", "cake", "calculator", 
    "calendar", "camel", "camera", "camouflage", "campfire"
]

# The directory where the data will be saved
DATA_DIR = Path("Quick_draw_dataset_50-samples")

def download_data():
    """Downloads all 50 .npy files for the project."""
    print(f"Starting download of {len(CLASSES)} data files...")
    
    # Create the data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)
    
    for class_name in tqdm(CLASSES, desc="Downloading files"):
        # Handle spaces in class names for the URL (e.g., "alarm clock" -> "alarm%20clock")
        encoded_class_name = urllib.parse.quote(class_name)
        file_url = f"{BASE_URL}{encoded_class_name}.npy"
        file_path = DATA_DIR / f"{class_name}.npy"

        # Download the file if it doesn't already exist
        if not file_path.exists():
            try:
                response = requests.get(file_url, stream=True)
                response.raise_for_status() # Raise an exception for bad status codes

                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except requests.exceptions.RequestException as e:
                print(f"\nError downloading {class_name}: {e}")
                # Clean up partially downloaded file
                if file_path.exists():
                    file_path.unlink()
    
    print("\n✅ Download complete!")

if __name__ == "__main__":
    download_data()

# To run this script, execute: python download_data.py