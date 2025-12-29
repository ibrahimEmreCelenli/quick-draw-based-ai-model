import requests
import numpy as np
import os
import tqdm
from concurrent.futures import ThreadPoolExecutor

# Constants
DATA_DIR = '../data'
BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
CATEGORIES_URL = 'https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt'
SAMPLES_PER_CLASS = 5000  # Increased to 5000
IMAGE_SIZE = 28
NUM_WORKERS = 8

os.makedirs(DATA_DIR, exist_ok=True)

def get_categories():
    print("Fetching categories list...")
    r = requests.get(CATEGORIES_URL)
    categories = [line.strip() for line in r.text.splitlines() if line.strip()]
    print(f"Found {len(categories)} categories.")
    return categories

def download_category_data(category):
    # Normalize category name for URL
    category_url_name = category.replace(' ', '%20')
    url = f"{BASE_URL}{category_url_name}.npy"
    
    # Calculate bytes to download
    # 5000 images * 784 bytes = 3,920,000 bytes.
    # Header is small (<1KB).
    # Safe bet: 4,000,000 bytes (~4MB) covers it easily for top 5000.
    
    try:
        # Download header + data
        headers = {'Range': f'bytes=0-{4000000}'} 
        response = requests.get(url, headers=headers)
        
        if response.status_code not in [200, 206]:
            print(f"Failed to download {category}: HTTP {response.status_code}")
            return None
            
        content = response.content
        
        # Parse numpy header manually to robustness
        if content[:6] != b'\x93NUMPY':
            print(f"Invalid numpy header for {category}")
            return None
            
        header_len = int.from_bytes(content[8:10], byteorder='little')
        offset = 10 + header_len
        
        data_bytes = content[offset:]
        
        # Ensure we have enough bytes
        required_bytes = SAMPLES_PER_CLASS * IMAGE_SIZE * IMAGE_SIZE
        if len(data_bytes) < required_bytes:
            # If slightly less, maybe range wasn't enough?
            # Retry with larger range if needed, or just take what we got (unlikely to fail with 785k)
            # Actually, let's just use what we have, truncated to nearest 784
            valid_len = (len(data_bytes) // (IMAGE_SIZE*IMAGE_SIZE)) * (IMAGE_SIZE*IMAGE_SIZE)
            data_bytes = data_bytes[:valid_len]
        else:
             data_bytes = data_bytes[:required_bytes]

        data = np.frombuffer(data_bytes, dtype=np.uint8)
        data = data.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
        
        # Ensure exactly SAMPLES_PER_CLASS
        if data.shape[0] > SAMPLES_PER_CLASS:
            data = data[:SAMPLES_PER_CLASS]
            
        return data
        
    except Exception as e:
        print(f"Error processing {category}: {e}")
        return None

def main():
    categories = get_categories()
    categories.sort() # Ensure consistent order
    
    # Map categories to indices
    class_names = categories
    
    X_list = []
    y_list = []
    
    print(f"Downloading {SAMPLES_PER_CLASS} samples for {len(categories)} categories...")
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm.tqdm(executor.map(download_category_data, categories), total=len(categories)))
        
    successful = 0
    for idx, data in enumerate(results):
        if data is not None and len(data) > 0:
            X_list.append(data)
            # Create labels
            y_list.append(np.full(len(data), idx))
            successful += 1
        else:
            print(f"Warning: No data for {categories[idx]}")
            
    print(f"Downloaded data for {successful}/{len(categories)} categories.")
    
    if not X_list:
        print("No data collected in X_list. Exiting.")
        return

    # Concatenate
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    print(f"Final dataset shape: X={X.shape}, y={y.shape}")
    
    # Normalize (0-255 -> 0-1) and add channel dim for PyTorch (N, 1, 28, 28) ?? 
    # Or keep as (N, 28, 28) and transform in Dataset?
    # Let's output valid numpy arrays. PyTorch expects (N, C, H, W).
    # We will expand dims here to be safe and ready.
    X = np.expand_dims(X, axis=1) # (N, 1, 28, 28)
    
    output_path = os.path.join(DATA_DIR, 'dataset_all.npz')
    np.savez_compressed(output_path, X=X, y=y, class_names=class_names)
    print(f"Saved dataset to {output_path}")

if __name__ == "__main__":
    main()
