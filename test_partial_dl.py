
import requests
import numpy as np
import io

def download_partial_npy(url, n_rows=1000):
    # Step 1: Download first 1KB to get header
    resp_header = requests.get(url, headers={'Range': 'bytes=0-1023'})
    content = resp_header.content
    
    # Parse header to find where data starts
    # .npy format: Magic(6) + Version(2) + HeaderLen(2) + Header(HeaderLen)
    magic = content[:6]
    if magic != b'\x93NUMPY':
        raise ValueError("Not a numpy file")
        
    major, minor = content[6], content[7]
    header_len = int.from_bytes(content[8:10], byteorder='little')
    total_header_size = 10 + header_len
    
    # Parse the header verify shape/dtype (usually {'descr': '|u1', 'fortran_order': False, 'shape': (N, 784), }
    header_str = content[10:10+header_len].decode('ascii').strip()
    print(f"Header: {header_str}")
    
    # Each row is 784 bytes (28x28 uint8)
    row_size = 784 
    needed_bytes = total_header_size + (n_rows * row_size)
    
    # Step 2: Download the exact amount needed
    print(f"Downloading {needed_bytes} bytes for {n_rows} rows...")
    resp = requests.get(url, headers={'Range': f'bytes=0-{needed_bytes-1}'})
    
    # Load into numpy
    # Since we have the partial valid file (header + partial data), we can try loading it if we fix the shape in header
    # OR we can just read the raw bytes after the header
    data_bytes = resp.content[total_header_size:]
    arr = np.frombuffer(data_bytes, dtype=np.uint8)
    
    # Reshape
    real_rows = len(arr) // 784
    arr = arr[:real_rows*784].reshape(real_rows, 784)
    return arr

# Test with Apple
url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/apple.npy'
data = download_partial_npy(url, n_rows=100)
print(f"Successfully loaded array with shape: {data.shape}")
print(f"Sample values: {data[0, 200:210]}")
