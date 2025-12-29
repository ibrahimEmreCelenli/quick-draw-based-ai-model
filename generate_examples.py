import numpy as np
from PIL import Image
import os
import random

# Settings
DATA_PATH = 'data/dataset_all.npz'
OUTPUT_DIR = 'examples'
NUM_EXAMPLES = 10

def generate_examples():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # Create examples directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading dataset...")
    data = np.load(DATA_PATH)
    X = data['X']
    y = data['y']
    class_names = data['class_names']
    
    total_samples = len(X)
    print(f"Dataset loaded. Total samples: {total_samples}")
    
    # Select random indices
    indices = random.sample(range(total_samples), NUM_EXAMPLES)
    
    for idx in indices:
        # Get image and label
        img_array = X[idx] # (1, 28, 28) or (28, 28)
        
        # Handle shape
        if len(img_array.shape) == 3:
            img_array = img_array[0] # (28, 28)
            
        label_idx = y[idx]
        label_name = class_names[label_idx]
        
        # Invert colors for display (QuickDraw is white stroke on black usually, or vice versa)
        # We want it to look like a drawing on white paper for Gradio
        # The data is likely 0-255.
        # Let's save as is first. If it's black background, maybe invert.
        # Usually QuickDraw data is generic. Let's check mean.
        # If mean is low (<50), it's black background.
        
        if img_array.mean() < 127:
             # It's likely white stroke on black bg. Invert to Black stroke on White bg.
             img_array = 255 - img_array
             
        img = Image.fromarray(img_array.astype('uint8'))
        
        # Resize for better visibility (28x28 is tiny)
        img = img.resize((280, 280), Image.NEAREST)
        
        filename = f"{OUTPUT_DIR}/{label_name}_{idx}.png"
        img.save(filename)
        print(f"Saved {filename}")

if __name__ == "__main__":
    generate_examples()
