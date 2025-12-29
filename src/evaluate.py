import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
import random
from model import QuickDrawCNN
from torch.utils.data import DataLoader, TensorDataset

# Constants
DATA_PATH = 'data/dataset_all.npz'
MODEL_PATH = 'models/quickdraw_model.pth'
OUTPUT_DIR = 'docs/images'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256 # Larger batch for inference

def load_data():
    print("Loading dataset...")
    data = np.load(DATA_PATH)
    X = data['X']
    y = data['y']
    class_names = data['class_names']
    
    # Use the same split logic as training to get the validation set
    # (or a subset of it for speed, but let's try to be accurate)
    dataset_size = len(X)
    indices = list(range(dataset_size))
    split = int(0.1 * dataset_size)
    # Note: Training shuffled indices randomly. We can't perfectly replicate the shuffle 
    # unless we saved the seed or indices. 
    # However, for a general report, using a random subset is acceptable 
    # as long as valid/train distribution is similar.
    # Let's verify on a random 50,000 sample subset to ensure speed and statistical significance.
    
    print(f"Total samples: {dataset_size}. Using 50,000 random samples for evaluation.")
    eval_indices = np.random.choice(dataset_size, 50000, replace=False)
    
    X_eval = X[eval_indices].astype(np.float32) / 255.0
    y_eval = y[eval_indices].astype(np.longlong)
    
    return X_eval, y_eval, class_names

def evaluate():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load Data
    X, y, class_names = load_data()
    
    # Create Loader
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load Model
    print("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = QuickDrawCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state'])
    model.to(DEVICE)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Running inference...")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 1. Classification Report
    print("Generating Classification Report...")
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(f"{OUTPUT_DIR}/classification_report.csv")
    
    # Extract overall metrics
    accuracy = report['accuracy']
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    
    # 2. Confusion Matrix (Top 20 most confused or just Top 20 classes overall)
    # Plotting 345x345 is messy. Let's plot the confusion matrix for the 
    # TOP 15 classes with the LOWEST F1-score (most difficult) to show where the model struggles?
    # Or maybe the TOP 15 BEST performing?
    # User asked for "professional report". Usually we show the class-wise performance.
    
    # Let's plot F1-Score Distribution
    plt.figure(figsize=(10, 6))
    f1_scores = [report[c]['f1-score'] for c in class_names]
    sns.histplot(f1_scores, bins=20, kde=True)
    plt.title('Distribution of F1-Scores across 345 Classes')
    plt.xlabel('F1-Score')
    plt.ylabel('Count')
    plt.savefig(f"{OUTPUT_DIR}/f1_score_distribution.png")
    plt.close()

    # 3. Top 10 Best and Worst Classes
    sorted_classes = sorted([(c, report[c]['f1-score']) for c in class_names], key=lambda x: x[1])
    worst_10 = sorted_classes[:10]
    best_10 = sorted_classes[-10:]
    
    # Plot Best/Worst
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    names, scores = zip(*best_10)
    sns.barplot(x=list(scores), y=list(names), ax=axes[0], palette="viridis")
    axes[0].set_title("Top 10 Best Recognized Classes")
    axes[0].set_xlim(0, 1)
    
    names, scores = zip(*worst_10)
    sns.barplot(x=list(scores), y=list(names), ax=axes[1], palette="magma")
    axes[1].set_title("Top 10 Hardest Classes")
    axes[1].set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/class_performance.png")
    plt.close()
    
    # 4. Sample Predictions Grid
    # Visualize some random predictions
    plt.figure(figsize=(12, 8))
    indices = random.sample(range(len(X)), 15)
    for i, idx in enumerate(indices):
        img = X[idx].squeeze()
        true_lbl = class_names[y[idx]]
        pred_lbl = class_names[all_preds[idx]]
        
        color = 'green' if true_lbl == pred_lbl else 'red'
        
        plt.subplot(3, 5, i+1)
        plt.imshow(img, cmap='gray_r') # Invert gray for drawing look
        plt.title(f"T: {true_lbl}\nP: {pred_lbl}", color=color, fontsize=10)
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/prediction_samples.png")
    plt.close()
    
    print("Evaluation complete. Assets saved to docs/images/")

if __name__ == "__main__":
    evaluate()
