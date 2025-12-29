import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import os
import matplotlib.pyplot as plt
from model import QuickDrawCNN

# Constants
DATA_PATH = 'data/dataset_all.npz'
MODEL_PATH = 'models/quickdraw_model.pth'
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QuickDrawDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.from_numpy(X).float() # (N, 1, 28, 28)
        self.y = torch.from_numpy(y).long()
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        # Transform expects image (C, H, W) or PIL
        img = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def train():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run data_loader.py first.")
        
    print("Loading dataset...")
    data = np.load(DATA_PATH)
    X = data['X']
    y = data['y']
    class_names = data['class_names']
    
    # Check shape
    # X: (N, 1, 28, 28) usually uint8 0-255
    # y: (N,) int
    
    print(f"Data shape: {X.shape}, labels: {y.shape}")
    print(f"Classes: {len(class_names)}")
    
    # Normalize X to 0-1 float32
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.longlong)
    
    # Split Train/Val indices
    dataset_size = len(X)
    indices = list(range(dataset_size))
    split = int(0.1 * dataset_size)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    # Define Transforms
    # Images are already float 0-1, (1, 28, 28)
    # RandomAffine expects Tensor (C, H, W) is fine.
    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        # transforms.RandomHorizontalFlip() # Not good for some drawings but QuickDraw is messy
    ])
    
    train_dataset = QuickDrawDataset(X_train, y_train, transform=train_transform)
    val_dataset = QuickDrawDataset(X_val, y_val, transform=None)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Init Model
    model = QuickDrawCNN(num_classes=len(class_names)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Train Loop
    print("Starting training...")
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        total_step = len(train_loader)
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}")
            
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        correct_1 = 0
        correct_3 = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                
                # Top 1
                _, predicted = torch.max(outputs.data, 1)
                correct_1 += (predicted == labels).sum().item()
                
                # Top 3
                _, top3 = torch.topk(outputs.data, 3, dim=1)
                correct_3 += (top3 == labels.view(-1, 1)).sum().item()
                
                total += labels.size(0)
        
        val_acc_1 = 100 * correct_1 / total
        val_acc_3 = 100 * correct_3 / total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_train_loss:.4f}, Acc@1: {val_acc_1:.2f}%, Acc@3: {val_acc_3:.2f}%")
        history['train_loss'].append(avg_train_loss)
        history['val_acc'].append(val_acc_1)
        
    print("Training finished.")
    
    # 4. Save Model
    state = {
        'model_state': model.state_dict(),
        'class_names': class_names
    }
    torch.save(state, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Plot history (optional)
    # plt.plot(history['train_loss'])
    # plt.title('Training Loss')
    # plt.savefig('loss.png')

if __name__ == "__main__":
    train()
