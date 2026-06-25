import os
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import sys

# Ensure models module is found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.tcn import TCN
from sklearn.utils.class_weight import compute_class_weight

# 1. Configuration
DATASET_DIR = "Dataset"
NUM_FRAMES = 10
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. ResNet Feature Extractor
print("Loading ResNet-18 Feature Extractor...")
resnet = models.resnet18(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1]) # Remove FC layer
resnet = resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Load Videos and Extract Features
classes = {"armflapping": 0, "headbanging": 1, "spinning": 2}
all_features = []
all_labels = []

print("Extracting features from videos...")
for cls_name, cls_idx in classes.items():
    folder_path = os.path.join(DATASET_DIR, cls_name)
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist.")
        continue
        
    videos = []
    videos.extend(glob.glob(os.path.join(folder_path, "*.mp4")))
    videos.extend(glob.glob(os.path.join(folder_path, "*.webm")))
    
    print(f"Found {len(videos)} videos in {cls_name}")
    for vid_path in videos:
        cap = cv2.VideoCapture(vid_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            frames.append(transform(img))
        cap.release()
        
        if len(frames) < NUM_FRAMES:
            if len(frames) == 0:
                continue
            while len(frames) < NUM_FRAMES:
                frames.append(frames[-1])
        
        # Extract exactly 5 evenly spaced clips per video to keep training fast
        clips = []
        max_clips = 5
        if len(frames) >= NUM_FRAMES:
            starts = np.linspace(0, len(frames) - NUM_FRAMES, max_clips, dtype=int)
            for start in starts:
                clips.append(frames[start:start+NUM_FRAMES])
        else:
            indices = np.linspace(0, len(frames) - 1, NUM_FRAMES, dtype=int)
            clips.append([frames[i] for i in indices])
            
        for clip in clips:
            batch_t = torch.stack(clip).to(device)
            with torch.no_grad():
                features = resnet(batch_t) # (10, 512, 1, 1)
                features = features.view(NUM_FRAMES, 512) # (10, 512)
            all_features.append(features.cpu().numpy())
            all_labels.append(cls_idx)
        safe_name = os.path.basename(vid_path).encode("ascii", "ignore").decode()
        print(f"  Processed {safe_name}")

if len(all_features) == 0:
    print("Error: No videos found or processed!")
    exit(1)

X = np.array(all_features) # (N, 10, 512)
Y = np.array(all_labels)

print(f"Total extracted clips: {len(X)}")

class_weights = compute_class_weight('balanced', classes=np.unique(Y), y=Y)
class_weights = torch.FloatTensor(class_weights).to(device)

class MemoryDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, idx):
        x = self.X[idx].copy() # (10, 512)
        
        # Simple data augmentation (noise)
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.02, x.shape)
            x = x + noise
            
        x = x.T # Transpose to (512, 10) for TCN
        y = self.Y[idx]
        return torch.FloatTensor(x), torch.LongTensor([y])[0]

dataset = MemoryDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 4. Train TCN
print("Building TCN Model...")
INPUT_SIZE = 512
HIDDEN_SIZE = 256
LEVEL_SIZE = 10
NUM_CHANNELS = [HIDDEN_SIZE] * (LEVEL_SIZE - 1) + [INPUT_SIZE]
K_SIZE = 2
DROPOUT = 0.3
FC_SIZE = 256
model = TCN(INPUT_SIZE, 3, NUM_CHANNELS, K_SIZE, DROPOUT, FC_SIZE)
model = model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(weight=class_weights)

print("Starting End-to-End Training...")
best_acc = 0.0
for epoch in range(EPOCHS):
    epoch_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        
    acc = 100 * correct / total
    
    if acc > best_acc:
        best_acc = acc
        os.makedirs("model_zoo/your_model_zoo", exist_ok=True)
        torch.save(model.state_dict(), "model_zoo/your_model_zoo/tcn.pkl")
        
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss/len(dataloader):.4f} Acc: {acc:.2f}%")

print(f"Successfully trained and saved model with Best Accuracy: {best_acc:.2f}%!")
