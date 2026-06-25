import os
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Configuration
DATASET_DIR = "Dataset"
NUM_FRAMES = 10
FRAME_SKIP = 5
MAX_CLIPS = 5
OUTPUT_DIR = "data/i3d_feature/train"

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load ResNet
print("Loading ResNet-18 Feature Extractor...")
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1]) # Remove FC layer
resnet = resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classes = {"armflapping": "arm", "headbanging": "head", "spinning": "spin"}

for cls_folder, prefix in classes.items():
    folder_path = os.path.join(DATASET_DIR, cls_folder)
    if not os.path.exists(folder_path):
        continue
        
    videos = []
    videos.extend(glob.glob(os.path.join(folder_path, "*.mp4")))
    videos.extend(glob.glob(os.path.join(folder_path, "*.webm")))
    
    print(f"Processing {len(videos)} videos in {cls_folder}...")
    
    for vid_path in videos:
        safe_name = os.path.basename(vid_path).encode("ascii", "ignore").decode()
        safe_name = safe_name.replace(".mp4", "").replace(".webm", "").replace(" ", "_")
        out_file_path = os.path.join(OUTPUT_DIR, f"{prefix}_{safe_name}.npy")
        
        if os.path.exists(out_file_path):
            print(f"  Skipping {safe_name} (already processed)")
            continue
            
        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue
            
        clips_features = []
        required_span = NUM_FRAMES * FRAME_SKIP
        
        if total_frames < required_span:
            # Fallback for very short videos
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frames = []
            fc = 0
            while len(frames) < NUM_FRAMES:
                ret, frame = cap.read()
                if not ret: break
                if fc % FRAME_SKIP == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(transform(Image.fromarray(frame)))
                fc += 1
            
            if len(frames) > 0:
                while len(frames) < NUM_FRAMES:
                    frames.append(frames[-1])
                
                batch_t = torch.stack(frames).to(device)
                with torch.no_grad():
                    features = resnet(batch_t).view(NUM_FRAMES, 512)
                clips_features.append(features.cpu().numpy())
        else:
            # FAST EXTRACTION: Jump instantly to exactly 5 evenly-spaced segments
            starts = np.linspace(0, total_frames - required_span, min(MAX_CLIPS, total_frames - required_span + 1), dtype=int)
            for start in starts:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start) # Instant seek!
                frames = []
                fc = 0
                while len(frames) < NUM_FRAMES:
                    ret, frame = cap.read()
                    if not ret: break
                    if fc % FRAME_SKIP == 0:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(transform(Image.fromarray(frame)))
                    fc += 1
                
                if len(frames) == NUM_FRAMES:
                    batch_t = torch.stack(frames).to(device)
                    with torch.no_grad():
                        features = resnet(batch_t).view(NUM_FRAMES, 512)
                    clips_features.append(features.cpu().numpy())
        
        cap.release()
        
        if len(clips_features) > 0:
            np.save(out_file_path, np.array(clips_features))
            print(f"  Processed and saved {safe_name}")
        else:
            print(f"  Failed to process {safe_name}")

print("Extraction complete! You can now run 'python train_with_features_tcn_improved.py' to train the model.")
