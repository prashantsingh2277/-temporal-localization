import torch
from torch.utils.data import DataLoader, Dataset
from model import TemporalLocalizationModel
import torch.nn.functional as F
import os
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.videos = [f for f in os.listdir(dataset_dir) if f.endswith('.npy')]
        print(f"Found {len(self.videos)} video file(s) in {dataset_dir}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = os.path.join(self.dataset_dir, self.videos[idx])
        video_array = np.load(video_path)  
        video_tensor = torch.from_numpy(video_array).float()  
        if self.transform:
            video_tensor = self.transform(video_tensor)
        T = video_tensor.shape[0]
        label = {
            'boundary': torch.randn(2, T) * 2 + 3,
            'segment':  torch.randn(2, T) * 2 + 3
        }
        return video_tensor, label

def custom_collate_fn(batch):

    videos, labels = zip(*batch)  

    max_length = max(video.shape[0] for video in videos)

    padded_videos = []
    for video in videos:
        t, fdim = video.shape
        if t < max_length:
            pad_len = max_length - t
            pad_tensor = torch.zeros(pad_len, fdim)
            padded_video = torch.cat([video, pad_tensor], dim=0)
        else:
            padded_video = video
        padded_videos.append(padded_video)

    video_batch = torch.stack(padded_videos, dim=0).transpose(1, 2)

    padded_boundaries = []
    padded_segments = []
    for label in labels:
        boundary = label['boundary'] 
        segment = label['segment']    
        T_label = boundary.shape[1]
        if T_label < max_length:
            pad_len = max_length - T_label
            pad_boundary = torch.zeros(boundary.shape[0], pad_len)
            pad_segment = torch.zeros(segment.shape[0], pad_len)
            boundary_padded = torch.cat([boundary, pad_boundary], dim=1)
            segment_padded = torch.cat([segment, pad_segment], dim=1)
        else:
            boundary_padded = boundary
            segment_padded = segment
        padded_boundaries.append(boundary_padded)
        padded_segments.append(segment_padded)

    label_batch = {
        'boundary': torch.stack(padded_boundaries, dim=0), 
        'segment':  torch.stack(padded_segments, dim=0)       
    }
    return video_batch, label_batch

dataset_path = './small_dataset'
if not os.path.exists(dataset_path):
    raise ValueError(f"Dataset directory '{dataset_path}' does not exist.")

dataset = VideoDataset(dataset_path)
if len(dataset) == 0:
    raise ValueError(f"No .npy files found in '{dataset_path}'.")

loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)

model = TemporalLocalizationModel(in_channels=2048)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.SmoothL1Loss()

num_epochs = 25
for epoch in range(num_epochs):
    for videos, labels in loader:
        optimizer.zero_grad()
        boundary_pred, segment_pred = model(videos)
        loss_boundary = criterion(boundary_pred, labels['boundary'])
        loss_segment = criterion(segment_pred, labels['segment'])
        loss = loss_boundary + loss_segment
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}: Loss = {loss.item():.4f}")

torch.save(model.state_dict(), 'model_checkpoint.pth')
print("Saved model_checkpoint.pth")
