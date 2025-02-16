import kagglehub
import os
import random
import shutil


dataset_path = kagglehub.dataset_download("valuejack/activitynet1-3")
print("Path to dataset files:", dataset_path)

def create_small_dataset(source_dir, dest_dir, num_videos=100):
    os.makedirs(dest_dir, exist_ok=True)
    videos = [f for f in os.listdir(source_dir) if f.endswith('.mp4')]
    selected = random.sample(videos, min(num_videos, len(videos)))
    for video in selected:
        shutil.copy(os.path.join(source_dir, video), os.path.join(dest_dir, video))

create_small_dataset(os.path.join(dataset_path, 'train'), './small_dataset')
