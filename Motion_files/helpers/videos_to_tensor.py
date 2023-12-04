import h5py
import torch
import os
import numpy as np
from decord import VideoReader, cpu
from torchvision.transforms import Compose, Lambda
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample

def preprocess_algonauts_video(video_path):
    max_frames = 64

    # Transformation pipeline for optical flow video with 224 channels
    transform = ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(max_frames),
            Lambda(lambda x: x.clone().detach().to(dtype=torch.float16)),  # Updated tensor copy method
        ]),
    )

    video = {'video': read_optical_flow_video(video_path)}
    video_data = transform(video)
    inputs = video_data["video"]

    return inputs

def read_optical_flow_video(file):
    vr = VideoReader(file, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, total_frames, dtype=np.int16)

    images = []
    for seg_ind in indices:
        frame = vr[seg_ind].asnumpy()
        images.append(frame)

    video = np.array(images)
    return torch.from_numpy(video).permute(0, 3, 1, 2)


# Step 2: Specify input and output folders
input_folder = r"C:\Users\andre\OneDrive\Documents\GitHub\Brainvision_Project\Motion_files\Optical_flow"
output_folder = r"C:\Users\andre\OneDrive\Documents\GitHub\Brainvision_Project\Motion_files\Tensors_opticalflow"  # Specify your desired output folder

# Step 3: Get a list of video files in the input folder
video_list = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith(".mp4")]

# Loop through each video file
for video_path in video_list:
    # Load and preprocess the video
    video_tensor = preprocess_algonauts_video(video_path)

    # Extract the file name without extension
    file_name = os.path.splitext(os.path.basename(video_path))[0]

    # Save using HDF5
    output_file = os.path.join(output_folder, f"{file_name}_tensor.h5")
    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset("data", data=video_tensor.numpy(), compression='gzip')

    print(f"Saved compressed tensor for {file_name} to {output_file}")