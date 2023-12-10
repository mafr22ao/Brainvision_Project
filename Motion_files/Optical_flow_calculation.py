import cv2 as cv
import numpy as np
import os
import re

def generate_optical_flow(video_path, output_path, target_flow_count=16, frame_skip=4):
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if total_frames < target_flow_count * frame_skip:
        print(f"Not enough frames in video ({total_frames}) for {target_flow_count} flows with frame skip of {frame_skip}. Skipping.")
        cap.release()
        return

    flow_stack = []

    for frame_count in range(0, total_frames, frame_skip):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if frame_count > 0:
            prev_frame = gray
            flow = cv.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_stack.append(flow[..., 0])
            flow_stack.append(flow[..., 1])

    cap.release()

    if len(flow_stack) < target_flow_count * 2:
        print("Not enough optical flows calculated in video:", video_path)
        return

    stacked_flow = np.stack(flow_stack[:target_flow_count * 2], axis=-1)

    # Extracting the first four digits from the video file name
    video_name = os.path.basename(video_path)
    match = re.search(r'\d{4}', video_name)
    if match:
        name_prefix = match.group()
    else:
        print("No digits found in video name:", video_name)
        return

    # Save the stacked flow
    output_filename = name_prefix + '_stackedopticalflow.npy'
    np.save(os.path.join(output_path, output_filename), stacked_flow)


def process_videos_in_folder(folder_path, output_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):  # Assuming videos are in .mp4 format
            video_path = os.path.join(folder_path, filename)
            generate_optical_flow(video_path, output_path)
            print(f"Processed {filename}")

def process_videos_in_folder(folder_path, output_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):  # Assuming videos are in .mp4 format
            video_path = os.path.join(folder_path, filename)
            generate_optical_flow(video_path, output_path)
            print(f"Processed {filename}")


# Example usage
folder_path = r"C:\Users\andre\OneDrive\Documents\GitHub\Brainvision_Project\Motion_files\videos_processed"
output_path = r"C:\Users\andre\OneDrive\Documents\GitHub\Brainvision_Project\Motion_files\Optical_flow\stacked_img"
process_videos_in_folder(folder_path, output_path)



