import cv2 as cv
import os
import re
import numpy as np

def calculate_optical_flow(video_path, output_folder, max_frames=64):
    cap = cv.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the video")
        cap.release()
        return

    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    # Create a folder to save the optical flow frames as images
    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0

    while frame_count < max_frames:  # Process up to max_frames frames
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        hue = (angle * 180 / np.pi / 2).astype(np.uint8)
        saturation = (magnitude * 255 / magnitude.max()).astype(np.uint8)
        value = np.ones_like(hue) * 255

        flow_rgb = cv.merge([hue, saturation, value])
        flow_rgb = cv.cvtColor(flow_rgb, cv.COLOR_HSV2BGR)

        # Save each frame as an individual RGB image
        output_image_path = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv.imwrite(output_image_path, flow_rgb)

        prev_gray = gray
        frame_count += 1

    cap.release()

def preprocess_all_videos(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        if file_name.endswith('_processed.mp4'):
            input_video_path = os.path.join(input_folder, file_name)

            # Create a subfolder for each video
            video_name = os.path.splitext(file_name)[0]
            video_output_folder = os.path.join(output_folder, video_name)

            # Calculate optical flow and save frames as images
            calculate_optical_flow(input_video_path, video_output_folder)

# Folder paths
input_video_folder = r"C:\Users\andre\OneDrive\Documents\GitHub\Brainvision_Project\Motion_files\videos_processed"
output_image_folder = r"C:\Users\andre\OneDrive\Documents\GitHub\Brainvision_Project\Motion_files\Optical_flow\frames"

# Preprocess all videos in the folder
preprocess_all_videos(input_video_folder, output_image_folder)
