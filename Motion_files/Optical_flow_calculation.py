import cv2 as cv
import os
import re
import numpy as np


def calculate_optical_flow(video_path, output_folder, skip_frames=1, frame_window=72):
    cap = cv.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the video")
        cap.release()
        return

    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format

    # Extract file identifier from the input video filename
    file_name = os.path.basename(video_path)
    identifier = re.match(r'(\d{4})_processed\.mp4', file_name)
    file_id = identifier.group(1) if identifier else "Unknown"

    # Create the output video filename
    output_video_name = f'{file_id}_flow.mp4'
    output_video_path = os.path.join(output_folder, output_video_name)

    # Adjust frame rate to 12 fps
    out = cv.VideoWriter(output_video_path, fourcc, 12, (first_frame.shape[1], first_frame.shape[0]), isColor=True)

    frame_count = 0
    flow_accumulator = np.zeros((first_frame.shape[0], first_frame.shape[1], 2), dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if frame_count % skip_frames == 0:
            if frame_count >= frame_window * skip_frames:
                magnitude, angle = cv.cartToPolar(flow_accumulator[..., 0], flow_accumulator[..., 1])

                hue = (angle * 180 / np.pi / 2).astype(np.uint8)
                saturation = (magnitude * 255 / magnitude.max()).astype(np.uint8)
                value = np.ones_like(hue) * 255

                flow_rgb = cv.merge([hue, saturation, value])
                flow_rgb = cv.cvtColor(flow_rgb, cv.COLOR_HSV2BGR)

                out.write(flow_rgb)

                flow_accumulator = np.zeros_like(flow_accumulator)

            flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_accumulator += flow

        prev_gray = gray
        frame_count += 1

    cap.release()
    out.release()


def preprocess_all_videos(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        if file_name.endswith('_processed.mp4'):
            input_video_path = os.path.join(input_folder, file_name)

            # Pass skip_frames and frame_window arguments
            calculate_optical_flow(input_video_path, output_folder, skip_frames=None, frame_window=72)


# Folder paths
input_video_folder = r"C:\Users\andre\OneDrive\Documents\GitHub\Brainvision_Project\Motion_files\videos_processed"
output_video_folder = r"C:\Users\andre\OneDrive\Documents\GitHub\Brainvision_Project\Motion_files\Optica_flow"

# Preprocess all videos in the folder
preprocess_all_videos(input_video_folder, output_video_folder)

