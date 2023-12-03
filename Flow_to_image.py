import os
import cv2 as cv
import numpy as np

def calculate_optical_flow(video_path, output_folder):
    cap = cv.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the video")
        cap.release()
        return

    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    frame_count = 0
    flow_accumulator = np.zeros((first_frame.shape[0], first_frame.shape[1], 2), dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if frame_count >= 1:
            # Calculate optical flow between consecutive frames
            flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_accumulator += flow

            # Save the optical flow frame
            flow_frame = cv.cvtColor(flow_to_color(flow_accumulator), cv.COLOR_BGR2GRAY)

            # Extract the video file name (excluding extension)
            file_name = os.path.splitext(os.path.basename(video_path))[0]

            # Create a folder with the video file name if it doesn't exist
            output_folder_path = os.path.join(output_folder, file_name)
            os.makedirs(output_folder_path, exist_ok=True)

            # Save the optical flow frame in the folder
            output_path = os.path.join(output_folder_path, f'flow_frame_{frame_count:04d}.png')
            cv.imwrite(output_path, flow_frame)

            flow_accumulator = np.zeros_like(flow_accumulator)

        prev_gray = gray
        frame_count += 1

    cap.release()

def flow_to_color(flow):
    # Convert optical flow to color representation for visualization
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hue = (angle * 180 / np.pi / 2).astype(np.uint8)
    saturation = (magnitude * 255 / magnitude.max()).astype(np.uint8)
    value = np.ones_like(hue) * 255
    flow_rgb = cv.merge([hue, saturation, value])
    return cv.cvtColor(flow_rgb, cv.COLOR_HSV2BGR)

def preprocess_all_videos(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        if file_name.endswith('_processed.mp4'):
            input_video_path = os.path.join(input_folder, file_name)

            # Pass the output folder as the second argument
            calculate_optical_flow(input_video_path, output_folder)
