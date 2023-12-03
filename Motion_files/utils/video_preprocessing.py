import cv2 as cv
import os
import re

def preprocess_video(input_video_path, output_video_path):
    # Open the input video
    cap = cv.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # You can change 'mp4v' to another codec if needed
    fps = 22.0  # FPS of the output video

    # Read the first frame to get the aspect ratio
    ret, frame = cap.read()
    if not ret:
        print("Error reading video frame.")
        cap.release()
        return

    # Calculate the new dimensions
    height, width = frame.shape[:2]
    if height > width:
        new_height = int(224 * height / width)
        new_width = 224
    else:
        new_width = int(224 * width / height)
        new_height = 224

    # Calculate the target number of frames (72 frames for 3 seconds at 24fps)
    target_num_frames = int(66)

    # Calculate the frame count of the input video
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # Calculate the number of frames to extend or trim
    frames_to_extend = max(0, target_num_frames - total_frames)
    frames_to_trim = max(0, total_frames - target_num_frames)

    # Create VideoWriter object
    out = cv.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))

    # Process the video
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        resized_frame = cv.resize(frame, (new_width, new_height), interpolation=cv.INTER_AREA)

        # Write the frame into the file
        out.write(resized_frame)

        frame_count += 1

        # Check if we need to extend or trim frames
        if frame_count >= total_frames:
            break

    # Extend the video by adding extra frames
    if frames_to_extend > 0:
        for _ in range(frames_to_extend):
            out.write(resized_frame)

    # Trim the video by removing extra frames
    elif frames_to_trim > 0:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame
            resized_frame = cv.resize(frame, (new_width, new_height), interpolation=cv.INTER_AREA)

            # Write the frame into the file
            out.write(resized_frame)

            frame_count += 1

            if frame_count >= target_num_frames:
                break

    # Release everything
    cap.release()
    out.release()

def preprocess_all_videos(input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.mp4'):
            input_video_path = os.path.join(input_folder, file_name)

            # Extract file identifier
            identifier = re.match(r'(\d{4})_', file_name)
            file_id = identifier.group(1) if identifier else "Unknown"

            output_video_path = os.path.join(output_folder, f'{file_id}_processed.mp4')
            preprocess_video(input_video_path, output_video_path)

# Folder paths
input_video_folder = r"C:\Users\andre\OneDrive\Documents\GitHub\Brainvision_Project\Motion_files\AlgonautsVideos268_All_30fpsmax"
output_video_folder = r"C:\Users\andre\OneDrive\Documents\GitHub\Brainvision_Project\Motion_files\videos_processed"

# Preprocess all videos in the folder
preprocess_all_videos(input_video_folder, output_video_folder)
