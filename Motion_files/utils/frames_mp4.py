import cv2 as cv
import os

def create_video_from_frames(frame_folder, output_video_path, frame_rate=12):
    frame_files = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.png')])
    if not frame_files:
        print(f"No frame images found in {frame_folder}.")
        return

    frame = cv.imread(frame_files[0])
    height, width, layers = frame.shape
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_video_path, fourcc, frame_rate, (width, height), isColor=True)

    for frame_file in frame_files:
        frame = cv.imread(frame_file)
        out.write(frame)

    out.release()

def create_videos_from_subfolders(input_folder, output_folder, frame_rate=12):
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            output_video_name = f"{subfolder}.mp4"
            output_video_path = os.path.join(output_folder, output_video_name)
            create_video_from_frames(subfolder_path, output_video_path, frame_rate)


# Folder paths
input_folder = r"C:\Users\andre\OneDrive\Documents\GitHub\Brainvision_Project\Motion_files\Optical_flow\frames"
output_video_folder = r"C:\Users\andre\OneDrive\Documents\GitHub\Brainvision_Project\Motion_files\Optical_flow\Video"

# Create videos from subfolders containing frame images
create_videos_from_subfolders(input_folder, output_video_folder)

