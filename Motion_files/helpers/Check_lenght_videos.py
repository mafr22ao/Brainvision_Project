import os
from moviepy.editor import VideoFileClip

def find_short_videos(directory_path, duration_threshold=3.0):
    # List all files in the specified directory
    files = os.listdir(directory_path)

    short_videos = []

    for file in files:
        if file.endswith(".mp4") or file.endswith(".avi"):
            video_path = os.path.join(directory_path, file)
            try:
                clip = VideoFileClip(video_path)
                duration = clip.duration
                clip.close()

                # Check if the duration is less than the specified threshold
                if duration < duration_threshold:
                    short_videos.append(file)
            except Exception as e:
                print(f"Error processing '{file}': {str(e)}")

    return short_videos

# Example usage:
directory_path = r"C:\Users\andre\OneDrive\Documents\GitHub\Brainvision_Project\Motion_files\videos_adjusted"
duration_threshold = 3.0
short_videos = find_short_videos(directory_path, duration_threshold)

if short_videos:
    print("Short videos (less than 3 seconds):")
    for video in short_videos:
        print(video)
else:
    print("No short videos found.")

