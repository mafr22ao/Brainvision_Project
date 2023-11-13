# @title Visualize video (adapted to PyCharm)
import cv2
import glob

video_dir = './AlgonautsVideos268_All_30fpsmax'
video_files = sorted(glob.glob(video_dir + '/*.mp4'))
vid_id = 0  # @param {type: "integer"}

if video_files:
    # Create a video capture object
    cap = cv2.VideoCapture(video_files[vid_id])

    # Check if the video capture object was successfully created
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        paused = False
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    # If no frames are left, break the loop
                    break
                cv2.imshow('Video', frame)

            key = cv2.waitKey(30)

            if key == ord('q'):
                # Quit the video player
                break
            elif key == ord('p'):
                # Toggle pause
                paused = not paused
            elif key == ord('r'):
                # Restart the video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                paused = False

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
else:
    print("No video files found in the specified directory.")