from utils import *
from trackers import *
from court_detector import *
import os
import time
import cv2
import argparse
import yt_dlp

def main(input_video):
    if "youtube.com" in input_video:
        # Extract direct stream URL from youtube
        ydl_opts = {"format": "best[ext=mp4]"}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(input_video, download=False)
            input_video_path = info_dict["url"]  # Direct stream URL
        name = info_dict["id"]
    else:
        # Read video
        input_video_path = input_video
        _, file_name = os.path.split(input_video_path)
        name, _ = os.path.splitext(file_name)

    # Open video file with OpenCV
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {input_video_path}")
        return
    
    # Initialize trackers
    player_tracker = PlayerTracker(model_path="./models/yolov5nu.pt")
    ball_tracker = BallTracker(model_path="./models/yolov5n6u_ball.pt")
    # court_detector = CourtDetector(is_manual=True)

    # Initialise complete tracker lists
    player_detections = []
    ball_detections = []

    # Process video frame by frame
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect players and ball
        frame_player_detections = player_tracker.detect_frames([frame])
        if len(frame_player_detections) > 0:
            player_detections.extend(frame_player_detections)
        frame_ball_detections = ball_tracker.detect_frames([frame])
        if len(frame_ball_detections) > 0:
            ball_detections.extend(frame_ball_detections)
        
        # Detect court lines
        # if frame_count == 0:  # Use the first frame to detect court keypoints
            # court_keypoints = court_detector.create_keypoints(frame, save_path="./tracker_stubs/court_keypoints.json")
        
        frame_count += 1
    
    # Release resources
    cap.release()

    # Interpolate ball detections
    ball_detections = ball_tracker.interpolate_ball_position(ball_detections)

    # Save detections
    with open(f"./output_media/{name}_player_detections.json", "w") as f:
        json.dump(player_detections, f)

    with open(f"./output_media/{name}_ball_detections.json", "w") as f:
        json.dump(ball_detections, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video file for player and ball tracking.")
    parser.add_argument("input_video", type=str, help="Path to the input video file")
    args = parser.parse_args()

    start_time = time.time()
    main(args.input_video)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")