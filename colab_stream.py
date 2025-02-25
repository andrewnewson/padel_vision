from utils import *
from trackers import *
from court_detector import *
import os
import time
import cv2

def main():
    # Read video
    input_video_path = "./input_media/padel_ten.mp4"

    # Open video file with OpenCV
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {input_video_path}")
        return
    
    # Initialize trackers
    player_tracker = PlayerTracker(model_path="./models/yolo11n.pt")
    ball_tracker = BallTracker(model_path="./models/yolov5n6u_ball.pt")
    court_detector = CourtDetector(is_manual=True)

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
        frame_player_detections = player_tracker.detect_frames([frame], read_from_stub=False, stub_path="./tracker_stubs/player_detections.pkl")
        player_detections.extend(frame_player_detections)
        frame_ball_detections = ball_tracker.detect_frames([frame], read_from_stub=False, stub_path="./tracker_stubs/ball_detections.pkl")
        ball_detections.extend(frame_ball_detections)
        
        # Detect court lines
        if frame_count == 0:  # Use the first frame to detect court keypoints
            court_keypoints = court_detector.create_keypoints(frame, save_path="./tracker_stubs/court_keypoints.json")
        
        frame_count += 1
    
    # Release resources
    cap.release()

    # Interpolate ball detections
    ball_detections = ball_tracker.interpolate_ball_position(ball_detections)

    # Save detections
    _, file_name = os.path.split(input_video_path)
    name, _ = os.path.splitext(file_name)
    with open(f"./output_media/{name}_player_detections.json", "w") as f:
        json.dump(player_detections, f)

    with open(f"./output_media/{name}_ball_detections.json", "w") as f:
        json.dump(ball_detections, f)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")