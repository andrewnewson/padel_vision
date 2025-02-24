from utils import *
from trackers import *
from court_detector import *
import os
import time
import cv2

def main():
    # Read video
    input_video_path = "./input_media/padel_ten.mp4"
    video_frames = read_video(input_video_path)

    # Detect players and ball
    player_tracker = PlayerTracker(model_path="./models/yolo11n.pt")
    ball_tracker = BallTracker(model_path="./models/yolov5n6u_ball.pt")

    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="./tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="./tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_position(ball_detections)

    # # Detect court lines (choice of manual or auto detection) (pass first frame of video)
    court_detector = CourtDetector(is_manual=True)
    court_keypoints = court_detector.create_keypoints(video_frames[0], save_path="./tracker_stubs/court_keypoints.json")
    
    # Draw bounding boxes around players and ball
    output_video_frames = player_tracker.draw_bounding_boxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bounding_boxes(output_video_frames, ball_detections)

    # Draw court lines
    output_video_frames = court_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Add frame number to video
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Save video with player detections and bounding boxes overlay
    _, file_name = os.path.split(input_video_path)
    name, _ = os.path.splitext(file_name)
    output_video_path = f"./output_media/{name}_output.avi"
    save_video(output_video_frames, output_video_path)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")