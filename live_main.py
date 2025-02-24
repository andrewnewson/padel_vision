import cv2
import os
import time
from utils import *
from trackers import *
from court_detector import *
import matplotlib.pyplot as plt


def main():
    # Read video (simulating a live video feed)
    input_video_path = "./input_media/padel_point.mp4"
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Instantiate models
    player_tracker = PlayerTracker(model_path="./models/yolo11n.pt")
    ball_tracker = BallTracker(model_path="./models/yolov5n6u_ball.pt")
    court_detector = CourtDetector(is_manual=True)

    # Get the first frame to detect court keypoints
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    court_keypoints = court_detector.create_keypoints(first_frame, save_path="./tracker_stubs/court_keypoints.json")
    print("Manual Court Keypoints:", court_keypoints)

    # Prepare for real-time processing
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Break if video ends

        start_time = time.time()  # Measure per-frame processing time

        # Track players and ball
        player_detections = player_tracker.detect_frame(frame)
        ball_detections = ball_tracker.detect_frame(frame)
        if ball_detections:
            ball_detections = ball_tracker.interpolate_ball_position([ball_detections])[0]
        else:
            ball_detections = []

        # Draw detections
        frame = player_tracker.draw_bounding_boxes([frame], [player_detections])[0]
        if len(ball_detections) > 0:
            frame = ball_tracker.draw_bounding_boxes([frame], [ball_detections])[0]
        frame = court_detector.draw_keypoints(frame, court_keypoints)

        # Add frame number
        cv2.putText(frame, f"Frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Show frame (simulating live video)
        # cv2.imshow("Simulated Live Tracking", frame)

        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.show()

        # Simulate real-time speed (assuming 30 FPS video)
        elapsed_time = time.time() - start_time
        wait_time = max(1, int(33 - (elapsed_time * 1000)))  # 33ms for ~30 FPS
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break  # Exit if 'q' is pressed

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Processing complete.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")