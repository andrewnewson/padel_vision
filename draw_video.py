import os
import cv2
import json
import argparse
import time

def main(video_path, player_detections, ball_detections):
    # Load YOLO detections (list where each item is detections for a frame)
    with open(player_detections, "r") as file:
        player_detections = json.load(file)

    with open(ball_detections, "r") as file:
        ball_detections = json.load(file)

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define codec and create VideoWriter object for AVI output
    _, file_name = os.path.split(video_path)
    name, _ = os.path.splitext(file_name)
    output_path = f"/content/drive/MyDrive/Colab Notebooks/padel_vision/{name}_output.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Use XVID for AVI format
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Function to draw bounding boxes
    def draw_bounding_boxes(frame, detections, color):
        for obj_id, coords in detections.items():
            x1, y1, x2, y2 = coords  # [x1, y1, x2, y2]
            thickness = 2
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            cv2.putText(frame, f"ID: {obj_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    # Process frame by frame
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Get detections for this frame (assuming yolo_detections matches video length)
        p_detections = player_detections[frame_idx] if frame_idx < len(player_detections) else []
        b_detections = ball_detections[frame_idx] if frame_idx < len(ball_detections) else []

        # Draw bounding boxes
        processed_frame = draw_bounding_boxes(frame, p_detections, (0, 255, 0))
        processed_frame = draw_bounding_boxes(processed_frame, b_detections, (0, 0, 255))

        # Write processed frame to output video
        out.write(processed_frame)

        frame_idx += 1

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Processing complete. Output saved to:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add bbox to a video file for player and ball tracking.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("player_detections", type=str, help="Path to the player detections JSON file")
    parser.add_argument("ball_detections", type=str, help="Path to the ball detections JSON file")

    args = parser.parse_args()

    start_time = time.time()
    main(args.video_path, args.player_detections, args.ball_detections)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")