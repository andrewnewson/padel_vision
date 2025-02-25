import os
from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import sys
sys.path.append("../")
from utils import *

class StreamPlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)  # load the player tracking model

    def detect_frame(self, frame):
        # Perform player detection and tracking with stream=True and persist=True to keep object IDs
        results = self.model.track(frame, stream=True, persist=True)[0]  # track players with persistence and streaming
        id_name_dict = results.names  # get the class names from the model

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])  # track_id persists across frames
            result = box.xyxy.tolist()[0]  # get bounding box coordinates
            object_cls_id = box.cls.tolist()[0]  # get class id
            object_cls_name = id_name_dict[object_cls_id]  # get class name

            if object_cls_name == "person":  # track "person" only
                player_dict[track_id] = result  # store bounding box for the player

        return player_dict

    def detect_frames(self, video_path, read_from_stub=False, stub_path=None):
        player_detections = []

        # If read_from_stub is True, load the detections from the pickle file
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        # Otherwise, perform the detection from the video
        cap = cv2.VideoCapture(video_path)  # Open video file

        while cap.isOpened():
            ret, frame = cap.read()  # Read next frame
            if not ret:
                break

            # Detect players in the current frame with persistence and streaming
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        cap.release()

        # Save the detections to a pickle file
        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def identify_players(self, court_keypoints, player_dict, n_players=2):
        distances = []
        for track_id, bbox in player_dict.items():
            player_centre = get_bbox_centre(bbox)

            min_distance = float("inf")
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_abs_distance(player_centre, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        distances.sort(key=lambda x: x[1])
        chosen_players = [distances[i][0] for i in range(n_players)]

        return chosen_players

    def draw_bounding_boxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID {track_id}", (int(x1), int(y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            output_video_frames.append(frame)
        return output_video_frames


class StreamBallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)  # load ball tracking model

    def detect_frame(self, frame):
        # Perform ball detection and tracking with stream=True and persist=True to keep object ID
        results = self.model.track(frame, stream=True, persist=True)[0]  # track ball with persistence and streaming

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]  # get bounding box coordinates
            ball_dict[1] = result  # store ball bounding box (ball ID=1)

        return ball_dict

    def interpolate_ball_position(self, ball_positions):
        # Interpolate missing ball positions
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
        df_ball_positions = df_ball_positions.interpolate()  # interpolate missing values
        df_ball_positions = df_ball_positions.bfill()  # backfill missing ball positions
        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self, video_path, read_from_stub=False, stub_path=None):
        ball_detections = []

        # If read_from_stub is True, load the detections from the pickle file
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        # Otherwise, perform the detection from the video
        cap = cv2.VideoCapture(video_path)  # Open video file

        while cap.isOpened():
            ret, frame = cap.read()  # Read next frame
            if not ret:
                break

            # Detect ball in the current frame with persistence and streaming
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        cap.release()

        # Save the detections to a pickle file
        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def draw_bounding_boxes(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID {track_id}", (int(x1), int(y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames