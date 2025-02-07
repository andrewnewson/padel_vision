import cv2
import json
import torchvision.models as models

class CourtDetector():
    def __init__(self, is_manual=True, model_path=None):
        """
        Initializes the class to either use manual keypoints selection or auto-detect using a model.
        
        Args:
            is_manual (bool): Flag indicating whether to use manual keypoints selection.
            model_path (str, optional): Path to the model for automatic keypoint detection. Required if is_manual=False.
        """
        self.is_manual = is_manual  # Whether to manually select keypoints or not

        if not is_manual:
            if model_path is None:
                raise ValueError("model_path must be provided when is_manual is False.")
            self.model = models.resnet50(weights=None)  # load resnet50 model for placeholder

    def click_event(self, event, x, y, flags, param):
        """Handles mouse click events to store selected keypoints."""
        if event == cv2.EVENT_LBUTTONDOWN:
            param[0].append((x, y))  # Append the keypoint to the list
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(param[1], f"{x},{y}", (x, y), font, 1, (255, 0, 0), 2)
            cv2.imshow("Select Court Keypoints", param[1])

    def create_keypoints(self, frame, save_path=None):
        """
        Allows a user to manually annotate keypoints on a given frame or automatically detect keypoints.
        
        Args:
            frame (numpy.ndarray): The first frame of the video where keypoints will be selected or detected.
            save_path (str, optional): Path to save the selected keypoints as a JSON file.

        Returns:
            list: A list of (x, y) tuples representing the selected or detected keypoints.
        """
        if self.is_manual:
            selected_keypoints = []  # Create the list to store selected keypoints
            
            # Manual keypoint selection
            cv2.imshow("Select Court Keypoints", frame)
            cv2.setMouseCallback("Select Court Keypoints", self.click_event, [selected_keypoints, frame])

            cv2.waitKey(0)  # Wait for user to select keypoints
            cv2.destroyAllWindows()

            # Save keypoints if a save path is provided
            if save_path:
                with open(save_path, "w") as f:
                    json.dump(selected_keypoints, f)

            return selected_keypoints
        else:
            # Automatic keypoint detection using the model
            return self.auto_detect_keypoints(frame)

    def auto_detect_keypoints(self, frame):
        print("Functionality not added yet.")

    def draw_keypoints(self, image, keypoints):
        """Draw selected keypoints on the image."""
        for i in range(0, len(keypoints), 2):  # assuming keypoints come in pairs (x, y)
            x = int(keypoints[i])  # x coordinate
            y = int(keypoints[i + 1])  # y coordinate
            cv2.putText(image, str(i // 2), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        """Draw keypoints on all video frames."""
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames