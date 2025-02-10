import cv2
import matplotlib.pyplot as plt
from ultralytics import NAS, YOLO

# model = NAS('models/yolo_nas_s.pt')
# model = YOLO('models/yolov5nu.pt')

model = YOLO("C:\\Users\\AndrewNewson\\Downloads\\best (4).pt")

for result in model.track(source='input_media/padel_point.mp4', stream=True):
    frame = result.plot()
    
    if frame is not None:
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show(block=False)
        plt.pause(0.001)
        plt.clf()