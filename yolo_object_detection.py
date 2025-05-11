import cv2
import numpy as np

# Load YOLO model (Make sure you have the weights and config files)
YOLO_WEIGHTS = "yolov3.weights"
YOLO_CONFIG = "yolov3.cfg"
YOLO_NAMES = "coco.names"

# Load the class labels
with open(YOLO_NAMES, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load the YOLO network
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Function to detect objects
def detect_objects(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Convert image to blob for YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    detected_objects = set()  # Store unique detected objects

    # Process detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Confidence threshold
                detected_objects.add(class_names[class_id])  # Add object to set

    print("Detected Objects:", detected_objects)
    return list(detected_objects)  # Return as a list
