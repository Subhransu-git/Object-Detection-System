from ultralytics import YOLO
import cv2

# Load the YOLO11 Nano model
model = YOLO("yolo11n.pt")

# Phone's current IP in which the video is streaming
url = "http://10.109.93.26:8080/video"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Could not connect to IP Webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    # Perform object detection
    results = model(frame, verbose=False)

    # Draw detections
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO11 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()