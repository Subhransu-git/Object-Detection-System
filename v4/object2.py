import cv2
import numpy as np
import threading
import time

# -----------------------------
# OpenCV optimization
# -----------------------------
cv2.setUseOptimized(True)
cv2.setNumThreads(0)

# -----------------------------
# Load YOLOv4
# -----------------------------
print("Loading YOLOv4...")

net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [
    layer_names[i - 1]
    for i in net.getUnconnectedOutLayers().flatten()
]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

print("YOLOv4 loaded.")


# -----------------------------
# Camera
# -----------------------------
camera_url = "http://10.34.162.154:8080/video"

cap = cv2.VideoCapture(camera_url)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()


# -----------------------------
# Latest-frame camera reader
# -----------------------------
latest_frame = None
frame_lock = threading.Lock()
running = True


def camera_reader():
    global latest_frame, running

    while running:
        ret, frame = cap.read()

        if not ret:
            print("Camera frame could not be read.")
            time.sleep(0.01)
            continue

        # Replace old frame with newest frame
        with frame_lock:
            latest_frame = frame


# Start camera reader thread
camera_thread = threading.Thread(
    target=camera_reader,
    daemon=True
)

camera_thread.start()


# -----------------------------
# Wait for first frame
# -----------------------------
print("Waiting for camera...")

while latest_frame is None:
    time.sleep(0.01)

print("Camera connected.")


# -----------------------------
# YOLO settings
# -----------------------------
frame_skip = 2
frame_id = 0

fps = 0
prev_time = time.time()

yolo_time = 0


# -----------------------------
# Main loop
# -----------------------------
while True:

    # Get newest frame
    with frame_lock:
        if latest_frame is None:
            continue

        frame = latest_frame.copy()

    frame_id += 1

    # Skip some frames to reduce YOLO workload
    if frame_id % frame_skip != 0:
        continue

    # -------------------------
    # YOLO preprocessing
    # -------------------------
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1 / 255,
        size=(320, 320),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)

    # -------------------------
    # YOLO inference
    # -------------------------
    start_yolo = time.time()

    outputs = net.forward(output_layers)

    yolo_time = time.time() - start_yolo

    # -------------------------
    # Detection
    # -------------------------
    boxes = []
    confidences = []
    class_ids = []

    height, width = frame.shape[:2]

    for output in outputs:

        for detection in output:

            scores = detection[5:]

            class_id = np.argmax(scores)

            confidence = scores[class_id]

            if confidence > 0.5:

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = center_x - w // 2
                y = center_y - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # -------------------------
    # Non-Maximum Suppression
    # -------------------------
    indexes = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        0.5,
        0.4
    )

    # -------------------------
    # Draw detections
    # -------------------------
    if len(indexes) > 0:

        for i in indexes.flatten():

            x, y, w, h = boxes[i]

            label = (
                f"{classes[class_ids[i]]}: "
                f"{confidences[i]:.2f}"
            )

            color = colors[class_ids[i]]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    # -------------------------
    # Calculate FPS
    # -------------------------
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # -------------------------
    # Display information
    # -------------------------
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"YOLO: {yolo_time * 1000:.0f} ms",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "YOLOv4 Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------
running = False

camera_thread.join(timeout=1)

cap.release()

cv2.destroyAllWindows()

print("Program stopped.")