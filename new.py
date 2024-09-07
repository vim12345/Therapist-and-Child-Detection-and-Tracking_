import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import time

# Load YOLOv8 model (for person detection)
model = YOLO("yolov8n.pt")  # You can use 'yolov8s.pt' for higher accuracy

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0, max_cosine_distance=0.2, nn_budget=100)

# Load input video
input_video_path = 'test_video.mp4'
video_capture = cv2.VideoCapture(input_video_path)

# Define output video parameters
output_video_path = 'output_with_tracking.mp4'
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_fps = video_capture.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (frame_width, frame_height))

# Thresholds for differentiating between child and therapist
child_threshold_height = 120  # Modify as needed for child height estimation

start_time = time.time()

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        print("End of video reached or video cannot be loaded.")
        break

    # Run YOLOv8 detection
    results = model(frame)

    # Extract bounding boxes and class (person detection)
    detections = []
    for result in results:
        if len(result.boxes) > 0:
            for box in result.boxes:
                coords = box.xyxy.tolist()[0] 
                x1, y1, x2, y2 = map(int, coords)   # Bounding box coordinates
                conf = box.conf.item()  # Confidence score
                cls = int(box.cls.item())  # Class ID

                if cls == 0:  # YOLO class 0 is 'person'
                    detections.append(([x1, y1, x2, y2], conf))

    # Update tracker with the detections
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    # Loop through tracked objects and draw bounding boxes
    for track in tracked_objects:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()  # left, top, right, bottom
        x1, y1, x2, y2 = map(int, ltrb)

        # Calculate bounding box height to differentiate between child and therapist
        box_height = y2 - y1
        color = (0, 255, 0) if box_height > child_threshold_height else (255, 0, 0)  # Green for Therapist, Blue for Child
        label = "Therapist" if color == (0, 255, 0) else "Child"
        label_with_id = f"{label} ID: {track_id}"

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label_with_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Calculate and display FPS
    fps = int(1 / (time.time() - start_time))
    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    start_time = time.time()

    # Write the frame with detections to the output video
    video_writer.write(frame)

    # Optional: Display the video for real-time debugging
    # Commented out due to OpenCV display issues
    # cv2.imshow("Child and Therapist Tracking", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     print("Exiting video playback.")
    #     break

# Release resources after video processing is completed
video_capture.release()
video_writer.release()
cv2.destroyAllWindows()
print("Processing completed, resources released.")
