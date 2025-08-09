import os
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
import time

from identify_group import cluster_bboxes_with_ids

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_video', type=str, required=True, help='Path to input video file')
parser.add_argument('--output_folder', type=str, required=True, help='Path to output folder')
parser.add_argument('--epsilon', type=float, default=50.0, help='DBSCAN epsilon value')
parser.add_argument('--threshold_overlap', type=float, default=0.7, help='Overlap threshold for old/new group determination')
args = parser.parse_args()

input_video = args.input_video
output_folder = args.output_folder
epsilon = args.epsilon
threshold_overlap = args.threshold_overlap
os.makedirs(output_folder, exist_ok=True)

# Load YOLO model (YOLOv8 by default, can change to yolov8n.pt if needed)
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(input_video)
frame_idx = 0
total_infer_time = 0
num_infer_frames = 0
all_detections = []

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

# Lấy thông tin video để tạo VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = os.path.join(output_folder, 'output_yolo_virat_bytetrack.mp4')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

groups_status = {}
max_group_id = -1
start_time = time.time()  # Thời gian bắt đầu

# Lặp qua từng frame của video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Dùng tracking của YOLO, chỉ lấy người (class 0)
    infer_start = time.time()
    results = model.track(
        frame,
        persist=True,
        classes=0,
        verbose=False,
        tracker="bytetrack.yaml"  # Sử dụng ByteTrack thay vì BoTSORT
    )[0]
    infer_time = time.time() - infer_start
    total_infer_time += infer_time
    num_infer_frames += 1
    # Lấy bbox và id nếu có
    if results.boxes.id is not None:
        ids = results.boxes.id.cpu().numpy().astype(int)
        boxes = results.boxes.xyxy.cpu().numpy()

        boxes_xywh = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)
            h = int(y2 - y1)
            boxes_xywh.append([x, y, w, h])

        cluster_results, groups_status, max_group_id = cluster_bboxes_with_ids(
            groups_status, boxes_xywh, ids.tolist(), max_group_id, eps=epsilon, min_samples=2, threshold_overlap=threshold_overlap)

        # Draw bbox and info for each person
        for person in cluster_results:
            id_p = person['id_p']
            color = person['color']
            id_g = person['id_g']
            bbox = person['bbox']
            center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)

            # Draw bbox
            cv2.rectangle(frame,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                        color, 1)

            # Draw center point
            center_x = int(center[0])
            center_y = int(center[1])
            cv2.circle(frame, (center_x, center_y), 3, color, -1)

            text = ""
            if id_g == -1:
                text = f"ID:{id_p} No group"
            else:
                text = f"ID:{id_p} G:{id_g}"

            cv2.putText(frame, text,
                        (center_x - 10, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, color, 1)
    print(f"Processed frame {frame_idx}", end='\r')
    out.write(frame)  # Ghi frame đã vẽ vào video output
    frame_idx += 1

end_time = time.time()  # Thời gian kết thúc

total_time = end_time - start_time
fps_process = frame_idx / total_time if total_time > 0 else 0
print(f"\nProcessing done.")
print(f"Total frames: {frame_idx}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Processing speed: {fps_process:.2f} FPS")

if num_infer_frames > 0:
    avg_infer_time = total_infer_time / num_infer_frames
    fps_infer = 1.0 / avg_infer_time
    print(f"YOLO inference time per frame: {avg_infer_time:.4f} seconds")
    print(f"YOLO inference FPS: {fps_infer:.2f}")

cap.release()
out.release()

