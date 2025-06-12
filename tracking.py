import numpy as np
import cv2
import os
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.application_util import preprocessing

# ===== CẤU HÌNH =====
frames_dir = "./output/test_detectron2_frames"
detection_file = os.path.join(frames_dir, "det.npy")
reid_model_path = "./deep_sort/resources/networks/mars-small128.pb"
output_video = "./output/tracked_output.mp4"

# ===== TẠI MÔ HÌNH REID =====
max_cosine_distance = 0.4
nn_budget = 100
model = gdet.create_box_encoder(reid_model_path, batch_size=32)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# ===== LOAD DETECTIONS =====
detections = np.load(detection_file)

# ===== LOAD FRAME LIST =====
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
total_frames = len(frame_files)

# ===== VIDEO WRITER (nếu muốn lưu video tracking) =====
sample_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
h, w = sample_frame.shape[:2]
video_path = './input/v3.mp4'
# Mở video
cap = cv2.VideoCapture(video_path)
# Kiểm tra xem video có mở được không
if not cap.isOpened():
    print("Không mở được video.")
else:
    # Lấy FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
# Giải phóng video
cap.release()
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# ===== TRACKING VỚI DEEPSORT =====
for frame_idx, frame_name in enumerate(frame_files):
    frame = cv2.imread(os.path.join(frames_dir, frame_name))
    current_frame_id = frame_idx + 1

    # Lấy tất cả detection của frame hiện tại
    frame_dets = detections[detections[:, 0] == current_frame_id]
    bboxes = frame_dets[:, 2:6]  # x, y, w, h
    scores = frame_dets[:, 6]

    features = model(frame, bboxes)
    det_objects = [Detection(bbox, score, feat) for bbox, score, feat in zip(bboxes, scores, features)]

    # NMS
    boxes = np.array([d.tlwh for d in det_objects])
    scores = np.array([d.confidence for d in det_objects])
    indices = preprocessing.non_max_suppression(boxes, 0.8, scores)
    det_objects = [det_objects[i] for i in indices]

    # UPDATE TRACKER
    tracker.predict()
    tracker.update(det_objects)

    # DRAW TRACKS
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        x, y, w, h = track.to_tlwh()
        track_id = track.track_id
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)
        cv2.putText(frame, f'ID {track_id}', (int(x), int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    # Hiển thị / lưu video
    out.write(frame)
    print(f"Tracking frame {current_frame_id}/{total_frames}", end='\r')

out.release()
print("\n🎉 Tracking hoàn tất. Video lưu tại:", output_video)
