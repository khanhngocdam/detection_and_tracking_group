import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import os

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer  # Thêm import này


# Cấu hình Detectron2 với Mask R-CNN
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # ngưỡng confidence
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

# Thư mục chứa ảnh đầu vào và thư mục kết quả
input_video = "./input/v3.mp4"
output_folder = "./output/test_detectron2_frames"
os.makedirs(output_folder, exist_ok=True)


# Đọc video đầu vào
cap = cv2.VideoCapture(input_video)
frame_idx = 0

if not cap.isOpened():
    print("Error: Không thể mở video")
    exit()

# Thêm list để lưu tất cả detections
all_detections = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    try:
        # Chạy Detectron2 trên frame
        outputs = predictor(frame)

        # Lấy instances và chuyển về CPU
        instances = outputs["instances"].to("cpu")

        # Lọc chỉ lấy người (class_id = 0)
        class_person = instances.pred_classes == 0
        instances = instances[class_person]

        # Lấy bounding boxes và scores của người
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()

        # Lưu detections cho frame hiện tại
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(float, box)  # Convert to float
            w = x2 - x1  # tính width
            h = y2 - y1  # tính height
            # Format cho DeepSORT: [frame_id, -1, bb_left, bb_top, width, height, conf]
            # frame_id bắt đầu từ 1
            detection = np.array([frame_idx + 1, -1, x1, y1, w, h, score])
            all_detections.append(detection)
        
         # Vẽ boxes lên frame
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Lưu frame đã xử lý vào thư mục output
        output_path = os.path.join(output_folder, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(output_path, frame)  # Lưu trực tiếp frame đã vẽ bbox
        frame_idx += 1
        
        print(f"Processed frame {frame_idx}", end='\r')
        
    except Exception as e:
        print(f"Lỗi khi xử lý frame {frame_idx}: {str(e)}")
        continue

cap.release()

# Chuyển list thành numpy array và lưu
all_detections = np.array(all_detections)
det_path = os.path.join(output_folder, 'det.npy')  # đổi tên file để phù hợp với DeepSORT
np.save(det_path, all_detections)
print(f"\nĐã lưu detections vào {det_path}")
print(f"Shape của detections: {all_detections.shape}")
# Format của mỗi dòng: [frame_id, track_id, bb_left, bb_top, width, height, confidence]