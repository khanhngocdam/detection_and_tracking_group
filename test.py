import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import torch
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from tranfer_coco_to_hm36 import coco_to_h36m
# Cấu hình Mask R-CNN
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # confidence threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

# Thư mục chứa ảnh đầu vào và thư mục kết quả
input_folder = "./TS18"
output_folder = "output_frames_detectron2"
os.makedirs(output_folder, exist_ok=True)

# Duyệt qua tất cả các ảnh trong thư mục input
for filename in sorted(os.listdir(input_folder)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Dự đoán
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        class_ids = instances.pred_classes.numpy()

        # Chỉ lấy các instance là người (class ID = 0)
        person_instances = instances[class_ids == 0]

        # Vẽ kết quả
        v = Visualizer(image[:, :, ::-1], metadata, scale=1.2)
        v = v.draw_instance_predictions(person_instances)
        result_image = v.get_image()[:, :, ::-1]

        # Lưu kết quả
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, result_image)
        print(f"Đã xử lý và lưu: {output_path}")

print("✅ Đã hoàn tất xử lý tất cả ảnh trong thư mục.")
