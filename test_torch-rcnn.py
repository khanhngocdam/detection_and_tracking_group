import os
import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm

# Thiết lập thiết bị (GPU nếu có)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải mô hình Faster R-CNN đã được huấn luyện sẵn
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# Danh sách các lớp trong COCO dataset
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Thư mục chứa các khung hình
input_folder = 'v3'
output_folder = 'output_frames_1'
os.makedirs(output_folder, exist_ok=True)

# Ngưỡng điểm tin cậy để hiển thị bounding box
confidence_threshold = 0.7

# Lấy danh sách các tệp ảnh trong thư mục và sắp xếp
frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))])

# Lặp qua từng khung hình
for frame_file in tqdm(frame_files, desc="Processing frames"):
    frame_path = os.path.join(input_folder, frame_file)
    image = Image.open(frame_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        outputs = model([image_tensor])

    # Lấy kết quả dự đoán
    boxes = outputs[0]['boxes'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()

    # Đọc ảnh bằng OpenCV để vẽ bounding box
    image_cv = cv2.imread(frame_path)

    for box, label, score in zip(boxes, labels, scores):
        if score >= confidence_threshold and COCO_INSTANCE_CATEGORY_NAMES[label] == 'person':
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_cv, f'Person: {score:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Lưu khung hình đã xử lý
    output_path = os.path.join(output_folder, frame_file)
    cv2.imwrite(output_path, image_cv)

# Tạo video từ các khung hình đã xử lý
output_video_path = 'output_video.mp4'
frame_example = cv2.imread(os.path.join(output_folder, frame_files[0]))
height, width, _ = frame_example.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30  # Điều chỉnh theo nhu cầu
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

for frame_file in frame_files:
    frame_path = os.path.join(output_folder, frame_file)
    frame = cv2.imread(frame_path)
    video_writer.write(frame)

video_writer.release()
print(f"Video đã được lưu tại: {output_video_path}")
