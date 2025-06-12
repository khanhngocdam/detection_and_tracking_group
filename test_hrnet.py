import os
import cv2
import numpy as np
import torch
import argparse
from tqdm import tqdm


from lib.config import cfg
from lib.config import update_config
from lib.models.pose_hrnet import get_pose_net
from lib.core.inference import get_max_preds
from lib.utils.transforms import get_affine_transform

# Cấu hình keypoint và màu sắc
KEYPOINT_COLORS = [
    (255, 0, 0),    # Nose - Đỏ
    (255, 85, 0),   # Left eye - Cam
    (255, 170, 0),  # Right eye - Vàng cam
    (255, 255, 0),  # Left ear - Vàng
    (170, 255, 0),  # Right ear - Xanh lá nhạt
    (85, 255, 0),   # Left shoulder - Xanh lá đậm
    (0, 255, 0),    # Right shoulder - Xanh lá
    (0, 255, 85),   # Left elbow - Xanh lá ngọc
    (0, 255, 170),  # Right elbow - Xanh ngọc nhạt
    (0, 255, 255),  # Left wrist - Xanh ngọc
    (0, 170, 255),  # Right wrist - Xanh dương nhạt
    (0, 85, 255),   # Left hip - Xanh dương
    (0, 0, 255),    # Right hip - Xanh dương đậm
    (85, 0, 255),   # Left knee - Tím nhạt
    (170, 0, 255),  # Right knee - Tím
    (255, 0, 255),  # Left ankle - Hồng đậm
    (255, 0, 170)   # Right ankle - Hồng
]

# Cấu hình skeleton (các đường nối giữa các keypoint)
SKELETON_PAIRS = [
    (0, 1), (0, 2),  # Nose to eyes
    (1, 3), (2, 4),  # Eyes to ears
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 6), (5, 11), (6, 12),  # Shoulders to hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16)   # Right leg
]

SKELETON_COLORS = [(0, 215, 255) for _ in range(len(SKELETON_PAIRS))]  # Màu cam cho tất cả đường nối


def parse_args():
    parser = argparse.ArgumentParser(description='Process video with HRNet')
    
    parser.add_argument('--cfg', type=str, default='experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml',
                      help='path to config file')
    parser.add_argument('--modelDir', type=str, default='./models',
                      help='path to model directory')
    parser.add_argument('--logDir', type=str, default='./log',
                      help='path to log directory')
    parser.add_argument('--dataDir', type=str, default='./',
                      help='path to data directory')
    parser.add_argument('--prevModelDir', type=str, default=None,
                      help='path to previous Model directory')
    
    parser.add_argument('--video', type=str, required=True,
                      help='path to input video file')
    parser.add_argument('--output', type=str, required=True,
                      help='path to output video file')
    parser.add_argument('--model', type=str, default='models/pose_hrnet_w32_256x192.pth',
                      help='path to pretrained model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Cập nhật cấu hình
    update_config(cfg, args)
    
    return args


def load_model(config, model_path, device):
    """Tải mô hình HRNet"""
    model = get_pose_net(config, is_train=False)
    
    print(f"Đang tải model từ {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Xử lý trường hợp checkpoint có state_dict hoặc là state_dict trực tiếp
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    return model


def preprocess_frame(frame, config):
    """Tiền xử lý khung hình cho mô hình HRNet"""
    # Kích thước đầu ra
    input_width, input_height = config.MODEL.IMAGE_SIZE
    
    # Trích xuất thông tin ảnh
    height, width, _ = frame.shape
    
    # Tính toán tỷ lệ scale và dịch chuyển
    c = np.array([width // 2, height // 2])
    s = np.array([width, height], dtype=np.float32) / 200.0
    r = 0
    
    # Tạo biến đổi affine
    trans = get_affine_transform(c, s, r, [input_width, input_height])
    input_img = cv2.warpAffine(
        frame, trans, (input_width, input_height), flags=cv2.INTER_LINEAR
    )
    
    # Chuẩn hóa ảnh
    input_img = input_img.astype(np.float32) / 255.0
    input_img = (input_img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # Chuyển từ HWC sang CHW (Kênh, Chiều cao, Chiều rộng)
    input_img = input_img.transpose(2, 0, 1)
    
    # Chuyển thành tensor
    input_tensor = torch.from_numpy(input_img).unsqueeze(0)
    
    return input_tensor, c, s


def process_frame(frame, model, config, device):
    """Xử lý khung hình và phát hiện keypoint"""
    # Tiền xử lý khung hình
    input_tensor, c, s = preprocess_frame(frame, config)
    input_tensor = input_tensor.to(device)
    
    # Dự đoán từ mô hình
    with torch.no_grad():
        output = model(input_tensor)
    
    # Chuyển kết quả về CPU nếu cần
    if device != 'cpu':
        output = output.cpu()

    # Lấy tọa độ keypoints
    preds, maxvals = get_max_preds(output.numpy())
    
    # Chuyển đổi tọa độ về kích thước gốc của khung hình
    preds = preds[0]  # Batch size = 1
    maxvals = maxvals[0]
    
    # Kích thước đầu ra của mô hình
    h, w = config.MODEL.HEATMAP_SIZE
    
    # Tạo biến đổi affine ngược lại
    trans = get_affine_transform(c, s, 0, config.MODEL.HEATMAP_SIZE, inv=1)
    
    # Áp dụng biến đổi affine ngược cho tất cả các keypoint
    coords = np.zeros_like(preds)
    for i in range(preds.shape[0]):
        if maxvals[i] > 0.3:  # Lọc các keypoint có độ tin cậy thấp
            coords[i, 0:2] = preds[i, 0:2]
            coords[i, 0:2] = affine_transform(coords[i, 0:2], trans)
            
    return coords, maxvals


def affine_transform(pt, t):
    """Áp dụng biến đổi affine cho một điểm"""
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def draw_keypoints(frame, keypoints, confidences, threshold=0.3):
    """Vẽ keypoint lên khung hình"""
    result_frame = frame.copy()
    img_h, img_w, _ = frame.shape
    
    # Vẽ các skeleton trước để không bị che bởi các keypoint
    for pair_id, pair in enumerate(SKELETON_PAIRS):
        p1, p2 = pair
        if confidences[p1] > threshold and confidences[p2] > threshold:
            x1, y1 = int(keypoints[p1][0]), int(keypoints[p1][1])
            x2, y2 = int(keypoints[p2][0]), int(keypoints[p2][1])
            
            # Kiểm tra tọa độ có nằm trong khung hình không
            if 0 <= x1 < img_w and 0 <= y1 < img_h and 0 <= x2 < img_w and 0 <= y2 < img_h:
                cv2.line(result_frame, (x1, y1), (x2, y2), SKELETON_COLORS[pair_id], 2)
    
    # Vẽ các keypoint
    for i, (point, conf) in enumerate(zip(keypoints, confidences)):
        if conf > threshold:  # Chỉ vẽ những keypoint có độ tin cậy cao
            x, y = int(point[0]), int(point[1])
            
            # Kiểm tra tọa độ có nằm trong khung hình không
            if 0 <= x < img_w and 0 <= y < img_h:
                cv2.circle(result_frame, (x, y), 5, KEYPOINT_COLORS[i], -1)
    
    return result_frame


def process_video(args):
    """Xử lý video và lưu kết quả"""
    # Tải mô hình
    model = load_model(cfg, args.model, args.device)
    
    # Mở video đầu vào
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video {args.video}")
    
    # Lấy thông tin video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Chuẩn bị video đầu ra
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Xử lý từng frame
    with tqdm(total=total_frames, desc="Xử lý video") as pbar:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Phát hiện keypoint
            keypoints, confidences = process_frame(frame, model, cfg, args.device)
            
            # Vẽ keypoint lên frame
            result_frame = draw_keypoints(frame, keypoints, confidences)
            
            # Ghi frame kết quả
            out.write(result_frame)
            
            # Cập nhật thanh tiến trình
            frame_count += 1
            pbar.update(1)
    
    # Giải phóng tài nguyên
    cap.release()
    out.release()
    print(f"Đã xử lý xong! Kết quả được lưu tại {args.output}")


if __name__ == "__main__":
    args = parse_args()
    process_video(args)