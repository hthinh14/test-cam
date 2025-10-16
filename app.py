import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import av 
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode 
import joblib
import os
from collections import deque

# ==============================
# CẤU HÌNH CƠ BẢN
# ==============================
# Lưu ý: Các file này phải nằm trong thư mục hiện tại hoặc đường dẫn tuyệt đối
MODEL_PATH = "softmax_model_best.pkl" # Chứa W, b, classes
SCALER_PATH = "scale.pkl"              # Chứa X_mean, X_std
LABEL_MAP_PATH = "label_map.json"

SMOOTH_WINDOW = 8
EPS = 1e-8 # Dùng 1e-8 như trong code huấn luyện
WINDOW_SIZE = 15 # Cửa sổ khung hình để tính đặc trưng thống kê

# ==============================
# HÀM DỰ ĐOÁN SOFTMAX
# ==============================
def softmax_predict(X, W, b):
    """
    Thực hiện dự đoán Softmax bằng các tham số W và b đã tải.
    X: Đặc trưng đã được scale (shape: (n_samples, n_features))
    W: Ma trận trọng số (shape: (n_features, n_classes))
    b: Vector bias (shape: (1, n_classes))
    """
    logits = X @ W + b
    # Tính toán softmax (không cần thiết vì chỉ cần argmax)
    # exp_z = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    # y_pred = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    # Lấy chỉ mục lớp có xác suất cao nhất
    return np.argmax(logits, axis=1)

@st.cache_resource
def load_assets():
    """Tải tham số W, b, scaler (mean, std) và label map"""
    try:
        # 1. Tải mô hình Softmax (W và b)
        with open(MODEL_PATH, "rb") as f:
            model_data = joblib.load(f)
            W = model_data["W"]
            b = model_data["b"]
            # CLASSES = model_data["classes"] # Có thể dùng để kiểm tra

        # 2. Tải scaler (mean và std)
        with open(SCALER_PATH, "rb") as f:
            scaler_data = joblib.load(f)
            # Sửa key theo code huấn luyện bạn đã gửi: X_mean và X_std
            mean_data = scaler_data["X_mean"] 
            std_data = scaler_data["X_std"]
            
        # 3. Tải label map
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)
        id2label = {int(v): k for k, v in label_map.items()} # Đảm bảo key là int
        
        # Trả về các tham số cần thiết
        return W, b, mean_data, std_data, id2label

    except FileNotFoundError as e:
        st.error(f"Lỗi File: Không tìm thấy file tài nguyên. Vui lòng kiểm tra đường dẫn: {e.filename}")
        st.stop()
    except KeyError as e:
        st.error(f"Lỗi Key: Kiểm tra cấu trúc file model/scaler (thiếu key: {e}).")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi Load: {e}")
        st.stop()

# Tải tài sản (Chạy một lần khi khởi động ứng dụng)
# Đã đổi tên biến từ 'model'/'tree' thành W, b
W, b, mean, std, id2label = load_assets() 
classes = list(id2label.values())


# ----------------------------------------------------------------------
## HÀM TÍNH ĐẶC TRƯNG (Giữ nguyên)
# ----------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
EYE_LEFT_IDX = np.array([33, 159, 145, 133, 153, 144])
EYE_RIGHT_IDX = np.array([362, 386, 374, 263, 380, 385])
MOUTH_IDX = np.array([61, 291, 0, 17, 78, 308])

def eye_aspect_ratio(landmarks, left=True):
    idx = EYE_LEFT_IDX if left else EYE_RIGHT_IDX
    pts = landmarks[idx, :2]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * (C + EPS))

def mouth_aspect_ratio(landmarks):
    pts = landmarks[MOUTH_IDX, :2]
    A = np.linalg.norm(pts[0] - pts[1])
    B = np.linalg.norm(pts[4] - pts[5])
    C = np.linalg.norm(pts[2] - pts[3])
    return (A + B) / (2.0 * (C + EPS))

def head_pose_yaw_pitch_roll(landmarks):
    left_eye = landmarks[33][:2]
    right_eye = landmarks[263][:2]
    nose = landmarks[1][:2]
    chin = landmarks[152][:2]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    roll = np.degrees(np.arctan2(dy, dx + EPS))
    interocular = np.linalg.norm(right_eye - left_eye) + EPS
    eyes_center = (left_eye + right_eye) / 2.0
    yaw = np.degrees(np.arctan2((nose[0] - eyes_center[0]), interocular))
    baseline = chin - eyes_center
    pitch = np.degrees(np.arctan2((nose[1] - eyes_center[1]), (np.linalg.norm(baseline) + EPS)))
    return yaw, pitch, roll

def get_extra_features(landmarks):
    nose, chin = landmarks[1], landmarks[152]
    # Lưu ý: Tính toán z-coord có thể không chính xác/ổn định từ MediaPipe/Streamlit
    angle_pitch_extra = np.degrees(np.arctan2(chin[1] - nose[1], (chin[2] - nose[2]) + EPS))
    forehead_y = np.mean(landmarks[[10, 338, 297, 332, 284], 1])
    cheek_dist = np.linalg.norm(landmarks[50] - landmarks[280])
    return angle_pitch_extra, forehead_y, cheek_dist


# ----------------------------------------------------------------------
## WEBRTC VIDEO PROCESSOR (Logic xử lý Real-time)
# ----------------------------------------------------------------------
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        # Tải tham số Softmax
        self.W = W 
        self.b = b
        self.mean = mean
        self.std = std
        self.id2label = id2label
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.frame_queue = deque(maxlen=WINDOW_SIZE)
        self.pred_queue = deque(maxlen=SMOOTH_WINDOW)
        self.last_pred_label = "CHỜ DỮ LIỆU" 

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frame_array = frame.to_ndarray(format="bgr24")
        h, w = frame_array.shape[:2]
        rgb = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        # --- 1. TRÍCH XUẤT ĐẶC TRƯNG ---
        if results.multi_face_landmarks:
            landmarks = np.array([[p.x * w, p.y * h, p.z * w] for p in results.multi_face_landmarks[0].landmark])

            # Tính 9 đặc trưng cơ bản
            ear_l = eye_aspect_ratio(landmarks, True); ear_r = eye_aspect_ratio(landmarks, False); mar = mouth_aspect_ratio(landmarks)
            yaw, pitch, roll = head_pose_yaw_pitch_roll(landmarks)
            angle_pitch_extra, forehead_y, cheek_dist = get_extra_features(landmarks)

            feat = np.array([ear_l, ear_r, mar, yaw, pitch, roll, 
                              angle_pitch_extra, forehead_y, cheek_dist], dtype=np.float32)
            self.frame_queue.append(feat) 
            
            # --- 2. DỰ ĐOÁN KHI ĐỦ 30 KHUNG HÌNH ---
            if len(self.frame_queue) == WINDOW_SIZE:
                window = np.array(self.frame_queue)
                
                # Tính 24 đặc trưng (giữ nguyên logic của bạn)
                mean_feats = window.mean(axis=0); std_feats = window.std(axis=0)
                yaw_diff = np.mean(np.abs(np.diff(window[:, 3]))); pitch_diff = np.mean(np.abs(np.diff(window[:, 4]))); roll_diff = np.mean(np.abs(np.diff(window[:, 5])))
                mar_mean = np.mean(window[:, 2]); ear_mean = np.mean((window[:, 0] + window[:, 1]) / 2.0)
                mar_ear_ratio = mar_mean / (ear_mean + EPS); yaw_pitch_ratio = np.mean(np.abs(window[:, 3])) / (np.mean(np.abs(window[:, 4])) + EPS)
                feats_24 = np.concatenate([mean_feats, std_feats, [yaw_diff, pitch_diff, roll_diff, np.max(window[:, 2]), mar_ear_ratio, yaw_pitch_ratio]])

                # Chuẩn hóa
                feats_scaled = (feats_24 - self.mean) / self.std
                
                # SỬ DỤNG HÀM DỰ ĐOÁN SOFTMAX TỰ XÂY DỰNG
                # (Đảm bảo input có shape (1, 24) cho hàm predict)
                pred_idx = softmax_predict(np.expand_dims(feats_scaled, axis=0), self.W, self.b)[0] 
                
                pred_label = self.id2label.get(pred_idx, f"Class {pred_idx}")
                self.pred_queue.append(pred_label)

                # Xóa 15 khung hình cũ (overlap)
                for _ in range(7):
                    if self.frame_queue:
                        self.frame_queue.popleft() 
        
        # --- 3. SMOOTHING ---
        if len(self.pred_queue) > 0:
            # Lấy nhãn xuất hiện nhiều nhất trong cửa sổ làm mịn
            self.last_pred_label = max(set(self.pred_queue), key=self.pred_queue.count)

        # --- 4. HIỂN THỊ KẾT QUẢ ---
        cv2.putText(frame_array, f"Trang thai: {self.last_pred_label.upper()}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 3)

        return av.VideoFrame.from_ndarray(frame_array, format="bgr24")

# ----------------------------------------------------------------------
## GIAO DIỆN STREAMLIT CHÍNH
# ----------------------------------------------------------------------
st.set_page_config(page_title="Demo Softmax", layout="wide")
st.title("🧠 Nhận diện trạng thái mất tập trung bằng mô hình học máy.")
st.success(f"Mô hình sẵn sàng! Các nhãn: {classes}")
st.warning("Vui lòng chấp nhận yêu cầu truy cập camera từ trình duyệt của bạn.")
st.markdown("---")


# Khởi tạo WebRTC Streamer
webrtc_streamer(
    key="softmax_driver_live",
    mode=WebRtcMode.SENDRECV,
    # Cấu hình STUN servers để thiết lập kết nối
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_processor_factory=DrowsinessProcessor,
    media_stream_constraints={"video": True, "audio": False}, # Chỉ bật video
    async_processing=True, # Cho phép xử lý không đồng bộ (tăng tốc độ)
)