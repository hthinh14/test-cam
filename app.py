import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle # Thay joblib bằng pickle nếu bạn lưu model bằng pickle
from collections import deque
import time
import json
import os

# ==============================
# CẤU HÌNH & LOAD MODEL (Đã sửa lỗi load)
# ==============================
MODEL_PATH = r"D:/python/decision_tree_model.pkl"
SCALER_PATH = r"D:/python/scaler.pkl"
LABEL_MAP_PATH = r"D:/python/label_map.json" # Thêm đường dẫn cho label_map

SMOOTH_WINDOW = 8
FPS_SMOOTH = 0.9
EPS = 1e-6

# Hàm dự đoán từ Cây Quyết định tự viết
def predict_one(x, tree):
    if not isinstance(tree, dict):
        return tree
    if x[tree["feature"]] <= tree["threshold"]:
        return predict_one(x, tree["left"])
    else:
        return predict_one(x, tree["right"])

def predict(X, tree):
    return np.array([predict_one(x, tree) for x in X])

@st.cache_resource
def load_assets():
    try:
        # Load Decision Tree (Giả định tree object được lưu TRỰC TIẾP)
        with open(MODEL_PATH, "rb") as f:
            tree = pickle.load(f)

        # Load scaler (mean/std)
        with open(SCALER_PATH, "rb") as f:
            scaler_data = pickle.load(f)
        
        # Load label map
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)
        id2label = {v: k for k, v in label_map.items()}
        
        return tree, scaler_data["mean"], scaler_data["std"], id2label
    except FileNotFoundError as e:
        st.error(f"Lỗi File: Không tìm thấy file tài nguyên. Vui lòng kiểm tra đường dẫn: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi Load: Kiểm tra cấu trúc file .pkl/.json: {e}")
        st.stop()

# Tải tài sản
tree, mean, std, id2label = load_assets()
classes = list(id2label.values())

# ==============================
# 1️⃣ GIAO DIỆN STREAMLIT
# ==============================
st.set_page_config(page_title="Demo Buồn Ngủ", layout="wide")
st.title("😴 Nhận diện trạng thái tài xế bằng Decision Tree")
st.success(f"✅ Mô hình sẵn sàng! Các nhãn: {classes}")

# ==============================
# 3️⃣ FACE MESH VÀ HÀM TÍNH ĐẶC TRƯNG (Đã bổ sung các feature còn thiếu)
# ==============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

EYE_LEFT_IDX = np.array([33, 159, 145, 133, 153, 144])
EYE_RIGHT_IDX = np.array([362, 386, 374, 263, 380, 385])
MOUTH_IDX = np.array([61, 291, 0, 17, 78, 308])

def eye_aspect_ratio(landmarks, left=True):
    # ... (giữ nguyên logic tính ear) ...
    idx = EYE_LEFT_IDX if left else EYE_RIGHT_IDX
    pts = landmarks[idx, :2]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * (C + EPS))

def mouth_aspect_ratio(landmarks):
    # ... (giữ nguyên logic tính mar) ...
    pts = landmarks[MOUTH_IDX, :2]
    A = np.linalg.norm(pts[0] - pts[1])
    B = np.linalg.norm(pts[4] - pts[5])
    C = np.linalg.norm(pts[2] - pts[3])
    return (A + B) / (2.0 * (C + EPS))

def head_pose_yaw_pitch_roll(landmarks):
    # ... (giữ nguyên logic tính yaw, pitch, roll) ...
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
    # Hàm bổ sung 3 feature phụ còn thiếu
    nose, chin = landmarks[1], landmarks[152]
    angle_pitch_extra = np.degrees(np.arctan2(chin[1] - nose[1], (chin[2] - nose[2]) + EPS))
    forehead_y = np.mean(landmarks[[10, 338, 297, 332, 284], 1])
    cheek_dist = np.linalg.norm(landmarks[50] - landmarks[280])
    return angle_pitch_extra, forehead_y, cheek_dist

# ==============================
# 4️⃣ CAMERA + XỬ LÝ
# ==============================
st.sidebar.header("Điều khiển Camera")
use_cam = st.sidebar.checkbox("📷 Bật camera", value=True)
run_detection = st.sidebar.checkbox("▶️ Bắt đầu Nhận diện", value=False)
FRAME_WINDOW = st.empty()


if use_cam:
    if 'cap' not in st.session_state:
        st.session_state.cap = cv2.VideoCapture(0)

    # Khởi tạo hàng đợi trong session state để giữ trạng thái qua các lần rerun của streamlit
    if 'frame_queue' not in st.session_state:
        st.session_state.frame_queue = deque(maxlen=30)
    if 'pred_queue' not in st.session_state:
        st.session_state.pred_queue = deque(maxlen=8)

    pTime = 0
    fps = 0
    FPS_SMOOTH = 0.9
    
    status_text = st.empty()

    if run_detection:
        status_text.info("Đang thu thập 30 khung hình đầu tiên...")
        while st.session_state.cap.isOpened():
            ret, frame = st.session_state.cap.read()
            if not ret:
                status_text.error("Không đọc được camera!")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            
            # Khởi tạo final_label trước
            final_label = "Chờ khởi tạo"

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                landmarks = np.array([[p.x * w, p.y * h, p.z * w] for p in face.landmark])

                # Tính 9 đặc trưng cơ bản (Đã đầy đủ)
                ear_l = eye_aspect_ratio(landmarks, True)
                ear_r = eye_aspect_ratio(landmarks, False)
                mar = mouth_aspect_ratio(landmarks)
                yaw, pitch, roll = head_pose_yaw_pitch_roll(landmarks)
                angle_pitch_extra, forehead_y, cheek_dist = get_extra_features(landmarks)

                feat = np.array([ear_l, ear_r, mar, yaw, pitch, roll, 
                                 angle_pitch_extra, forehead_y, cheek_dist], dtype=np.float32)
                st.session_state.frame_queue.append(feat) # Thêm vào queue
                
                # ===== DỰ ĐOÁN KHI ĐỦ 30 KHUNG HÌNH (WINDOW_SIZE) =====
                if len(st.session_state.frame_queue) == 30:
                    status_text.success("Bắt đầu nhận diện...")
                    window = np.array(st.session_state.frame_queue)
                    
                    # 1. Tính Mean/Std (9+9)
                    mean_feats = window.mean(axis=0)
                    std_feats = window.std(axis=0)

                    # 2. Tính 6 đặc trưng thống kê động bổ sung
                    yaw_diff = np.mean(np.abs(np.diff(window[:, 3])))
                    pitch_diff = np.mean(np.abs(np.diff(window[:, 4])))
                    roll_diff = np.mean(np.abs(np.diff(window[:, 5])))
                    mar_max = np.max(window[:, 2])
                    mar_mean = np.mean(window[:, 2])
                    ear_mean = np.mean((window[:, 0] + window[:, 1]) / 2.0)
                    mar_ear_ratio = mar_mean / (ear_mean + EPS)
                    yaw_pitch_ratio = np.mean(np.abs(window[:, 3])) / (np.mean(np.abs(window[:, 4])) + EPS)
                    
                    # Ghép 24 đặc trưng
                    feats_24 = np.concatenate([mean_feats, std_feats,
                                               [yaw_diff, pitch_diff, roll_diff, 
                                                mar_max, mar_ear_ratio, yaw_pitch_ratio]])

                    # Chuẩn hóa dữ liệu
                    feats_scaled = (feats_24 - mean) / std

                    # Dự đoán (dùng hàm predict tự viết)
                    # feats_scaled phải có shape (1, 24)
                    pred_idx = predict(np.expand_dims(feats_scaled, axis=0), tree)[0] 
                    final_label = id2label.get(pred_idx, f"Class {pred_idx}")
                    st.session_state.pred_queue.append(final_label)

                    # Xóa 15 khung hình cũ (overlap)
                    for _ in range(15): 
                        if st.session_state.frame_queue:
                            st.session_state.frame_queue.popleft() 
                
            # ======== SMOOTH PREDICTION ========
            if len(st.session_state.pred_queue) > 0:
                final_label = max(set(st.session_state.pred_queue), key=st.session_state.pred_queue.count)
            
            # ======== HIỂN THỊ ========
            cTime = time.time()
            fps = FPS_SMOOTH * fps + (1 - FPS_SMOOTH) * (1 / (cTime - pTime + EPS))
            pTime = cTime

            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Trang thai: {final_label}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 3)

            FRAME_WINDOW.image(frame, channels='BGR')
            
    else:
        status_text.empty()
        # Đóng camera khi không chạy
        if st.session_state.cap.isOpened():
             st.session_state.cap.release()
        del st.session_state.cap # Xóa khỏi session state để reset
        
elif 'cap' in st.session_state:
    st.session_state.cap.release()
    del st.session_state.cap
    
st.markdown("---")
st.info("Nhấn 'Bắt đầu Nhận diện' ở thanh bên để khởi chạy mô hình.")