import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque
import time
import json
import os
import tempfile # Cần thiết để tạo file tạm thời cho video
import shutil # Cần thiết để xóa thư mục tạm

# ==============================
# CẤU HÌNH & LOAD MODEL (Giữ nguyên)
# ==============================
MODEL_PATH = "decision_tree_model.pkl"
SCALER_PATH = "scaler.pkl"
LABEL_MAP_PATH = "label_map.json"

SMOOTH_WINDOW = 8
FPS_SMOOTH = 0.9
EPS = 1e-6
WINDOW_SIZE = 30 # Cửa sổ khung hình

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
        with open(MODEL_PATH, "rb") as f:
            tree = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler_data = pickle.load(f)
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
# 3️⃣ FACE MESH VÀ HÀM TÍNH ĐẶC TRƯNG (Giữ nguyên)
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
# 1️⃣ GIAO DIỆN STREAMLIT
# ==============================
st.set_page_config(page_title="Demo Buồn Ngủ", layout="wide")
st.title("😴 Nhận diện trạng thái tài xế bằng Decision Tree")
st.success(f"Mô hình sẵn sàng! Các nhãn: {classes}")
st.markdown("---")

# ==============================
# 4️⃣ UPLOAD VIDEO (Giải pháp thay thế cho Camera)
# ==============================
uploaded_file = st.file_uploader("📂 Tải lên file video (.mp4, .avi) để phân tích", type=['mp4', 'avi'])

FRAME_WINDOW = st.empty()
status_text = st.empty()
progress_bar = st.progress(0)

if uploaded_file is not None:
    # --- Khởi tạo ---
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    
    frame_queue = deque(maxlen=WINDOW_SIZE)
    pred_queue = deque(maxlen=SMOOTH_WINDOW)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.info(f"Video đã tải: {frame_count} khung hình. Đang xử lý...")

    # --- Vòng lặp Xử lý Video ---
    processed_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frames += 1
        
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        final_label = "Đang xử lý..."
        
        if results.multi_face_landmarks:
            landmarks = np.array([[p.x * w, p.y * h, p.z * w] for p in results.multi_face_landmarks[0].landmark])

            # Tính 9 đặc trưng cơ bản
            ear_l = eye_aspect_ratio(landmarks, True); ear_r = eye_aspect_ratio(landmarks, False); mar = mouth_aspect_ratio(landmarks)
            yaw, pitch, roll = head_pose_yaw_pitch_roll(landmarks)
            angle_pitch_extra, forehead_y, cheek_dist = get_extra_features(landmarks)

            feat = np.array([ear_l, ear_r, mar, yaw, pitch, roll, 
                             angle_pitch_extra, forehead_y, cheek_dist], dtype=np.float32)
            st.session_state.frame_queue.append(feat) 

            # ===== DỰ ĐOÁN KHI ĐỦ 30 KHUNG HÌNH (WINDOW_SIZE) =====
            if len(st.session_state.frame_queue) == WINDOW_SIZE:
                window = np.array(st.session_state.frame_queue)
                
                # Tính 24 đặc trưng
                mean_feats = window.mean(axis=0); std_feats = window.std(axis=0)
                yaw_diff = np.mean(np.abs(np.diff(window[:, 3]))); pitch_diff = np.mean(np.abs(np.diff(window[:, 4]))); roll_diff = np.mean(np.abs(np.diff(window[:, 5])))
                mar_mean = np.mean(window[:, 2]); ear_mean = np.mean((window[:, 0] + window[:, 1]) / 2.0)
                mar_ear_ratio = mar_mean / (ear_mean + EPS); yaw_pitch_ratio = np.mean(np.abs(window[:, 3])) / (np.mean(np.abs(window[:, 4])) + EPS)
                feats_24 = np.concatenate([mean_feats, std_feats, [yaw_diff, pitch_diff, roll_diff, np.max(window[:, 2]), mar_ear_ratio, yaw_pitch_ratio]])

                # Chuẩn hóa, Dự đoán
                feats_scaled = (feats_24 - mean) / std
                pred_idx = predict(np.expand_dims(feats_scaled, axis=0), tree)[0] 
                pred_label = id2label.get(pred_idx, f"Class {pred_idx}")
                st.session_state.pred_queue.append(pred_label)

                # Xóa 15 khung hình cũ (overlap)
                for _ in range(15): 
                    if st.session_state.frame_queue:
                        st.session_state.frame_queue.popleft() 
        
        # ======== SMOOTH PREDICTION & HIỂN THỊ ========
        if len(st.session_state.pred_queue) > 0:
            final_label = max(set(st.session_state.pred_queue), key=st.session_state.pred_queue.count)
        else:
            final_label = "Đang chờ dữ liệu"

        cv2.putText(frame, f"Trang thai: {final_label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 3)
        FRAME_WINDOW.image(frame, channels='BGR')
        
        progress_bar.progress(processed_frames / frame_count)

    # --- Dọn dẹp sau khi kết thúc Video ---
    cap.release()
    os.unlink(tfile.name)
    progress_bar.empty()
    status_text.success("Phân tích video đã hoàn tất!")

# Khởi tạo session state cho hàng đợi
if 'frame_queue' not in st.session_state:
    st.session_state.frame_queue = deque(maxlen=WINDOW_SIZE)
if 'pred_queue' not in st.session_state:
    st.session_state.pred_queue = deque(maxlen=SMOOTH_WINDOW)