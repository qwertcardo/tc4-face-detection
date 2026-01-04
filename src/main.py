"""
Versão final (multi-rostos) — Dlib + RetinaFace (DeepFace) para boxes,
MediaPipe Tasks (FaceLandmarker) para landmarks, com:

- Tracking por IoU (IDs persistentes)
- Contagem de piscadas por ID (EAR com EMA + histerese)
- Contagem de aberturas de boca por ID (MAR com EMA + histerese)
- Desenho: caixa do rosto + pontos dos olhos e boca + texto por rosto
- Suporte a múltiplos rostos no mesmo vídeo

Dependências:
  pip install opencv-python mediapipe numpy tqdm deepface dlib

Modelo:
  baixe e coloque o FaceLandmarker .task em:
    models/face_landmarker.task
"""

import os
import cv2
import dlib
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

from deepface import DeepFace

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "assets", "video.mp4")
OUTPUT_PATH = os.path.join(BASE_DIR, "assets", "output", "output_1.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")


# -----------------------------
# FaceMesh indices (olhos/boca)
# -----------------------------
# Para desenhar (conjuntos maiores):
LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
    95, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    415, 310, 311, 312, 13, 82, 81, 42, 183, 78
]

# Para EAR (6 pontos):
LEFT_EAR_IDX  = [33, 160, 158, 133, 153, 144]   # p1,p2,p3,p4,p5,p6
RIGHT_EAR_IDX = [362, 385, 387, 263, 373, 380]  # p1,p2,p3,p4,p5,p6

# Para MAR (4 pontos):
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_UP = 13
MOUTH_DOWN = 14


# -----------------------------
# Utilitários geométricos
# -----------------------------
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter_area = iw * ih

    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    union = area_a + area_b - inter_area

    return 0.0 if union == 0 else inter_area / union


def clip_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(x1), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    x2 = max(1, min(int(x2), W))   # exclusivo
    y2 = max(1, min(int(y2), H))   # exclusivo
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)
    return x1, y1, x2, y2


def square_expand_box(x1, y1, x2, y2, W, H, scale=1.25):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1)
    bh = (y2 - y1)
    side = max(bw, bh) * scale
    nx1 = cx - side / 2
    ny1 = cy - side / 2
    nx2 = cx + side / 2
    ny2 = cy + side / 2
    return clip_box(nx1, ny1, nx2, ny2, W, H)


# -----------------------------
# Detecção de faces: RetinaFace (DeepFace) + Dlib
# -----------------------------
def deepface_retina_boxes(bgr_frame):
    faces = DeepFace.extract_faces(
        img_path=bgr_frame,
        detector_backend="retinaface",
        enforce_detection=False
    )
    out = []
    for f in faces:
        fa = f.get("facial_area", {})
        x, y, w, h = fa.get("x"), fa.get("y"), fa.get("w"), fa.get("h")
        if None in (x, y, w, h):
            continue
        score = f.get("confidence", None)
        out.append((x, y, x + w, y + h, score))
    return out


def dlib_boxes(gray_frame, detector, upsample=0):
    rects = detector(gray_frame, upsample)
    out = []
    for r in rects:
        out.append((r.left(), r.top(), r.right() + 1, r.bottom() + 1, None))
    return out


def merge_boxes(dlib_list, retina_list, iou_thr=0.4):
    """
    dlib_list: [(x1,y1,x2,y2,None), ...]
    retina_list: [(x1,y1,x2,y2,score), ...]
    Prioriza RetinaFace quando há match.
    """
    merged = []
    used_retina = set()

    for dx1, dy1, dx2, dy2, _ in dlib_list:
        best_j = None
        best_iou = 0.0
        for j, (rx1, ry1, rx2, ry2, score) in enumerate(retina_list):
            if j in used_retina:
                continue
            v = iou((dx1, dy1, dx2, dy2), (rx1, ry1, rx2, ry2))
            if v > best_iou:
                best_iou = v
                best_j = j

        if best_j is not None and best_iou >= iou_thr:
            merged.append(retina_list[best_j])
            used_retina.add(best_j)
        else:
            merged.append((dx1, dy1, dx2, dy2, None))

    for j, box in enumerate(retina_list):
        if j not in used_retina:
            merged.append(box)

    return merged


# -----------------------------
# EAR / MAR
# -----------------------------
def ear_from_landmarks(landmarks, eye_idx, w, h):
    pts = []
    for i in eye_idx:
        x = landmarks[i].x * w
        y = landmarks[i].y * h
        pts.append((x, y))
    p1, p2, p3, p4, p5, p6 = pts
    A = np.linalg.norm(np.array(p2) - np.array(p6))
    B = np.linalg.norm(np.array(p3) - np.array(p5))
    C = np.linalg.norm(np.array(p1) - np.array(p4))
    return 0.0 if C == 0 else (A + B) / (2.0 * C)


def mar_from_landmarks(landmarks, w, h):
    def pt(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h], dtype=np.float32)

    left = pt(MOUTH_LEFT)
    right = pt(MOUTH_RIGHT)
    up = pt(MOUTH_UP)
    down = pt(MOUTH_DOWN)

    horiz = np.linalg.norm(right - left)
    vert = np.linalg.norm(down - up)
    return 0.0 if horiz == 0 else vert / horiz


# -----------------------------
# Tracker IoU + Estado por rosto
# -----------------------------
@dataclass
class FaceState:
    # Blink
    blinks: int = 0
    closed_frames: int = 0
    was_closed: bool = False
    ear_ema: float = 0.0

    # Mouth open
    mouth_opens: int = 0
    open_frames: int = 0
    was_open: bool = False
    mar_ema: float = 0.0

    last_seen: int = 0


class IoUTracker:
    def __init__(self, iou_thr=0.3, max_missed=15):
        self.iou_thr = iou_thr
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks = {}   # id -> (box(x1,y1,x2,y2), missed)
        self.states = {}   # id -> FaceState

    def update(self, boxes, frame_idx):
        # boxes: [(x1,y1,x2,y2,score), ...]
        assigned = {}
        used_track_ids = set()

        # Match detections -> existing tracks
        for b in boxes:
            x1, y1, x2, y2, _ = b
            best_id = None
            best = 0.0

            for tid, (tbox, missed) in self.tracks.items():
                if tid in used_track_ids:
                    continue
                v = iou((x1, y1, x2, y2), tbox)
                if v > best:
                    best = v
                    best_id = tid

            if best_id is not None and best >= self.iou_thr:
                assigned[best_id] = b
                used_track_ids.add(best_id)
            else:
                tid = self.next_id
                self.next_id += 1
                assigned[tid] = b
                used_track_ids.add(tid)

        # Build new tracks, reset missed
        new_tracks = {}
        for tid, b in assigned.items():
            x1, y1, x2, y2, _ = b
            new_tracks[tid] = ((x1, y1, x2, y2), 0)
            if tid not in self.states:
                self.states[tid] = FaceState()
            self.states[tid].last_seen = frame_idx

        # Carry over unmatched old tracks (increment missed)
        for tid, (tbox, missed) in self.tracks.items():
            if tid not in new_tracks:
                missed += 1
                if missed <= self.max_missed:
                    new_tracks[tid] = (tbox, missed)

        self.tracks = new_tracks
        return assigned  # dict: id -> box(x1,y1,x2,y2,score)


# -----------------------------
# Atualizações robustas (EMA + histerese)
# -----------------------------
# Blink (ajuste se necessário)
EAR_CLOSE = 0.21
EAR_OPEN = 0.24
MIN_CLOSED_FRAMES = 2
EAR_EMA_ALPHA = 0.35

# Mouth (ajuste se necessário)
MAR_OPEN = 0.35
MAR_CLOSE = 0.30
MIN_OPEN_FRAMES = 2
MAR_EMA_ALPHA = 0.35


def update_blink(state: FaceState, ear_raw: float) -> float:
    if state.ear_ema == 0.0:
        state.ear_ema = ear_raw
    else:
        state.ear_ema = EAR_EMA_ALPHA * ear_raw + (1.0 - EAR_EMA_ALPHA) * state.ear_ema

    ear = state.ear_ema

    if ear < EAR_CLOSE:
        state.closed_frames += 1
        state.was_closed = True
    elif ear > EAR_OPEN:
        if state.was_closed and state.closed_frames >= MIN_CLOSED_FRAMES:
            state.blinks += 1
        state.closed_frames = 0
        state.was_closed = False

    return ear


def update_mouth(state: FaceState, mar_raw: float) -> float:
    if state.mar_ema == 0.0:
        state.mar_ema = mar_raw
    else:
        state.mar_ema = MAR_EMA_ALPHA * mar_raw + (1.0 - MAR_EMA_ALPHA) * state.mar_ema

    mar = state.mar_ema

    if mar > MAR_OPEN:
        state.open_frames += 1
        state.was_open = True
    elif mar < MAR_CLOSE:
        if state.was_open and state.open_frames >= MIN_OPEN_FRAMES:
            state.mouth_opens += 1
        state.open_frames = 0
        state.was_open = False

    return mar


# -----------------------------
# Desenho no frame global
# -----------------------------
def draw_points_global(frame, landmarks, idx_list, roi_x1, roi_y1, roi_w, roi_h, color, radius=2):
    for idx in idx_list:
        lm = landmarks[idx]
        gx = roi_x1 + int(lm.x * roi_w)
        gy = roi_y1 + int(lm.y * roi_h)
        cv2.circle(frame, (gx, gy), radius, color, -1)


# -----------------------------
# Main
# -----------------------------
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo .task não encontrado em: {MODEL_PATH}")

    # MediaPipe Tasks - FaceLandmarker (VIDEO)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    mediapipe_detector = vision.FaceLandmarker.create_from_options(options)

    # Detectores de boxes
    dlib_detector = dlib.get_frontal_face_detector()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Erro abrindo vídeo: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    tracker = IoUTracker(iou_thr=0.3, max_missed=15)

    frame_idx = 0
    ts_count = 0;
    with tqdm(total=total_frames if total_frames > 0 else None, desc="Processando", unit="frame") as pbar:
        while True:
            ts_count += 1
            ret, frame = cap.read()
            if not ret:
                break

            # timestamp em ms (base do VIDEO mode)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC) + ts_count)

            # 1) Boxes por RetinaFace + Dlib
            retina_boxes = deepface_retina_boxes(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dlib_faces = dlib_boxes(gray, dlib_detector)

            combined = merge_boxes(dlib_faces, retina_boxes, iou_thr=0.4)

            # 2) Tracking (IDs)
            assigned = tracker.update(combined, frame_idx)

            # 3) Para cada rosto (por ID), roda landmarks na ROI e atualiza contadores por ID
            for face_id, (x1, y1, x2, y2, score) in assigned.items():
                # filtro de score (RetinaFace). Se score None (dlib), aceita.
                if score is not None and score < 0.6:
                    continue

                # evita ROIs minúsculas (landmarks ficam instáveis)
                if (x2 - x1) < 60 or (y2 - y1) < 60:
                    continue

                roi_x1, roi_y1, roi_x2, roi_y2 = square_expand_box(x1, y1, x2, y2, width, height, scale=1.25)
                roi_bgr = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                if roi_bgr.size == 0:
                    continue

                roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                mp_roi = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)

                # Garantir monotonicidade mesmo com múltiplas ROIs no mesmo frame:
                # (timestamp_ms + face_id) mantém crescimento ao longo do vídeo
                result = mediapipe_detector.detect_for_video(mp_roi, timestamp_ms + face_id)

                # Desenha box do detector (verde)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                state = tracker.states[face_id]

                if not result.face_landmarks:
                    cv2.putText(frame, f"ID {face_id} (sem landmarks)", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    continue

                lms = result.face_landmarks[0]
                roi_w = roi_x2 - roi_x1
                roi_h = roi_y2 - roi_y1

                # EAR / MAR raw
                left_ear = ear_from_landmarks(lms, LEFT_EAR_IDX, roi_w, roi_h)
                right_ear = ear_from_landmarks(lms, RIGHT_EAR_IDX, roi_w, roi_h)
                ear_raw = (left_ear + right_ear) / 2.0
                mar_raw = mar_from_landmarks(lms, roi_w, roi_h)

                # Atualiza estados (EMA + histerese)
                ear = update_blink(state, ear_raw)
                mar = update_mouth(state, mar_raw)

                # Desenha landmarks (olhos verdes, boca vermelha)
                draw_points_global(frame, lms, LEFT_EYE_IDX, roi_x1, roi_y1, roi_w, roi_h, (0, 255, 0), radius=2)
                draw_points_global(frame, lms, RIGHT_EYE_IDX, roi_x1, roi_y1, roi_w, roi_h, (0, 255, 0), radius=2)
                draw_points_global(frame, lms, MOUTH_IDX, roi_x1, roi_y1, roi_w, roi_h, (0, 0, 255), radius=2)

                # HUD por face
                label = f"ID {face_id}  blinks:{state.blinks}  EAR:{ear:.3f}  mouth:{state.mouth_opens}  MAR:{mar:.3f}"
                cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            out.write(frame)

            frame_idx += 1
            pbar.update(1)

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    mediapipe_detector.close()

    # Resumo final (por ID que ainda existe no tracker.states)
    # Observação: IDs podem “sumir” se rostos saírem do frame; o estado fica no dict até acabar o vídeo.
    print("Resumo por rosto (ID):")
    for face_id, st in sorted(tracker.states.items(), key=lambda x: x[0]):
        print(f"  ID {face_id}: blinks={st.blinks}, mouth_opens={st.mouth_opens}")

    print(f"\nVídeo salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()