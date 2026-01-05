"""
Versão final (multi-rostos) — Dlib + RetinaFace (DeepFace) para boxes,
MediaPipe Tasks (FaceLandmarker) para landmarks, com:

- Tracking por rosto (ReID): IoU + embedding (DeepFace.represent)
- Estado dos olhos por ID: OPEN/CLOSED (EAR com EMA + histerese)
- Desenho: caixa + pontos dos olhos + label por rosto
- Timestamp monotônico no MediaPipe (VIDEO mode)

Dependências:
  pip install opencv-python mediapipe numpy tqdm deepface dlib

Modelo:
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
VIDEO_PATH = os.path.join(BASE_DIR, "../assets", "video.mp4")
OUTPUT_PATH = os.path.join(BASE_DIR, "../assets", "output", "output_2.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "../models", "face_landmarker.task")


# -----------------------------
# FaceMesh indices (olhos/boca)
# -----------------------------
LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

LEFT_EAR_IDX  = [33, 160, 158, 133, 153, 144]   # p1,p2,p3,p4,p5,p6
RIGHT_EAR_IDX = [362, 385, 387, 263, 373, 380]  # p1,p2,p3,p4,p5,p6


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
    x2 = max(1, min(int(x2), W))
    y2 = max(1, min(int(y2), H))
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


def filter_boxes(boxes, W, H, max_area_ratio=0.60):
    out = []
    frame_area = float(W * H)

    for (x1, y1, x2, y2, score) in boxes:
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        if bw == 0 or bh == 0:
            continue

        area_ratio = (bw * bh) / frame_area
        if area_ratio >= max_area_ratio:
            continue

        out.append((x1, y1, x2, y2, score))

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
# EAR
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


# -----------------------------
# Olhos: EMA + histerese (sem contagem)
# -----------------------------
EAR_CLOSE = 0.21
EAR_OPEN  = 0.24
EAR_EMA_ALPHA = 0.35


@dataclass
class FaceState:
    ear_ema: float = 0.0
    eyes_closed: bool = False   # estado final (histerese)
    last_seen: int = 0

    # ReID
    embedding: np.ndarray | None = None
    last_embed_frame: int = -9999


def update_eyes_state(state: FaceState, ear_raw: float) -> float:
    if state.ear_ema == 0.0:
        state.ear_ema = ear_raw
    else:
        state.ear_ema = EAR_EMA_ALPHA * ear_raw + (1.0 - EAR_EMA_ALPHA) * state.ear_ema

    ear = state.ear_ema

    # Histerese: só muda quando cruza o limiar oposto
    if state.eyes_closed:
        if ear > EAR_OPEN:
            state.eyes_closed = False
    else:
        if ear < EAR_CLOSE:
            state.eyes_closed = True

    return ear


# -----------------------------
# DeepFace embedding (ReID)
# -----------------------------
def get_embedding_from_roi(roi_bgr, model_name="Facenet512"):
    """
    Retorna embedding L2-normalizado (np.ndarray shape (d,))
    """
    try:
        reps = DeepFace.represent(
            img_path=roi_bgr,
            model_name=model_name,
            detector_backend="skip",     # ROI já é o rosto
            enforce_detection=False,
            normalization="base"
        )
        if not reps:
            return None
        emb = np.array(reps[0]["embedding"], dtype=np.float32)
        n = np.linalg.norm(emb)
        return emb if n == 0 else emb / n
    except Exception:
        return None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    return float(np.dot(a, b))

def is_hard_cut(prev_gray_small, gray_small, thr=18.0):
    # diff médio absoluto
    diff = cv2.absdiff(prev_gray_small, gray_small)
    return float(diff.mean()) > thr

# -----------------------------
# Tracker ReID (IoU + embedding)
# -----------------------------
class ReIDTracker:
    def __init__(
        self,
        iou_thr=0.25,
        sim_thr=0.55,
        max_missed=20,
        embed_every_n_frames=10,
        model_name="Facenet512"
    ):
        self.iou_thr = iou_thr
        self.sim_thr = sim_thr
        self.max_missed = max_missed
        self.embed_every_n_frames = embed_every_n_frames
        self.model_name = model_name

        self.next_id = 1
        self.tracks = {}  # id -> (box(x1,y1,x2,y2), missed)
        self.states = {}  # id -> FaceState

    def _ensure_state(self, tid):
        if tid not in self.states:
            self.states[tid] = FaceState()

    def update(self, detections, frame, frame_idx, W, H):
        """
        detections: list[(x1,y1,x2,y2,score)]
        retorna: dict[id] -> (x1,y1,x2,y2,score)
        """
        assigned = {}
        used_track_ids = set()

        # Pré-calcula ROI e embedding para detecções (somente quando necessário)
        det_infos = []
        for (x1, y1, x2, y2, score) in detections:
            # filtro básico (mantém seu comportamento)
            if score is not None and score < 0.6:
                continue
            if (x2 - x1) < 60 or (y2 - y1) < 60:
                continue

            roi_x1, roi_y1, roi_x2, roi_y2 = square_expand_box(x1, y1, x2, y2, W, H, scale=1.25)
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            if roi.size == 0:
                continue

            det_infos.append({
                "box": (x1, y1, x2, y2),
                "score": score,
                "roi_box": (roi_x1, roi_y1, roi_x2, roi_y2),
                "roi": roi,
                "emb": None,
            })

        # Matching: para cada detecção, tenta:
        # 1) IoU forte -> mesmo ID
        # 2) senão, embedding parecido -> mesmo ID (ReID)
        for det in det_infos:
            x1, y1, x2, y2 = det["box"]

            best_id = None
            best_iou = 0.0

            # 1) IoU match
            for tid, (tbox, missed) in self.tracks.items():
                if tid in used_track_ids:
                    continue
                v = iou((x1, y1, x2, y2), tbox)
                if v > best_iou:
                    best_iou = v
                    best_id = tid

            if best_id is not None and best_iou >= self.iou_thr:
                # match por IoU
                tid = best_id
                assigned[tid] = (x1, y1, x2, y2, det["score"], det["roi_box"], det["roi"])
                used_track_ids.add(tid)
                self._ensure_state(tid)
                self.states[tid].last_seen = frame_idx
                continue

            # 2) ReID match (embedding)
            # calcula embedding da detecção
            det["emb"] = get_embedding_from_roi(det["roi"], model_name=self.model_name)

            best_tid = None
            best_sim = -1.0
            for tid, (tbox, missed) in self.tracks.items():
                if tid in used_track_ids:
                    continue
                self._ensure_state(tid)
                st = self.states[tid]
                if st.embedding is None or det["emb"] is None:
                    continue
                sim = cosine_sim(det["emb"], st.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_tid = tid

            if best_tid is not None and best_sim >= self.sim_thr:
                tid = best_tid
                assigned[tid] = (x1, y1, x2, y2, det["score"], det["roi_box"], det["roi"])
                used_track_ids.add(tid)
                self.states[tid].last_seen = frame_idx

                # opcional: atualiza embedding do track com o da detecção (média móvel simples)
                st = self.states[tid]
                if det["emb"] is not None:
                    if st.embedding is None:
                        st.embedding = det["emb"]
                    else:
                        st.embedding = st.embedding * 0.7 + det["emb"] * 0.3
                        n = np.linalg.norm(st.embedding)
                        if n != 0:
                            st.embedding = st.embedding / n
                continue

            # 3) novo ID
            tid = self.next_id
            self.next_id += 1
            assigned[tid] = (x1, y1, x2, y2, det["score"], det["roi_box"], det["roi"])
            used_track_ids.add(tid)
            self._ensure_state(tid)
            self.states[tid].last_seen = frame_idx

            # guarda embedding inicial, se houver
            st = self.states[tid]
            st.embedding = det["emb"]

        # Atualiza tracks: reset missed para assigned, incrementa missed para os demais
        new_tracks = {}
        for tid, (x1, y1, x2, y2, score, roi_box, roi) in assigned.items():
            new_tracks[tid] = ((x1, y1, x2, y2), 0)

        for tid, (tbox, missed) in self.tracks.items():
            if tid not in new_tracks:
                missed += 1
                if missed <= self.max_missed:
                    new_tracks[tid] = (tbox, missed)

        self.tracks = new_tracks

        # Retorna só (id -> box+roi)
        return assigned

    def reset_tracks(self, keep_states=False):
        self.tracks = {}
        if not keep_states:
            self.states = {}
        self.next_id = 1 if not keep_states else self.next_id


# -----------------------------
# Desenho
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

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,  # você roda por ROI
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    mediapipe_detector = vision.FaceLandmarker.create_from_options(options)

    dlib_detector = dlib.get_frontal_face_detector()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Erro abrindo vídeo: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    tracker = ReIDTracker(
        iou_thr=0.25,
        sim_thr=0.55,            # ajuste: 0.50-0.65 costuma ser um range bom
        max_missed=25,
        embed_every_n_frames=10, # (não usamos aqui por frame, mas fica pronto p/ evoluir)
        model_name="Facenet512"
    )

    prev_small = None
    frame_idx = 0
    mp_ts = 0  # timestamp monotônico do MediaPipe (1 por chamada)
    with tqdm(total=total_frames if total_frames > 0 else None, desc="Processando", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            retina_boxes = deepface_retina_boxes(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (160, 90))
            cut = False
            if prev_small is not None:
                cut = is_hard_cut(prev_small, small, thr=18.0)
            prev_small = small
            if cut:
                tracker.reset_tracks(keep_states=False)  # ou True, depende do que você quer

            dlib_faces = dlib_boxes(gray, dlib_detector)

            combined = merge_boxes(dlib_faces, retina_boxes, iou_thr=0.4)
            # Remove fallback de frames sem deteção de rosto para nao printar uma ROI da tela inteira para o mediapipe
            combined = filter_boxes(combined, width, height)

            # Tracking + ReID (retorna também ROI recortada pra evitar recortar 2x)
            assigned = tracker.update(combined, frame, frame_idx, width, height)

            # Iteração estável (ID ordenado)
            for face_id in sorted(assigned.keys()):
                x1, y1, x2, y2, score, (roi_x1, roi_y1, roi_x2, roi_y2), roi_bgr = assigned[face_id]

                roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                mp_roi = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)

                # timestamp monotônico (corrige o ValueError)
                mp_ts += 1
                result = mediapipe_detector.detect_for_video(mp_roi, mp_ts)

                state = tracker.states[face_id]

                if not result.face_landmarks:
                    cv2.putText(frame, f"ID {face_id}  eyes: ?", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    continue

                lms = result.face_landmarks[0]
                roi_w = roi_x2 - roi_x1
                roi_h = roi_y2 - roi_y1

                left_ear = ear_from_landmarks(lms, LEFT_EAR_IDX, roi_w, roi_h)
                right_ear = ear_from_landmarks(lms, RIGHT_EAR_IDX, roi_w, roi_h)
                ear_raw = (left_ear + right_ear) / 2.0
                ear = update_eyes_state(state, ear_raw)

                # Desenha pontos dos olhos (verde)
                draw_points_global(frame, lms, LEFT_EYE_IDX, roi_x1, roi_y1, roi_w, roi_h, (0, 255, 0), radius=2)
                draw_points_global(frame, lms, RIGHT_EYE_IDX, roi_x1, roi_y1, roi_w, roi_h, (0, 255, 0), radius=2)

                eyes_txt = "CLOSED" if state.eyes_closed else "OPEN"
                label = f"ID {face_id}  eyes: {eyes_txt}  EAR:{ear:.3f}"
                cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Debug/visual: desenha TODOS os boxes do merge (independente de tracking/mediapipe)
            for (mx1, my1, mx2, my2, mscore) in combined:
                cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 255, 0), 2)

            out.write(frame)
            frame_idx += 1
            pbar.update(1)

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    mediapipe_detector.close()

    print(f"Vídeo salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()