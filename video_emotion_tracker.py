import argparse
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import tensorflow as tf


DEFAULT_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO_PATH = os.path.join(PROJECT_DIR, "video_demo.mp4")
DEFAULT_MODEL_DIR = os.path.join(PROJECT_DIR, "model")


@dataclass
class FaceTrack:
    track_id: int
    bbox: Tuple[int, int, int, int]
    missed_frames: int = 0
    prob_history: deque = field(default_factory=lambda: deque(maxlen=8))

    def update(self, bbox: Tuple[int, int, int, int], probs: Optional[np.ndarray]) -> None:
        self.bbox = bbox
        self.missed_frames = 0
        if probs is not None:
            self.prob_history.append(probs)

    def mark_missed(self) -> None:
        self.missed_frames += 1

    def smoothed_probs(self) -> Optional[np.ndarray]:
        if not self.prob_history:
            return None
        return np.mean(np.stack(self.prob_history, axis=0), axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play an MP4, track faces, and overlay emotion labels from a saved Keras model."
    )
    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO_PATH,
        help=f"Path to the input .mp4 video (default: {DEFAULT_VIDEO_PATH})",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Directory containing model.json and model.weights.h5 (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument("--output", default="", help="Optional output video path (.mp4) for annotated frames")
    parser.add_argument("--max-missed", type=int, default=12, help="Max frames a track can be missing before removal")
    parser.add_argument("--iou-threshold", type=float, default=0.30, help="IoU threshold for track/detection matching")
    parser.add_argument("--min-face-size", type=int, default=40, help="Minimum detected face size in pixels")
    return parser.parse_args()


def load_model(model_dir: str) -> tf.keras.Model:
    model_json_path = os.path.join(model_dir, "model.json")
    weights_path = os.path.join(model_dir, "model.weights.h5")

    if not os.path.exists(model_json_path):
        raise FileNotFoundError(f"Could not find model architecture: {model_json_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Could not find model weights: {weights_path}")

    with open(model_json_path, "r", encoding="utf-8") as fp:
        model_json = fp.read()

    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(weights_path)
    return model


def get_input_shape(model: tf.keras.Model) -> Tuple[int, int, int]:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if input_shape is None or len(input_shape) != 4:
        raise ValueError(f"Unexpected model input shape: {input_shape}")
    _, h, w, c = input_shape
    if h is None or w is None or c is None:
        raise ValueError(f"Model input has undefined dimensions: {input_shape}")
    return int(h), int(w), int(c)


def get_num_classes(model: tf.keras.Model) -> int:
    output_shape = model.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    if output_shape is None or len(output_shape) < 2 or output_shape[-1] is None:
        raise ValueError(f"Unexpected model output shape: {output_shape}")
    return int(output_shape[-1])


def get_labels(num_classes: int) -> List[str]:
    if num_classes == len(DEFAULT_LABELS):
        return DEFAULT_LABELS.copy()
    return [f"class_{idx}" for idx in range(num_classes)]


def preprocess_face(face_bgr: np.ndarray, input_shape: Tuple[int, int, int]) -> np.ndarray:
    in_h, in_w, in_c = input_shape

    if in_c == 1:
        face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (in_w, in_h), interpolation=cv2.INTER_AREA)
        face = face.astype(np.float32) / 255.0
        face = np.expand_dims(face, axis=-1)
    elif in_c == 3:
        face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (in_w, in_h), interpolation=cv2.INTER_AREA)
        face = face.astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unsupported input channels: {in_c}")

    return np.expand_dims(face, axis=0)


def detect_faces(
    detector: cv2.CascadeClassifier,
    gray_frame: np.ndarray,
    min_face_size: int,
) -> List[Tuple[int, int, int, int]]:
    boxes = detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_face_size, min_face_size),
    )
    detections: List[Tuple[int, int, int, int]] = []
    for box in boxes:
        x, y, w, h = box
        detections.append((int(x), int(y), int(w), int(h)))
    return detections


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    a_x2, a_y2 = ax + aw, ay + ah
    b_x2, b_y2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def match_tracks(
    tracks: Dict[int, FaceTrack],
    detections: Sequence[Tuple[int, int, int, int]],
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    track_ids = list(tracks.keys())
    if not track_ids or not detections:
        return [], track_ids, list(range(len(detections)))

    iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
    for row, track_id in enumerate(track_ids):
        for col, det in enumerate(detections):
            iou_matrix[row, col] = bbox_iou(tracks[track_id].bbox, det)

    matches: List[Tuple[int, int]] = []
    matched_tracks = set()
    matched_dets = set()

    while True:
        flat_index = int(np.argmax(iou_matrix))
        row, col = np.unravel_index(flat_index, iou_matrix.shape)
        row = int(row)
        col = int(col)
        best_iou = float(iou_matrix[row, col])

        if best_iou < iou_threshold:
            break

        track_id = track_ids[row]
        if track_id in matched_tracks or col in matched_dets:
            iou_matrix[row, col] = -1.0
            continue

        matches.append((track_id, col))
        matched_tracks.add(track_id)
        matched_dets.add(col)

        iou_matrix[row, :] = -1.0
        iou_matrix[:, col] = -1.0

    unmatched_tracks = [track_id for track_id in track_ids if track_id not in matched_tracks]
    unmatched_dets = [idx for idx in range(len(detections)) if idx not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


def clamp_bbox(
    bbox: Tuple[int, int, int, int], frame_w: int, frame_h: int
) -> Optional[Tuple[int, int, int, int]]:
    x, y, w, h = bbox
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_w, x + w)
    y2 = min(frame_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def run() -> None:
    args = parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(
            f"Video not found at {args.video}. Put video_demo.mp4 in the project root or pass --video <path>."
        )

    model = load_model(args.model_dir)
    input_shape = get_input_shape(model)
    labels = get_labels(get_num_classes(model))

    cv2_data = getattr(cv2, "data", None)
    if cv2_data is None or not hasattr(cv2_data, "haarcascades"):
        raise RuntimeError("OpenCV installation does not expose Haar cascades path via cv2.data.haarcascades")

    cascade_path = os.path.join(str(cv2_data.haarcascades), "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Failed to load OpenCV Haar cascade from {cascade_path}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    wait_ms = max(1, int(1000 / fps))

    writer = None
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        video_writer_fourcc = getattr(cv2, "VideoWriter_fourcc", None)
        if video_writer_fourcc is None:
            raise RuntimeError("OpenCV installation does not expose VideoWriter_fourcc")
        fourcc = video_writer_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_w, frame_h))

    tracks: Dict[int, FaceTrack] = {}
    next_track_id = 1

    last_time = time.time()
    window_name = "Emotion Tracking"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detect_faces(detector, gray, args.min_face_size)

        det_probs: List[Optional[np.ndarray]] = []
        for det in detections:
            clamped = clamp_bbox(det, frame_w, frame_h)
            if clamped is None:
                det_probs.append(None)
                continue

            x, y, w, h = clamped
            face_crop = frame[y : y + h, x : x + w]
            if face_crop.size == 0:
                det_probs.append(None)
                continue

            face_input = preprocess_face(face_crop, input_shape)
            probs = model.predict(face_input, verbose=0)[0]
            det_probs.append(probs)

        matches, unmatched_tracks, unmatched_dets = match_tracks(
            tracks=tracks,
            detections=detections,
            iou_threshold=args.iou_threshold,
        )

        for track_id, det_idx in matches:
            tracks[track_id].update(detections[det_idx], det_probs[det_idx])

        for track_id in unmatched_tracks:
            tracks[track_id].mark_missed()

        stale_track_ids = [track_id for track_id, tr in tracks.items() if tr.missed_frames > args.max_missed]
        for track_id in stale_track_ids:
            del tracks[track_id]

        for det_idx in unmatched_dets:
            tracks[next_track_id] = FaceTrack(track_id=next_track_id, bbox=detections[det_idx])
            tracks[next_track_id].update(detections[det_idx], det_probs[det_idx])
            next_track_id += 1

        now = time.time()
        fps_est = 1.0 / max(1e-6, now - last_time)
        last_time = now

        for tr in tracks.values():
            x, y, w, h = tr.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 220, 70), 2)

            probs = tr.smoothed_probs()
            if probs is not None:
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
                label = f"ID {tr.track_id} | {labels[pred_idx]} {confidence:.2f}"
            else:
                label = f"ID {tr.track_id} | unknown"

            cv2.putText(
                frame,
                label,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (20, 20, 20),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                label,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"FPS: {fps_est:.1f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if writer is not None:
            writer.write(frame)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()