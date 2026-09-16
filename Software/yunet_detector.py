"""YuNet lightweight face detector using OpenCV DNN."""

import os
import cv2


DEFAULT_MODEL = os.path.join(
    "models", "face_detection_yunet_2023mar.onnx"
)


def load_yunet(model_path=DEFAULT_MODEL, score_threshold=0.6,
               nms_threshold=0.3, top_k=5000):
    """Create an OpenCV YuNet FaceDetectorYN instance."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"YuNet model not found: {model_path}\n"
            "Download face_detection_yunet_2023mar.onnx from the "
            "official OpenCV Zoo model repository and place it in models/."
        )

    detector = cv2.FaceDetectorYN.create(
        model_path,
        "",
        (320, 320),
        score_threshold,
        nms_threshold,
        top_k,
    )
    return detector


def detect_faces_yunet(image, detector):
    """Detect faces and return [(x, y, w, h), ...]."""
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, detections = detector.detect(image)

    if detections is None:
        return []

    boxes = []
    for detection in detections:
        x, y, w, h = detection[:4]
        boxes.append((
            max(0, int(round(x))),
            max(0, int(round(y))),
            max(1, int(round(w))),
            max(1, int(round(h))),
        ))

    return boxes


def draw_yunet_boxes(image, boxes, thickness=2):
    output = image.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(
            output, (x, y), (x + w, y + h), (0, 255, 0), thickness
        )
    return output
