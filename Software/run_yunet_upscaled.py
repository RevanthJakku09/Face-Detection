import os
import time

import cv2

from evaluation import match_detections, precision_recall_f1
from wider_annotations import load_wider_annotations
from yunet_detector import load_yunet, detect_faces_yunet


DATASET_DIR = "datasets/WIDER_subset/images"

ANNOTATION_FILE = (
    "datasets/WIDER_subset/annotations/"
    "wider_face_val_bbx_gt.txt"
)

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

SCORE_THRESHOLD = 0.60
NMS_THRESHOLD = 0.30
TOP_K = 5000

UPSCALE_FACTOR = 2.0

IOU_THRESHOLD = 0.50


print()
print("=" * 80)
print("UPSCALED YUNET SMALL-FACE EXPERIMENT")
print("=" * 80)

print()
print("Configuration:")
print(f"Score threshold : {SCORE_THRESHOLD}")
print(f"NMS threshold   : {NMS_THRESHOLD}")
print(f"Upscale factor  : {UPSCALE_FACTOR}x")
print(f"IoU evaluation  : {IOU_THRESHOLD}")

print()
print("Loading WIDER annotations...")

ground_truth = load_wider_annotations(
    ANNOTATION_FILE
)

print(
    f"Annotated images: {len(ground_truth)}"
)

print()
print("Loading YuNet...")

detector = load_yunet(
    MODEL_PATH,
    score_threshold=SCORE_THRESHOLD,
    nms_threshold=NMS_THRESHOLD,
    top_k=TOP_K
)

print("YuNet loaded.")


total_tp = 0
total_fp = 0
total_fn = 0

total_detection_time = 0.0

processed = 0


for filename, gt_boxes in ground_truth.items():

    image_path = os.path.join(
        DATASET_DIR,
        filename
    )

    if not os.path.exists(image_path):
        continue

    image = cv2.imread(image_path)

    if image is None:
        continue

    original_h, original_w = image.shape[:2]

    # --------------------------------------------------------
    # Upscale image
    # --------------------------------------------------------

    upscaled_w = int(
        original_w * UPSCALE_FACTOR
    )

    upscaled_h = int(
        original_h * UPSCALE_FACTOR
    )

    upscaled = cv2.resize(
        image,
        (upscaled_w, upscaled_h),
        interpolation=cv2.INTER_LINEAR
    )

    # --------------------------------------------------------
    # Run YuNet on upscaled image
    # --------------------------------------------------------

    start = time.perf_counter()

    detections = detect_faces_yunet(
        upscaled,
        detector
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    total_detection_time += elapsed

    # --------------------------------------------------------
    # Convert detections back to original coordinates
    # --------------------------------------------------------

    predictions = []

    for x, y, w, h in detections:

        predictions.append(
            (
                round(x / UPSCALE_FACTOR),
                round(y / UPSCALE_FACTOR),
                round(w / UPSCALE_FACTOR),
                round(h / UPSCALE_FACTOR)
            )
        )

    # --------------------------------------------------------
    # Evaluate against original GT
    # --------------------------------------------------------

    result = match_detections(
        predictions,
        gt_boxes,
        iou_threshold=IOU_THRESHOLD
    )

    total_tp += result[
        "true_positives"
    ]

    total_fp += result[
        "false_positives"
    ]

    total_fn += result[
        "false_negatives"
    ]

    processed += 1

    if processed % 50 == 0:

        print(
            f"Processed {processed} images..."
        )


# ============================================================
# RESULTS
# ============================================================

metrics = precision_recall_f1(
    total_tp,
    total_fp,
    total_fn
)

print()
print("=" * 80)
print("UPSCALED YUNET RESULTS")
print("=" * 80)

print(
    f"Images processed : {processed}"
)

print(
    f"TP               : {total_tp}"
)

print(
    f"FP               : {total_fp}"
)

print(
    f"FN               : {total_fn}"
)

print(
    f"Precision        : "
    f"{metrics['precision'] * 100:.2f}%"
)

print(
    f"Recall           : "
    f"{metrics['recall'] * 100:.2f}%"
)

print(
    f"F1-score         : "
    f"{metrics['f1_score'] * 100:.2f}%"
)

if processed > 0:

    mean_latency = (
        total_detection_time
        / processed
    )

    fps = (
        processed
        / total_detection_time
    )

    print(
        f"Mean latency     : "
        f"{mean_latency * 1000:.2f} ms"
    )

    print(
        f"FPS              : "
        f"{fps:.2f}"
    )

print("=" * 80)