import os
import time

import cv2

from wider_annotations import load_wider_annotations
from yunet_detector import load_yunet, detect_faces_yunet
from evaluation import match_detections, precision_recall_f1


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "datasets/WIDER_subset/images"

ANNOTATION_FILE = (
    "datasets/WIDER_subset/annotations/"
    "wider_face_val_bbx_gt.txt"
)

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

# Keep the same YuNet parameters as the baseline
SCORE_THRESHOLD = 0.60
NMS_THRESHOLD = 0.30
TOP_K = 5000

# Tile configuration
TILES_X = 2
TILES_Y = 2
OVERLAP = 0.20

# Evaluation
IOU_THRESHOLD = 0.50


# ============================================================
# CREATE TILES
# ============================================================

def create_tiles(image, tiles_x=2, tiles_y=2, overlap=0.20):

    height, width = image.shape[:2]

    tile_width = int(
        width / (
            tiles_x - overlap * (tiles_x - 1)
        )
    )

    tile_height = int(
        height / (
            tiles_y - overlap * (tiles_y - 1)
        )
    )

    tile_width = min(tile_width, width)
    tile_height = min(tile_height, height)

    step_x = int(
        tile_width * (1 - overlap)
    )

    step_y = int(
        tile_height * (1 - overlap)
    )

    tiles = []

    for row in range(tiles_y):

        for col in range(tiles_x):

            x1 = col * step_x
            y1 = row * step_y

            x2 = min(
                x1 + tile_width,
                width
            )

            y2 = min(
                y1 + tile_height,
                height
            )

            # Ensure last tiles reach the image boundary
            if col == tiles_x - 1:
                x1 = max(
                    0,
                    width - tile_width
                )
                x2 = width

            if row == tiles_y - 1:
                y1 = max(
                    0,
                    height - tile_height
                )
                y2 = height

            tile = image[
                y1:y2,
                x1:x2
            ]

            tiles.append(
                (
                    tile,
                    x1,
                    y1
                )
            )

    return tiles


# ============================================================
# MERGE OVERLAPPING DETECTIONS
# ============================================================

def calculate_iou(box_a, box_b):

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2 = ax + aw
    ay2 = ay + ah

    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(
        0,
        inter_x2 - inter_x1
    )

    inter_h = max(
        0,
        inter_y2 - inter_y1
    )

    intersection = (
        inter_w * inter_h
    )

    area_a = aw * ah
    area_b = bw * bh

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def merge_boxes(boxes, iou_threshold=0.50):

    if not boxes:
        return []

    # Remove duplicates created by overlapping tiles
    boxes = sorted(
        boxes,
        key=lambda b: b[2] * b[3],
        reverse=True
    )

    kept = []

    for box in boxes:

        duplicate = False

        for existing in kept:

            if calculate_iou(
                box,
                existing
            ) >= iou_threshold:

                duplicate = True
                break

        if not duplicate:
            kept.append(box)

    return kept


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 80)
print("TILED YUNET SMALL-FACE EXPERIMENT")
print("=" * 80)

print()
print("Configuration:")
print(f"Score threshold : {SCORE_THRESHOLD}")
print(f"NMS threshold   : {NMS_THRESHOLD}")
print(f"Tiles           : {TILES_X} x {TILES_Y}")
print(f"Overlap         : {OVERLAP * 100:.0f}%")
print(f"IoU evaluation  : {IOU_THRESHOLD}")

# ------------------------------------------------------------
# Load annotations
# ------------------------------------------------------------

print()
print("Loading WIDER annotations...")

ground_truth = load_wider_annotations(
    ANNOTATION_FILE
)

print(
    f"Annotated images: {len(ground_truth)}"
)

# ------------------------------------------------------------
# Load YuNet
# ------------------------------------------------------------

print()
print("Loading YuNet...")

detector = load_yunet(
    MODEL_PATH,
    score_threshold=SCORE_THRESHOLD,
    nms_threshold=NMS_THRESHOLD,
    top_k=TOP_K
)

print("YuNet loaded.")


# ============================================================
# EVALUATION STORAGE
# ============================================================

total_tp = 0
total_fp = 0
total_fn = 0

total_detection_time = 0.0

processed = 0


# ============================================================
# PROCESS DATASET
# ============================================================

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

    # --------------------------------------------------------
    # Create overlapping tiles
    # --------------------------------------------------------

    tiles = create_tiles(
        image,
        tiles_x=TILES_X,
        tiles_y=TILES_Y,
        overlap=OVERLAP
    )

    image_predictions = []

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Run YuNet on every tile
    # --------------------------------------------------------

    for tile, offset_x, offset_y in tiles:

        detections = detect_faces_yunet(
            tile,
            detector
        )

        # Convert tile coordinates
        # back to original-image coordinates
        for detection in detections:

            x, y, w, h = detection

            original_x = x + offset_x
            original_y = y + offset_y

            image_predictions.append(
                (
                    original_x,
                    original_y,
                    w,
                    h
                )
            )

    # --------------------------------------------------------
    # Merge detections from overlapping tiles
    # --------------------------------------------------------

    image_predictions = merge_boxes(
        image_predictions,
        iou_threshold=0.50
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    total_detection_time += elapsed

    # --------------------------------------------------------
    # Evaluate against original GT boxes
    # --------------------------------------------------------

    result = match_detections(
        image_predictions,
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
# FINAL METRICS
# ============================================================

metrics = precision_recall_f1(
    total_tp,
    total_fp,
    total_fn
)

print()
print("=" * 80)
print("TILED YUNET RESULTS")
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

if total_detection_time > 0:

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