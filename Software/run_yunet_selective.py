import os
import time

import cv2

from evaluation import match_detections, precision_recall_f1
from wider_annotations import load_wider_annotations
from yunet_detector import load_yunet, detect_faces_yunet


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "datasets/WIDER_subset/images"

ANNOTATION_FILE = (
    "datasets/WIDER_subset/annotations/"
    "wider_face_val_bbx_gt.txt"
)

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

# Same detector settings as our frozen baseline
SCORE_THRESHOLD = 0.60
NMS_THRESHOLD = 0.30
TOP_K = 5000

# Selective second-pass settings
UPSCALE_FACTOR = 2.0

# A quadrant is considered sparse if it contains
# this number of first-pass detections or fewer.
SPARSE_DETECTION_LIMIT = 2

IOU_THRESHOLD = 0.50

# Duplicate removal after combining detections
MERGE_IOU_THRESHOLD = 0.50


# ============================================================
# QUADRANT CREATION
# ============================================================

def create_quadrants(image):

    height, width = image.shape[:2]

    mid_x = width // 2
    mid_y = height // 2

    quadrants = [
        (
            image[0:mid_y, 0:mid_x],
            0,
            0,
        ),
        (
            image[0:mid_y, mid_x:width],
            mid_x,
            0,
        ),
        (
            image[mid_y:height, 0:mid_x],
            0,
            mid_y,
        ),
        (
            image[mid_y:height, mid_x:width],
            mid_x,
            mid_y,
        ),
    ]

    return quadrants


# ============================================================
# IoU FOR MERGING TILE DETECTIONS
# ============================================================

def calculate_iou(box_a, box_b):

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2 = ax + aw
    ay2 = ay + ah

    bx2 = bx + bw
    by2 = by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

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


def merge_boxes(boxes):

    if not boxes:
        return []

    boxes = sorted(
        boxes,
        key=lambda box: box[2] * box[3],
        reverse=True
    )

    kept = []

    for box in boxes:

        duplicate = False

        for existing in kept:

            if calculate_iou(
                box,
                existing
            ) >= MERGE_IOU_THRESHOLD:

                duplicate = True
                break

        if not duplicate:
            kept.append(box)

    return kept


# ============================================================
# UPSCALE A QUADRANT
# ============================================================

def upscale_quadrant(
    quadrant,
    offset_x,
    offset_y,
):

    original_h, original_w = quadrant.shape[:2]

    new_w = int(
        original_w * UPSCALE_FACTOR
    )

    new_h = int(
        original_h * UPSCALE_FACTOR
    )

    upscaled = cv2.resize(
        quadrant,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    return upscaled, offset_x, offset_y


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 80)
print("SELECTIVE UPSCALING YUNET EXPERIMENT")
print("=" * 80)

print()
print("Configuration:")
print(f"Score threshold       : {SCORE_THRESHOLD}")
print(f"NMS threshold         : {NMS_THRESHOLD}")
print(f"Upscale factor        : {UPSCALE_FACTOR}x")
print(f"Sparse detection limit: {SPARSE_DETECTION_LIMIT}")
print(f"IoU evaluation        : {IOU_THRESHOLD}")

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


# ============================================================
# STATISTICS
# ============================================================

total_tp = 0
total_fp = 0
total_fn = 0

total_detection_time = 0.0

processed = 0

total_quadrants = 0
upscaled_quadrants = 0


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

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # FIRST PASS — NORMAL FULL IMAGE
    # --------------------------------------------------------

    first_pass = detect_faces_yunet(
        image,
        detector
    )

    final_boxes = list(first_pass)

    # --------------------------------------------------------
    # SECOND PASS — SPARSE QUADRANTS ONLY
    # --------------------------------------------------------

    quadrants = create_quadrants(image)

    total_quadrants += len(quadrants)

    for quadrant, offset_x, offset_y in quadrants:

        # ----------------------------------------------------
        # Determine how many first-pass detections
        # belong to this quadrant.
        # ----------------------------------------------------

        q_height, q_width = quadrant.shape[:2]

        detection_count = 0

        for x, y, w, h in first_pass:

            center_x = x + (w / 2)
            center_y = y + (h / 2)

            if (
                offset_x <= center_x < offset_x + q_width
                and
                offset_y <= center_y < offset_y + q_height
            ):
                detection_count += 1

        # ----------------------------------------------------
        # Only upscale sparse quadrants
        # ----------------------------------------------------

        if detection_count > SPARSE_DETECTION_LIMIT:
            continue

        upscaled_quadrants += 1

        upscaled, qx, qy = upscale_quadrant(
            quadrant,
            offset_x,
            offset_y
        )

        # ----------------------------------------------------
        # Run YuNet on upscaled quadrant
        # ----------------------------------------------------

        second_pass = detect_faces_yunet(
            upscaled,
            detector
        )

        # ----------------------------------------------------
        # Convert back to original coordinates
        # ----------------------------------------------------

        for x, y, w, h in second_pass:

            original_x = (
                qx
                + x / UPSCALE_FACTOR
            )

            original_y = (
                qy
                + y / UPSCALE_FACTOR
            )

            original_w = (
                w / UPSCALE_FACTOR
            )

            original_h = (
                h / UPSCALE_FACTOR
            )

            final_boxes.append(
                (
                    round(original_x),
                    round(original_y),
                    round(original_w),
                    round(original_h),
                )
            )

    # --------------------------------------------------------
    # Merge duplicates
    # --------------------------------------------------------

    final_boxes = merge_boxes(
        final_boxes
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    total_detection_time += elapsed

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    result = match_detections(
        final_boxes,
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
# FINAL RESULTS
# ============================================================

metrics = precision_recall_f1(
    total_tp,
    total_fp,
    total_fn
)

print()
print("=" * 80)
print("SELECTIVE UPSCALING RESULTS")
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

print()
print(
    f"Total quadrants  : {total_quadrants}"
)

print(
    f"Upscaled quadrants: {upscaled_quadrants}"
)

if total_quadrants > 0:

    percentage = (
        upscaled_quadrants
        / total_quadrants
        * 100
    )

    print(
        f"Upscaled fraction : "
        f"{percentage:.2f}%"
    )

print("=" * 80)