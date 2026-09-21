import os
from collections import Counter

import cv2

from wider_annotations import load_wider_annotations
from yunet_detector import load_yunet, detect_faces_yunet
from preprocessing import resize_image
from evaluation import iou


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "datasets/WIDER_subset/images"

ANNOTATION_FILE = (
    "datasets/WIDER_subset/annotations/"
    "wider_face_val_bbx_gt.txt"
)

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

SCORE_THRESHOLD = 0.60
NMS_THRESHOLD = 0.30
TOP_K = 5000
MAX_DIMENSION = 1200
IOU_THRESHOLD = 0.50


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 80)
print("PRECISE FALSE-NEGATIVE FACE-SIZE ANALYSIS")
print("=" * 80)

print("\nLoading WIDER annotations...")

ground_truth = load_wider_annotations(
    ANNOTATION_FILE
)

print(
    f"Annotated images: {len(ground_truth)}"
)

print("\nLoading YuNet...")

detector = load_yunet(
    MODEL_PATH,
    score_threshold=SCORE_THRESHOLD,
    nms_threshold=NMS_THRESHOLD,
    top_k=TOP_K
)

print("YuNet loaded.")


# ============================================================
# STORAGE
# ============================================================

fn_widths = []
fn_heights = []

total_gt = 0
total_tp = 0
total_fp = 0
total_fn = 0

processed = 0


# ============================================================
# PROCESS IMAGES
# ============================================================

for filename, gt_boxes_original in ground_truth.items():

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
    # Calculate resize scale
    # --------------------------------------------------------

    height, width = image.shape[:2]

    largest = max(
        height,
        width
    )

    if largest <= MAX_DIMENSION:

        scale = 1.0

    else:

        scale = (
            MAX_DIMENSION /
            float(largest)
        )

    # --------------------------------------------------------
    # Resize image using project's function
    # --------------------------------------------------------

    resized_image = resize_image(
        image,
        max_dimension=MAX_DIMENSION
    )

    # --------------------------------------------------------
    # Scale ground-truth boxes
    # --------------------------------------------------------

    gt_boxes = []

    for box in gt_boxes_original:

        x, y, w, h = box

        gt_boxes.append(
            (
                x * scale,
                y * scale,
                w * scale,
                h * scale
            )
        )

    # --------------------------------------------------------
    # Run YuNet
    # --------------------------------------------------------

    detections = detect_faces_yunet(
        resized_image,
        detector
    )

    pred_boxes = []

    for detection in detections:

        x, y, w, h = detection[:4]

        pred_boxes.append(
            (
                float(x),
                float(y),
                float(w),
                float(h)
            )
        )

    # ========================================================
    # SAME GREEDY IoU MATCHING LOGIC AS evaluation.py
    # ========================================================

    matches = []

    for pred_index, prediction in enumerate(
        pred_boxes
    ):

        for gt_index, ground_truth_box in enumerate(
            gt_boxes
        ):

            score = iou(
                prediction,
                ground_truth_box
            )

            if score >= IOU_THRESHOLD:

                matches.append(
                    (
                        score,
                        pred_index,
                        gt_index
                    )
                )

    # Highest IoU first
    matches.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    used_predictions = set()
    used_ground_truth = set()

    true_positives = 0

    for score, pred_index, gt_index in matches:

        if pred_index in used_predictions:
            continue

        if gt_index in used_ground_truth:
            continue

        used_predictions.add(
            pred_index
        )

        used_ground_truth.add(
            gt_index
        )

        true_positives += 1

    # --------------------------------------------------------
    # Calculate FP / FN
    # --------------------------------------------------------

    false_positives = (
        len(pred_boxes)
        - true_positives
    )

    false_negatives = (
        len(gt_boxes)
        - true_positives
    )

    total_gt += len(gt_boxes)
    total_tp += true_positives
    total_fp += false_positives
    total_fn += false_negatives

    # --------------------------------------------------------
    # ACTUAL FN BOXES
    #
    # Any GT index that was not matched is an FN.
    # --------------------------------------------------------

    unmatched_gt = (
        set(range(len(gt_boxes)))
        - used_ground_truth
    )

    for gt_index in unmatched_gt:

        x, y, w, h = gt_boxes[gt_index]

        fn_widths.append(w)
        fn_heights.append(h)

    processed += 1

    if processed % 50 == 0:

        print(
            f"Processed {processed} images..."
        )


# ============================================================
# OVERALL CHECK
# ============================================================

print()
print("=" * 80)
print("OVERALL CHECK")
print("=" * 80)

print(
    f"Images processed : {processed}"
)

print(
    f"Ground-truth     : {total_gt}"
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


# ============================================================
# FN SIZE DISTRIBUTION
# ============================================================

print()
print("=" * 80)
print("ACTUAL FALSE-NEGATIVE FACE SIZE DISTRIBUTION")
print("=" * 80)


bins = [
    ("<10 px", 0, 10),
    ("10-19 px", 10, 20),
    ("20-39 px", 20, 40),
    ("40-79 px", 40, 80),
    ("80+ px", 80, float("inf")),
]

counts = Counter()

for width in fn_widths:

    for name, low, high in bins:

        if low <= width < high:

            counts[name] += 1

            break


for name, low, high in bins:

    count = counts[name]

    percentage = (
        count / len(fn_widths) * 100
        if fn_widths
        else 0
    )

    print(
        f"{name:10s} : "
        f"{count:5d} faces "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# KEY RESULTS
# ============================================================

print()
print("=" * 80)
print("KEY RESULT")
print("=" * 80)

if fn_widths:

    smaller_20 = sum(
        1
        for width in fn_widths
        if width < 20
    )

    smaller_40 = sum(
        1
        for width in fn_widths
        if width < 40
    )

    average_width = (
        sum(fn_widths)
        / len(fn_widths)
    )

    average_height = (
        sum(fn_heights)
        / len(fn_heights)
    )

    print(
        f"FN faces smaller than 20 px : "
        f"{smaller_20} / {len(fn_widths)} "
        f"({smaller_20 / len(fn_widths) * 100:.2f}%)"
    )

    print(
        f"FN faces smaller than 40 px : "
        f"{smaller_40} / {len(fn_widths)} "
        f"({smaller_40 / len(fn_widths) * 100:.2f}%)"
    )

    print(
        f"Average FN width : "
        f"{average_width:.2f} px"
    )

    print(
        f"Average FN height: "
        f"{average_height:.2f} px"
    )

print("=" * 80)