"""
evaluation.py
-------------
Face detection evaluation using IoU matching.
"""


def box_to_corners(box):

    x, y, w, h = box

    return (
        x,
        y,
        x + w,
        y + h,
    )


def iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_to_corners(
        box_a
    )

    bx1, by1, bx2, by2 = box_to_corners(
        box_b
    )

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0,
        ix2 - ix1
    )

    ih = max(
        0,
        iy2 - iy1
    )

    intersection = iw * ih

    area_a = (
        max(0, ax2 - ax1) *
        max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1) *
        max(0, by2 - by1)
    )

    union = (
        area_a +
        area_b -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def match_detections(
    predicted_boxes,
    ground_truth_boxes,
    iou_threshold=0.5,
):
    """
    Match predictions to ground-truth boxes.

    Each ground-truth face can be matched only once.
    """

    matches = []

    for pred_index, prediction in enumerate(
        predicted_boxes
    ):

        for gt_index, ground_truth in enumerate(
            ground_truth_boxes
        ):

            score = iou(
                prediction,
                ground_truth
            )

            if score >= iou_threshold:

                matches.append(
                    (
                        score,
                        pred_index,
                        gt_index,
                    )
                )

    # Best matches first.
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

    false_positives = (
        len(predicted_boxes)
        - true_positives
    )

    false_negatives = (
        len(ground_truth_boxes)
        - true_positives
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def precision_recall_f1(
    true_positives,
    false_positives,
    false_negatives,
):

    precision = (
        true_positives /
        (
            true_positives +
            false_positives
        )
        if (
            true_positives +
            false_positives
        ) > 0
        else 0.0
    )

    recall = (
        true_positives /
        (
            true_positives +
            false_negatives
        )
        if (
            true_positives +
            false_negatives
        ) > 0
        else 0.0
    )

    f1 = (
        2 *
        precision *
        recall /
        (precision + recall)
        if (
            precision +
            recall
        ) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def evaluate_dataset(
    predictions,
    ground_truth,
    iou_threshold=0.5,
):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_image = {}

    for filename, gt_boxes in ground_truth.items():

        pred_boxes = predictions.get(
            filename,
            []
        )

        result = match_detections(
            pred_boxes,
            gt_boxes,
            iou_threshold=iou_threshold,
        )

        per_image[filename] = result

        total_tp += result[
            "true_positives"
        ]

        total_fp += result[
            "false_positives"
        ]

        total_fn += result[
            "false_negatives"
        ]

    metrics = precision_recall_f1(
        total_tp,
        total_fp,
        total_fn,
    )

    metrics.update({
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "detection_rate": metrics[
            "recall"
        ],
        "per_image": per_image,
    })

    return metrics