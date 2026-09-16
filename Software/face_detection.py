"""
face_detection.py
------------------
High-recall Haar Cascade face detector with vote-based merging.

Uses multiple human-face Haar classifiers, each run at the original
scale and an upscaled scale (6 passes total), then merges detections
using an ensemble "voting" strategy: a box only survives if enough
independent passes agree it's a face.

This voting step is the key lever for balancing precision vs. recall:
  - min_votes=1 behaves like the old "keep everything non-overlapping"
    approach -> maximum recall, weak precision (lots of single-cascade
    false positives survive).
  - min_votes=2 or 3 requires multiple cascades/scales to agree ->
    false positives (which are usually cascade-specific and don't
    reproduce across passes) get filtered out, while real faces
    (which tend to be found by several passes) survive.

Tune `min_votes` for your dataset using main.py's --auto-tune mode.
"""

import cv2
import os


# Human-face Haar classifiers available in OpenCV
CASCADE_NAMES = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt.xml",
    "haarcascade_frontalface_alt2.xml",
]


def load_cascade(cascade_name):
    """Load one Haar Cascade from OpenCV."""
    cascade_path = os.path.join(cv2.data.haarcascades, cascade_name)
    classifier = cv2.CascadeClassifier(cascade_path)

    if classifier.empty():
        raise IOError(f"Failed to load Haar Cascade:\n{cascade_path}")

    return classifier


def load_all_cascades():
    """Load all selected face cascades."""
    return {name: load_cascade(name) for name in CASCADE_NAMES}


def detect_single(gray_image, classifier, scale_factor=1.05, min_neighbors=4, min_size=(10, 10)):
    """Run one Haar classifier."""
    faces = classifier.detectMultiScale(
        gray_image,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    return [tuple(map(int, face)) for face in faces]


def box_iou(box_a, box_b):
    """Calculate IoU between two (x, y, w, h) bounding boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def cluster_boxes(boxes, iou_threshold=0.4):
    """Group boxes into clusters of mutually-overlapping detections.

    Uses simple union-find: any two boxes with IoU >= iou_threshold are
    linked into the same cluster. Each cluster represents detections
    that likely refer to the same underlying face.

    Args:
        boxes (list): List of (x, y, w, h) boxes from all detection passes.
        iou_threshold (float): Minimum IoU for two boxes to be linked.

    Returns:
        list[list]: List of clusters, each a list of boxes.
    """
    n = len(boxes)
    if n == 0:
        return []

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if box_iou(boxes[i], boxes[j]) >= iou_threshold:
                union(i, j)

    clusters = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(boxes[i])

    return list(clusters.values())


def merge_detections_by_vote(boxes, iou_threshold=0.4, min_votes=2):
    """Merge overlapping detections and keep only clusters with enough votes.

    This is the precision/recall control knob: a cluster of overlapping
    boxes is only kept if it was detected `min_votes` or more times across
    all cascades/scales. The representative box for a surviving cluster is
    the average of its member boxes.

    Args:
        boxes (list): All (x, y, w, h) boxes from every detection pass.
        iou_threshold (float): IoU threshold used to cluster boxes together.
        min_votes (int): Minimum number of agreeing detections required to
            keep a face. Higher = fewer false positives, but risks dropping
            real faces only one or two passes caught. Lower = more recall,
            more false positives.

    Returns:
        list[tuple[int, int, int, int]]: Final merged (x, y, w, h) boxes.
    """
    if not boxes:
        return []

    clusters = cluster_boxes(boxes, iou_threshold=iou_threshold)
    merged = []

    for cluster in clusters:
        votes = len(cluster)
        if votes < min_votes:
            continue

        xs = [b[0] for b in cluster]
        ys = [b[1] for b in cluster]
        ws = [b[2] for b in cluster]
        hs = [b[3] for b in cluster]
        n = float(votes)

        representative = (
            int(round(sum(xs) / n)),
            int(round(sum(ys) / n)),
            int(round(sum(ws) / n)),
            int(round(sum(hs) / n)),
        )
        merged.append(representative)

    return merged


def detect_faces_high_recall(
    gray_image,
    cascades,
    scale_factor=1.05,
    min_neighbors=4,
    min_size=(10, 10),
    use_upscale=True,
    upscale_factor=1.5,
    merge_iou_threshold=0.40,
    min_votes=2,
):
    """
    Multi-cascade, multi-scale face detection with vote-based merging.

    Detection passes (up to 6 total, one per cascade x scale):
      1-3. Original image with each of the 3 Haar cascades
      4-6. Upscaled image with each of the 3 Haar cascades

    All raw detections are pooled, clustered by overlap, and a box is
    only kept if `min_votes` or more passes agree on it.

    Args:
        gray_image (numpy.ndarray): Preprocessed grayscale image.
        cascades (dict): name -> cv2.CascadeClassifier, from load_all_cascades().
        scale_factor (float): Haar Cascade scaleFactor for each pass.
        min_neighbors (int): Haar Cascade minNeighbors for each pass.
        min_size (tuple): Minimum face size (w, h) in pixels for each pass.
        use_upscale (bool): Whether to also run detection on an upscaled copy
            (helps catch small faces the original scale would miss).
        upscale_factor (float): How much to upscale the image for the extra pass.
        merge_iou_threshold (float): IoU threshold for clustering detections
            together as "the same face".
        min_votes (int): Minimum number of agreeing passes to keep a detection.
            This is the main precision/recall tuning knob — see main.py's
            --auto-tune mode to search for a good value on your dataset.

    Returns:
        list[tuple[int, int, int, int]]: Final (x, y, w, h) boxes.
    """
    all_boxes = []

    for classifier in cascades.values():
        boxes = detect_single(
            gray_image, classifier,
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
            min_size=min_size,
        )
        all_boxes.extend(boxes)

    if use_upscale:
        upscaled = cv2.resize(
            gray_image, None,
            fx=upscale_factor, fy=upscale_factor,
            interpolation=cv2.INTER_CUBIC,
        )

        for classifier in cascades.values():
            boxes = detect_single(
                upscaled, classifier,
                scale_factor=scale_factor,
                min_neighbors=min_neighbors,
                min_size=min_size,
            )
            for x, y, w, h in boxes:
                all_boxes.append((
                    int(round(x / upscale_factor)),
                    int(round(y / upscale_factor)),
                    int(round(w / upscale_factor)),
                    int(round(h / upscale_factor)),
                ))

    return merge_detections_by_vote(
        all_boxes,
        iou_threshold=merge_iou_threshold,
        min_votes=min_votes,
    )


def draw_bounding_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    """Draw face bounding boxes on a copy of the image."""
    output = image.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
    return output
