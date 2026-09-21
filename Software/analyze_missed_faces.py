import json
from collections import Counter

# ------------------------------------------------------------
# Load YuNet evaluation results
# ------------------------------------------------------------

with open("output/yunet/metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)

# ------------------------------------------------------------
# Load WIDER FACE annotations
# ------------------------------------------------------------

from wider_annotations import load_wider_annotations

ANNOTATION_FILE = (
    "datasets/WIDER_subset/annotations/"
    "wider_face_val_bbx_gt.txt"
)

ground_truth = load_wider_annotations(ANNOTATION_FILE)

# ------------------------------------------------------------
# Load predictions
#
# IMPORTANT:
# metrics.json contains TP/FP/FN counts per image,
# but not the individual matched boxes.
#
# Therefore this first analysis uses the FN count
# together with the ground-truth face-size distribution
# of the highest-FN images.
# ------------------------------------------------------------

top_images = sorted(
    metrics["per_image"].items(),
    key=lambda x: x[1]["false_negatives"],
    reverse=True
)[:20]

print()
print("=" * 80)
print("MISSED-FACE / FACE-SIZE ANALYSIS")
print("=" * 80)

size_counts = Counter()

for filename, result in top_images:

    gt_boxes = ground_truth.get(filename, [])

    print()
    print(filename)
    print(
        f"GT={result['true_positives'] + result['false_negatives']} "
        f"Detected={result['true_positives'] + result['false_positives']} "
        f"TP={result['true_positives']} "
        f"FP={result['false_positives']} "
        f"FN={result['false_negatives']}"
    )

    for box in gt_boxes:

        x, y, w, h = box

        # Use face width as the primary size measure
        if w < 10:
            category = "<10 px"
        elif w < 20:
            category = "10-19 px"
        elif w < 40:
            category = "20-39 px"
        elif w < 80:
            category = "40-79 px"
        else:
            category = "80+ px"

        size_counts[category] += 1

print()
print("=" * 80)
print("GROUND-TRUTH FACE SIZE DISTRIBUTION")
print("(Top 20 highest-FN images)")
print("=" * 80)

order = [
    "<10 px",
    "10-19 px",
    "20-39 px",
    "40-79 px",
    "80+ px",
]

total = sum(size_counts.values())

for category in order:
    count = size_counts[category]
    percentage = (count / total * 100) if total else 0

    print(
        f"{category:10s} : "
        f"{count:5d} faces "
        f"({percentage:6.2f}%)"
    )

print()
print(f"Total ground-truth faces analysed: {total}")
print("=" * 80)