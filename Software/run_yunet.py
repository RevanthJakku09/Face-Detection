"""Evaluate YuNet on the project's WIDER FACE 100-image subset."""

import argparse
import json
import os
import time

import cv2

from evaluation import evaluate_dataset
from preprocessing import resize_image
from wider_annotations import load_wider_annotations
from yunet_detector import load_yunet, detect_faces_yunet, draw_yunet_boxes


EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def find_images(root):
    paths = []
    for directory, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith(EXTENSIONS):
                paths.append(os.path.join(directory, name))
    return sorted(paths)


def key_for(path, dataset):
    return os.path.relpath(path, dataset).replace(os.sep, "/")


def scale_boxes(boxes, sx, sy):
    return [
        (round(x * sx), round(y * sy), round(w * sx), round(h * sy))
        for x, y, w, h in boxes
    ]


def parse_args():
    p = argparse.ArgumentParser(description="YuNet WIDER FACE evaluation")
    p.add_argument("--dataset", default="datasets/WIDER_subset/images")
    p.add_argument(
        "--annotations",
        default="datasets/WIDER_subset/annotations/wider_face_val_bbx_gt.txt",
    )
    p.add_argument("--model", default="models/face_detection_yunet_2023mar.onnx")
    p.add_argument("--output", default="output/yunet")
    p.add_argument("--score-threshold", type=float, default=0.6)
    p.add_argument("--nms-threshold", type=float, default=0.3)
    p.add_argument("--top-k", type=int, default=5000)
    p.add_argument("--max-dimension", type=int, default=1200)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--save-images", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("\n" + "=" * 65)
    print("YUNET FACE DETECTOR")
    print("=" * 65)
    print(f"Model              : {args.model}")
    print(f"Score threshold    : {args.score_threshold}")
    print(f"NMS threshold      : {args.nms_threshold}")
    print(f"Max dimension      : {args.max_dimension}")
    print(f"IoU threshold      : {args.iou_threshold}")

    gt_original = load_wider_annotations(args.annotations)
    image_paths = find_images(args.dataset)
    print(f"Images found       : {len(image_paths)}")
    print(f"Annotated images   : {len(gt_original)}")

    detector = load_yunet(
        args.model,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        top_k=args.top_k,
    )

    predictions = {}
    ground_truth_resized = {}
    total_detection_time = 0.0

    for index, image_path in enumerate(image_paths, 1):
        key = key_for(image_path, args.dataset)
        print(f"{index:3d}/{len(image_paths):3d} {key}")

        image = cv2.imread(image_path)
        if image is None:
            print("    ERROR: could not read image")
            continue

        original_h, original_w = image.shape[:2]
        resized = resize_image(image, max_dimension=args.max_dimension)
        resized_h, resized_w = resized.shape[:2]

        sx = resized_w / original_w
        sy = resized_h / original_h

        if key not in gt_original:
            continue

        ground_truth_resized[key] = scale_boxes(gt_original[key], sx, sy)

        start = time.perf_counter()
        boxes = detect_faces_yunet(resized, detector)
        elapsed = time.perf_counter() - start
        total_detection_time += elapsed
        predictions[key] = boxes

        print(f"    detections       : {len(boxes)}")
        print(f"    detector time    : {elapsed * 1000:.2f} ms")

        if args.save_images:
            output_image = draw_yunet_boxes(resized, boxes)
            destination = os.path.join(args.output, key)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            cv2.imwrite(destination, output_image)

    metrics = evaluate_dataset(
        predictions,
        ground_truth_resized,
        iou_threshold=args.iou_threshold,
    )

    count = len(predictions)
    mean_latency_ms = (total_detection_time / count * 1000) if count else 0.0
    fps = (count / total_detection_time) if total_detection_time else 0.0

    metrics["mean_detection_latency_ms"] = mean_latency_ms
    metrics["fps"] = fps
    metrics["images_processed"] = count

    result_path = os.path.join(args.output, "metrics.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"TP                 : {metrics['true_positives']}")
    print(f"FP                 : {metrics['false_positives']}")
    print(f"FN                 : {metrics['false_negatives']}")
    print(f"Precision          : {metrics['precision']:.4f}")
    print(f"Recall             : {metrics['recall']:.4f}")
    print(f"F1                 : {metrics['f1_score']:.4f}")
    print(f"Detection rate     : {metrics['detection_rate']:.4f}")
    print(f"Mean latency       : {mean_latency_ms:.2f} ms")
    print(f"FPS                : {fps:.2f}")
    print(f"Saved metrics      : {result_path}")


if __name__ == "__main__":
    main()
