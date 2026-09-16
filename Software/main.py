"""
main.py
-------
High-recall multi-cascade Haar face detector, with vote-based merging
and an optional auto-tune mode that grid-searches parameters to try to
hit a target precision/recall on YOUR dataset.

Typical usage:

    # Run once with a chosen config
    python main.py --min-neighbors 4 --min-votes 2

    # Search for a config that hits >=60% precision and >=50% recall
    python main.py --auto-tune --target-precision 0.60 --target-recall 0.50

Images are preprocessed once and cached in memory; only the Haar
detection + merge step is re-run per parameter combination during
--auto-tune, so the search is much cheaper than re-running main.py
by hand over and over.
"""

import argparse
import os
import time
import itertools
import cv2

from preprocessing import preprocess

from face_detection import (
    load_all_cascades,
    detect_faces_high_recall,
    draw_bounding_boxes,
)

from evaluation import evaluate_dataset

from wider_annotations import load_wider_annotations


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def find_images(dataset_dir):
    image_paths = []
    for root, _, files in os.walk(dataset_dir):
        for filename in files:
            if filename.lower().endswith(SUPPORTED_EXTENSIONS):
                image_paths.append(os.path.join(root, filename))
    return sorted(image_paths)


def create_prediction_key(image_path, dataset_dir):
    relative_path = os.path.relpath(image_path, dataset_dir)
    return relative_path.replace(os.sep, "/")


def scale_boxes(boxes, scale_x, scale_y):
    scaled = []
    for x, y, w, h in boxes:
        scaled.append((
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            int(round(w * scale_x)),
            int(round(h * scale_y)),
        ))
    return scaled


def parse_arguments():
    parser = argparse.ArgumentParser(description="High-recall Haar face detector")

    parser.add_argument("--dataset", default="datasets/WIDER_subset/images")
    parser.add_argument("--annotations",
                         default="datasets/WIDER_subset/annotations/wider_face_val_bbx_gt.txt")
    parser.add_argument("--output", default="output/wider_high_recall")

    # Haar parameters
    parser.add_argument("--scale-factor", type=float, default=1.05,
                         help="Image scale step. Lower values search more scales.")
    parser.add_argument("--min-neighbors", type=int, default=4,
                         help="Detection strictness. Lower values favor recall.")
    parser.add_argument("--min-size", type=int, nargs=2, default=(10, 10),
                         help="Minimum detectable face size.")

    # Ensemble merge parameters
    parser.add_argument("--min-votes", type=int, default=2,
                         help="Minimum number of agreeing detection passes (out of "
                              "up to 6: 3 cascades x 2 scales) required to keep a face. "
                              "1 = old behavior (max recall, weak precision). "
                              "2-3 = filters lone false positives, the main precision lever.")
    parser.add_argument("--merge-iou-threshold", type=float, default=0.40,
                         help="IoU threshold for treating two detections as the same face.")

    # Image parameters
    parser.add_argument("--max-dimension", type=int, default=1200)
    parser.add_argument("--upscale-factor", type=float, default=1.5)
    parser.add_argument("--no-upscale", action="store_true")

    # Evaluation
    parser.add_argument("--iou-threshold", type=float, default=0.5)

    # Auto-tune
    parser.add_argument("--auto-tune", action="store_true",
                         help="Grid-search min-neighbors/scale-factor/min-votes/min-size "
                              "to find a config meeting the target precision/recall.")
    parser.add_argument("--target-precision", type=float, default=0.60)
    parser.add_argument("--target-recall", type=float, default=0.50)
    parser.add_argument("--auto-tune-sample-size", type=int, default=25,
                         help="Run the search on a random subset of this many images "
                              "first (much faster), then confirm the winner on the full "
                              "dataset. Pass 0 to search the full dataset directly "
                              "(slow — not recommended for 100+ images).")

    return parser.parse_args()


def load_and_cache_images(image_paths, dataset_dir, ground_truth_original, max_dimension):
    """Preprocess every image once and cache grayscale + scale info.

    This is the expensive, param-independent part of the pipeline
    (disk I/O, resize, grayscale, histogram equalization). Caching it
    means --auto-tune only has to re-run the cheap Haar detection step
    for each parameter combination, not full preprocessing.

    Returns:
        list[dict]: one entry per successfully loaded image, containing
        filename, gray image, color image (for drawing), and
        ground-truth boxes already resized to match the processed image.
    """
    cache = []

    for image_path in image_paths:
        filename = create_prediction_key(image_path, dataset_dir)

        original = cv2.imread(image_path)
        if original is None:
            print(f"    ERROR: Could not read image: {image_path}")
            continue

        original_height, original_width = original.shape[:2]

        processed_color, gray = preprocess(
            image_path, max_dimension=max_dimension, use_histogram_equalization=True,
        )
        processed_height, processed_width = processed_color.shape[:2]

        scale_x = processed_width / float(original_width)
        scale_y = processed_height / float(original_height)

        gt_boxes = ground_truth_original.get(filename, [])
        gt_boxes_resized = scale_boxes(gt_boxes, scale_x, scale_y)

        cache.append({
            "filename": filename,
            "gray": gray,
            "color": processed_color,
            "gt_boxes": gt_boxes_resized,
        })

    return cache


def run_detection_pass(cache, cascades, config, save_output_dir=None, show_progress=False):
    """Run detection + merge over every cached image for one parameter config.

    Args:
        cache (list[dict]): Output of load_and_cache_images().
        cascades (dict): Loaded Haar cascades.
        config (dict): scale_factor, min_neighbors, min_size, use_upscale,
            upscale_factor, merge_iou_threshold, min_votes.
        save_output_dir (str or None): If given, save annotated images here.
        show_progress (bool): If True, print a live "image i/N" progress line
            (overwritten in place) so long runs don't look frozen.

    Returns:
        tuple: (predictions dict, ground_truth dict, mean_latency_seconds)
    """
    predictions = {}
    ground_truth = {}
    total_time = 0.0
    n = len(cache)

    for index, entry in enumerate(cache, start=1):
        if show_progress:
            print(f"\r    image {index}/{n} ({entry['filename']})" + " " * 10,
                  end="", flush=True)

        start = time.perf_counter()
        boxes = detect_faces_high_recall(
            entry["gray"], cascades,
            scale_factor=config["scale_factor"],
            min_neighbors=config["min_neighbors"],
            min_size=tuple(config["min_size"]),
            use_upscale=config["use_upscale"],
            upscale_factor=config["upscale_factor"],
            merge_iou_threshold=config["merge_iou_threshold"],
            min_votes=config["min_votes"],
        )
        total_time += time.perf_counter() - start

        predictions[entry["filename"]] = boxes
        ground_truth[entry["filename"]] = entry["gt_boxes"]

        if save_output_dir:
            output_image = draw_bounding_boxes(entry["color"], boxes)
            output_path = os.path.join(save_output_dir, entry["filename"])
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, output_image)

    if show_progress:
        print("\r" + " " * 60 + "\r", end="", flush=True)

    mean_latency = total_time / len(cache) if cache else 0.0
    return predictions, ground_truth, mean_latency


def print_metrics(metrics, mean_latency, num_images, header):
    fps = 1.0 / mean_latency if mean_latency > 0 else 0.0
    print()
    print("=" * 65)
    print(header)
    print("=" * 65)
    print(f"True Positives  : {metrics['true_positives']}")
    print(f"False Positives : {metrics['false_positives']}")
    print(f"False Negatives : {metrics['false_negatives']}")
    print(f"Precision       : {metrics['precision']:.3f}")
    print(f"Recall          : {metrics['recall']:.3f}")
    print(f"F1-score        : {metrics['f1_score']:.3f}")
    print()
    print(f"Images processed : {num_images}")
    print(f"Mean latency     : {mean_latency * 1000:.2f} ms")
    print(f"Throughput       : {fps:.2f} FPS")


def auto_tune(cache, cascades, args):
    """Grid-search parameter combinations to hit the target precision/recall.

    Search space is kept deliberately small (48 combos) since each combo
    requires a full detection pass over the (possibly sampled) dataset.
    """
    scale_factors = [1.05, 1.1]
    min_neighbors_options = [3, 4, 5, 6]
    min_size_options = [(10, 10), (15, 15)]
    min_votes_options = [1, 2, 3]

    search_cache = cache
    if args.auto_tune_sample_size and args.auto_tune_sample_size < len(cache):
        import random
        search_cache = random.sample(cache, args.auto_tune_sample_size)
        print(f"\nSearching on a random sample of {len(search_cache)} / {len(cache)} images "
              f"for speed. The winning config will be re-checked on the full dataset.")

    combos = list(itertools.product(
        scale_factors, min_neighbors_options, min_size_options, min_votes_options
    ))
    print(f"\nGrid-searching {len(combos)} parameter combinations "
          f"(target: precision >= {args.target_precision:.0%}, recall >= {args.target_recall:.0%})...")
    print(f"Each combo runs {len(search_cache)} image(s) x up to 6 detection passes each.\n")

    results = []
    combo_start_all = time.perf_counter()

    for combo_index, (scale_factor, min_neighbors, min_size, min_votes) in enumerate(combos, start=1):
        config = {
            "scale_factor": scale_factor,
            "min_neighbors": min_neighbors,
            "min_size": min_size,
            "use_upscale": not args.no_upscale,
            "upscale_factor": args.upscale_factor,
            "merge_iou_threshold": args.merge_iou_threshold,
            "min_votes": min_votes,
        }

        combo_start = time.perf_counter()
        predictions, ground_truth, mean_latency = run_detection_pass(
            search_cache, cascades, config, show_progress=True,
        )
        combo_elapsed = time.perf_counter() - combo_start
        metrics = evaluate_dataset(predictions, ground_truth, iou_threshold=args.iou_threshold)

        meets_target = (
            metrics["precision"] >= args.target_precision
            and metrics["recall"] >= args.target_recall
        )
        results.append((config, metrics, meets_target))

        flag = "  <-- MEETS TARGET" if meets_target else ""
        avg_combo_time = (time.perf_counter() - combo_start_all) / combo_index
        remaining = avg_combo_time * (len(combos) - combo_index)
        print(f"  [{combo_index}/{len(combos)}, {combo_elapsed:.1f}s, ~{remaining/60:.1f} min left] "
              f"scale={scale_factor:<5} min_neighbors={min_neighbors:<2} "
              f"min_size={min_size} min_votes={min_votes}  ->  "
              f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
              f"F1={metrics['f1_score']:.3f}{flag}", flush=True)

    passing = [r for r in results if r[2]]

    if passing:
        # Among passing configs, prefer the highest F1
        best = max(passing, key=lambda r: r[1]["f1_score"])
        print(f"\n{len(passing)} config(s) met the target. Picking the best F1-score.")
    else:
        # No config hit both targets; report the closest by F1 as a starting point
        best = max(results, key=lambda r: r[1]["f1_score"])
        print("\nNo config hit both targets on this sample. Showing the closest (highest F1) "
              "as a starting point — you'll likely need to expand the search ranges above, "
              "check your ground-truth annotations, or accept a different precision/recall "
              "tradeoff for this dataset.")

    best_config, best_metrics_sample, _ = best
    print(f"\nBest config: scale_factor={best_config['scale_factor']}, "
          f"min_neighbors={best_config['min_neighbors']}, "
          f"min_size={best_config['min_size']}, min_votes={best_config['min_votes']}")

    if search_cache is not cache:
        print("\nRe-checking best config on the FULL dataset...")
        predictions, ground_truth, mean_latency = run_detection_pass(cache, cascades, best_config)
        metrics = evaluate_dataset(predictions, ground_truth, iou_threshold=args.iou_threshold)
        print_metrics(metrics, mean_latency, len(cache), "FINAL RESULT (full dataset)")
    else:
        print_metrics(best_metrics_sample, 0.0, len(search_cache), "FINAL RESULT (search set)")

    return best_config


def main():
    args = parse_arguments()
    os.makedirs(args.output, exist_ok=True)

    print()
    print("=" * 65)
    print("HIGH-RECALL HAAR FACE DETECTOR")
    print("=" * 65)

    print("\nLoading WIDER annotations...")
    ground_truth_original = load_wider_annotations(args.annotations)

    image_paths = find_images(args.dataset)
    print(f"Images found       : {len(image_paths)}")
    print(f"Annotated images   : {len(ground_truth_original)}")

    print("\nLoading Haar classifiers...")
    cascades = load_all_cascades()
    for cascade_name in cascades:
        print(f"  Loaded: {cascade_name}")

    print("\nPreprocessing and caching images (done once)...")
    cache = load_and_cache_images(image_paths, args.dataset, ground_truth_original, args.max_dimension)
    print(f"Cached {len(cache)} images.")

    if args.auto_tune:
        auto_tune(cache, cascades, args)
        return

    config = {
        "scale_factor": args.scale_factor,
        "min_neighbors": args.min_neighbors,
        "min_size": tuple(args.min_size),
        "use_upscale": not args.no_upscale,
        "upscale_factor": args.upscale_factor,
        "merge_iou_threshold": args.merge_iou_threshold,
        "min_votes": args.min_votes,
    }

    print("\nConfiguration")
    print("-" * 65)
    for key, value in config.items():
        print(f"{key:20s}: {value}")

    predictions, ground_truth, mean_latency = run_detection_pass(
        cache, cascades, config, save_output_dir=args.output,
    )
    metrics = evaluate_dataset(predictions, ground_truth, iou_threshold=args.iou_threshold)
    print_metrics(metrics, mean_latency, len(cache), "DETECTION QUALITY & PERFORMANCE")
    print(f"\nAnnotated images saved to: {args.output}")


if __name__ == "__main__":
    main()
