# Face Detection – Software Baseline (Python + OpenCV + Haar Cascade)

This is the **software / CPU implementation** for the *Face Detection Using
Haar Cascade and FPGA* project. It detects faces in static images using
OpenCV's pre-trained Haar Cascade classifier, measures detection quality and
performance, and serves as the baseline that the future FPGA implementation
will be compared against.

## Status

✅ **Milestone 1 complete** — a fully working Python + OpenCV + Haar Cascade
static-image face-detection model with bounding-box output and initial
detection/performance measurements (see [Demo results](#demo-results) below).

## Folder Structure

```
Software/
├── datasets/                 → Test images (+ optional ground_truth.json)
│   ├── lena.jpg
│   ├── messi5.jpg
│   └── ground_truth.json
├── output/                   → Annotated output images (generated)
├── face_detection.py         → Haar Cascade face detection
├── preprocessing.py          → Image preprocessing
├── evaluation.py             → Detection evaluation (precision/recall/F1)
├── performance.py            → Latency/FPS measurement
├── main.py                   → Main program
├── requirements.txt          → Python dependencies
└── README.md                 → This file
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run on the sample images in `datasets/`:

```bash
python main.py
```

With custom options:

```bash
python main.py --dataset datasets --output output \
                --scale-factor 1.05 --min-neighbors 6 --min-size 30 30
```

| Flag              | Default                        | Description                                    |
|-------------------|---------------------------------|-------------------------------------------------|
| `--dataset`        | `datasets`                     | Folder of input images                         |
| `--output`         | `output`                       | Folder for annotated output images             |
| `--ground-truth`   | `datasets/ground_truth.json`   | Optional annotations for precision/recall/F1   |
| `--scale-factor`   | `1.1`                          | Haar Cascade `scaleFactor`                     |
| `--min-neighbors`  | `5`                            | Haar Cascade `minNeighbors`                    |
| `--min-size`       | `30 30`                        | Minimum face size in pixels                    |
| `--repeats`        | `5`                            | Repeats per image for latency averaging        |

## Ground-Truth Format (optional)

To get precision / recall / F1-score, add a `ground_truth.json` file to the
dataset folder:

```json
{
  "image1.jpg": [[x, y, w, h], [x, y, w, h]],
  "image2.jpg": []
}
```

Boxes are matched to predictions with IoU ≥ 0.5 by default (see
`evaluation.py`).

## Demo Results

Run on two classic OpenCV sample images (`lena.jpg`, `messi5.jpg`):

| Image       | Faces detected | Mean latency | 
|-------------|-----------------|--------------|
| lena.jpg    | 1                | ~127 ms      |
| messi5.jpg  | 2 (1 correct, 1 false positive in crowd) | ~115 ms |

**Aggregate:** ~121 ms mean latency, ~8.3 FPS on CPU.

**With ground truth:** Precision 0.667 / Recall 1.000 / F1 0.800 — the false
positive on `messi5.jpg` (a face-like pattern in the crowd) is exactly the
kind of error the "Software Improvement" tuning phase (adjusting
`scaleFactor`, `minNeighbors`, `minSize`) is meant to reduce.

## Next Steps

- [ ] Expand `datasets/` with a larger, more representative test set and
      full ground-truth annotations.
- [ ] Sweep `scaleFactor` / `minNeighbors` / `minSize` to tune precision vs.
      recall and lock in the improved software baseline.
- [ ] Feed the same finalized test images into the FPGA (Nexus 4 DDR)
      hardware pipeline for a fair CPU-vs-FPGA comparison.
- [ ] (Optional/later) Real-time webcam face detection.
