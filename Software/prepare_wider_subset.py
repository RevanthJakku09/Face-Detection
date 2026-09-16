import os
import random
import shutil

# ============================================================
# PATHS
# ============================================================

# Original WIDER FACE dataset
WIDER_VAL_DIR = r"D:\Final_Project\archive (1)\WIDER_val\WIDER_val\images"

ANNOTATION_FILE = (
    r"D:\Final_Project\archive (1)\wider_face_annotations"
    r"\wider_face_split\wider_face_val_bbx_gt.txt"
)

# Our project subset
OUTPUT_IMAGE_DIR = r"datasets\WIDER_subset\images"
OUTPUT_ANNOTATION_DIR = r"datasets\WIDER_subset\annotations"

OUTPUT_ANNOTATION_FILE = os.path.join(
    OUTPUT_ANNOTATION_DIR,
    "wider_face_val_bbx_gt.txt"
)

# Number of images to select
NUM_IMAGES = 100

# Fixed seed so the same images are selected every time
RANDOM_SEED = 42


# ============================================================
# READ WIDER FACE ANNOTATIONS
# ============================================================

def read_annotations(annotation_file):
    entries = []

    with open(annotation_file, "r", encoding="utf-8") as f:
        while True:
            image_path = f.readline()

            if not image_path:
                break

            image_path = image_path.strip()

            if not image_path:
                continue

            num_faces_line = f.readline()

            if not num_faces_line:
                break

            num_faces = int(num_faces_line.strip())

            boxes = []

            for _ in range(num_faces):
                box = f.readline().strip()
                boxes.append(box)

            entries.append({
                "image": image_path,
                "num_faces": num_faces,
                "boxes": boxes
            })

    return entries


# ============================================================
# MAIN
# ============================================================

def main():

    # Check original dataset
    if not os.path.exists(WIDER_VAL_DIR):
        raise FileNotFoundError(
            f"WIDER_val folder not found:\n{WIDER_VAL_DIR}"
        )

    if not os.path.exists(ANNOTATION_FILE):
        raise FileNotFoundError(
            f"Annotation file not found:\n{ANNOTATION_FILE}"
        )

    # Create output folders
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_ANNOTATION_DIR, exist_ok=True)

    print("Reading WIDER FACE validation annotations...")

    entries = read_annotations(ANNOTATION_FILE)

    print(f"Total validation images found: {len(entries)}")

    # Make sure we have enough images
    if len(entries) < NUM_IMAGES:
        raise ValueError(
            f"Only {len(entries)} images found, "
            f"but {NUM_IMAGES} were requested."
        )

    # Select random images
    random.seed(RANDOM_SEED)

    selected = random.sample(entries, NUM_IMAGES)

    # Copy images and write matching annotations
    copied = 0

    with open(
        OUTPUT_ANNOTATION_FILE,
        "w",
        encoding="utf-8"
    ) as out:

        for entry in selected:

            relative_image_path = entry["image"]

            source_image = os.path.join(
                WIDER_VAL_DIR,
                relative_image_path
            )

            destination_image = os.path.join(
                OUTPUT_IMAGE_DIR,
                relative_image_path
            )

            # Check image exists
            if not os.path.exists(source_image):
                print(
                    f"WARNING: Image not found: {source_image}"
                )
                continue

            # Create category/event folder
            os.makedirs(
                os.path.dirname(destination_image),
                exist_ok=True
            )

            # Copy image
            shutil.copy2(
                source_image,
                destination_image
            )

            # Write annotation
            out.write(relative_image_path + "\n")
            out.write(str(entry["num_faces"]) + "\n")

            for box in entry["boxes"]:
                out.write(box + "\n")

            copied += 1

    print()
    print("========================================")
    print("WIDER FACE SUBSET CREATED")
    print("========================================")
    print(f"Images selected : {NUM_IMAGES}")
    print(f"Images copied   : {copied}")
    print()
    print(f"Images:")
    print(f"  {OUTPUT_IMAGE_DIR}")
    print()
    print(f"Annotations:")
    print(f"  {OUTPUT_ANNOTATION_FILE}")
    print("========================================")


if __name__ == "__main__":
    main()