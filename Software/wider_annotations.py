"""
wider_annotations.py
--------------------
Read WIDER FACE bounding-box annotations.
"""


def load_wider_annotations(
    annotation_file
):

    annotations = {}

    with open(
        annotation_file,
        "r",
        encoding="utf-8"
    ) as file:

        while True:

            image_path = file.readline()

            if not image_path:
                break

            image_path = image_path.strip()

            if not image_path:
                continue

            number_line = file.readline()

            if not number_line:
                break

            number_of_faces = int(
                number_line.strip()
            )

            boxes = []

            for _ in range(
                number_of_faces
            ):

                line = file.readline().strip()

                values = line.split()

                x = int(values[0])
                y = int(values[1])
                w = int(values[2])
                h = int(values[3])

                boxes.append(
                    (x, y, w, h)
                )

            annotations[
                image_path
            ] = boxes

    return annotations