import cv2
import time

from yunet_detector import load_yunet, detect_faces_yunet


MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"


def main():
    print("Loading YuNet model...")

    detector = load_yunet(
        MODEL_PATH,
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
    )

    print("Opening webcam...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print()
    print("YuNet real-time face detection started.")
    print("Press Q to quit.")

    previous_time = time.perf_counter()

    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read webcam frame.")
            break

        # Measure detection time
        start_time = time.perf_counter()

        boxes = detect_faces_yunet(frame, detector)

        detection_time = time.perf_counter() - start_time

        # Draw bounding boxes
        for x, y, w, h in boxes:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # Calculate FPS
        current_time = time.perf_counter()
        frame_time = current_time - previous_time
        previous_time = current_time

        fps = 1.0 / frame_time if frame_time > 0 else 0.0

        # Display information
        cv2.putText(
            frame,
            f"Faces: {len(boxes)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Detection: {detection_time * 1000:.1f} ms",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Show live video
        cv2.imshow(
            "YuNet Real-Time Face Detection",
            frame
        )

        # Press Q to quit
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Webcam detection stopped.")


if __name__ == "__main__":
    main()