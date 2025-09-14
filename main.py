import cv2
import os
import signal
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

YOLO_MODEL_PATH = "yolov8s.pt"  # Use yolov8s.pt for better accuracy
CONFIDENCE_THRESHOLD = 0.35

def main():
    cap = None
    try:
        print("🔄 Loading YOLOv8 model...")
        model = YOLO(YOLO_MODEL_PATH)
        print(f"Available classes: {model.names}")

        print("🔄 Initializing Deep SORT tracker...")
        tracker = DeepSort(max_age=10, n_init=3)  # max_age reduced to 10 for quicker track deletion

        for i in range(5):
            temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if temp_cap.isOpened():
                ret, frame = temp_cap.read()
                if ret:
                    cap = temp_cap
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
                    print(f"✅ Using webcam at index {i}")
                    break
                else:
                    temp_cap.release()

        if cap is None:
            print("❌ No working webcam found. Exiting...")
            return

        class_names = model.names
        print("▶️ Starting object tracking... Press 'q' to quit.")

        prev_time = time.time()

        window_name = "YOLOv8 + Deep SORT Object Tracking"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame read failed. Exiting...")
                break

            # Run YOLO detection
            results = model(frame, verbose=False)[0]

            detections = []
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                cls_id = int(box.cls[0])
                class_name = class_names[cls_id]
                w, h = x2 - x1, y2 - y1

                detections.append(([x1, y1, w, h], conf, class_name))

            # Update tracker with current detections
            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                class_name = getattr(track, 'det_class', "Unknown")

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"ID:{track_id} {class_name}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Calculate and show FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 'q' pressed. Exiting...")
                break

    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt detected. Exiting...")

    finally:
        if cap is not None:
            print("🔻 Releasing camera...")
            cap.release()

        print("🧹 Closing all OpenCV windows...")
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        print("✅ Program exited successfully.")

if __name__ == "__main__":
    main()
