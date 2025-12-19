import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cv2
from ultralytics import YOLO
from weight_utils import estimate_weight_index


def process_video(
    video_path="input/input.mp4",
    output_path="output/output_video.mp4",
    conf=0.25,
    sample_fps=5
):

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open video file")
        return {}

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_interval = int(original_fps / sample_fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, sample_fps, (width, height))

    frame_id = 0
    counts_over_time = {}

    id_map = {}
    next_id = 1

    # storage for all w_index values per ID
    weight_storage = {}

    print("Processing video... please wait")

    while True:

        success, frame = cap.read()
        if not success:
            break

        if frame_id % frame_interval == 0:

            results = model.track(
                frame,
                persist=True,
                conf=conf,
                iou=0.30,
                device="cpu"
            )

            bird_count = 0

            if results[0].boxes is not None:

                for box in results[0].boxes:

                    cls = int(box.cls)

                    # only detect birds (class id = 14 in COCO)
                    if cls != 14:
                        continue

                    if box.id is None:
                        continue

                    original_id = int(box.id)

                    # assign new sequential id if not exists
                    if original_id not in id_map:
                        id_map[original_id] = next_id
                        next_id += 1

                    tid = id_map[original_id]
                    bird_count += 1

                    x1, y1, x2, y2 = box.xyxy[0]

                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"ID:{tid}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    box_area = (x2 - x1) * (y2 - y1)
                    w_index = estimate_weight_index(float(box_area))

                    # store weights for averaging
                    if tid not in weight_storage:
                        weight_storage[tid] = []

                    weight_storage[tid].append(w_index)

                    cv2.putText(
                        frame,
                        f"W:{w_index}",
                        (int(x1), int(y2) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2
                    )

            counts_over_time[frame_id] = bird_count

            cv2.putText(
                frame,
                f"Count:{bird_count}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

            writer.write(frame)

        frame_id += 1

    writer.release()
    cap.release()

    print("Processing Completed.")

    # final list of IDs
    unique_ids = list(weight_storage.keys())

    # average weight by ID
    final_weights = {
        bid: round(sum(weight_storage[bid]) / len(weight_storage[bid]), 3)
        for bid in unique_ids
    }

    return {
        "frame_counts": counts_over_time,
        "unique_ids": unique_ids,
        "weights": final_weights
    }


if __name__ == "__main__":
    output = process_video()
    print("Counts Over Time =")
    print(output)
