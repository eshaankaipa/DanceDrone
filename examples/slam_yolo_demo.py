
import argparse
import sys
import time

from djitellopy import Tello, YoloSLAM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tello SLAM + YOLO demo.")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO weights path or model name understood by ultralytics.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Minimum detection confidence.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional demo duration in seconds.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the annotated frames using OpenCV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    tello = Tello()
    slam = YoloSLAM(
        tello,
        model_path=args.model,
        confidence=args.confidence,
    )

    try:
        slam.run(duration=args.duration, display=args.display)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        try:
            tello.streamoff()
        except Exception:
            pass
        try:
            tello.end()
        except AttributeError:
            # The simplified developer repo may not implement `end`.
            pass
        time.sleep(0.5)

    trajectory = slam.trajectory
    if trajectory.size:
        print("Trajectory samples:")
        for idx, pose in enumerate(trajectory[-10:]):
            print(f"{idx:02d}: {pose}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

