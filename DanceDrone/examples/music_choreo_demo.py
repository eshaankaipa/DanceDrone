"""
Example: synchronise Tello movements with a music track using librosa.

Usage:
    python examples/music_choreo_demo.py --audio path/to/song.mp3

The script loads the audio file, detects beats, builds a basic choreography,
and then plays it on a connected Tello. Pass `--preview` to print the planned
commands without flying the drone.
"""

import argparse
import sys
import time

from djitellopy import MusicChoreographer, Tello


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tello music choreography demo.")
    parser.add_argument("--audio", required=True, help="Path to an audio file supported by librosa.")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Only print the schedule (no connection or commands sent).",
    )
    parser.add_argument(
        "--latency",
        type=float,
        default=0.25,
        help="Command lead time in seconds to offset network latency.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    tello = None if args.preview else Tello()

    choreographer = MusicChoreographer(
        tello=tello,
        audio_path=args.audio,
        command_latency=args.latency,
    )

    choreographer.detect_beats()
    choreographer.build_schedule()

    if args.preview:
        choreographer.play_schedule(preview_only=True)
        return 0

    try:
        tello.connect()
        tello.streamoff()
        choreographer.play_schedule()

        while True:
            time.sleep(0.2)
            if choreographer._playback_thread is None or not choreographer._playback_thread.is_alive():
                break
    except KeyboardInterrupt:
        choreographer.stop()
    finally:
        choreographer.stop()
        try:
            tello.end()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())

