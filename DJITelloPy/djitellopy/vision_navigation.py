

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - runtime check
    raise ImportError(
        "ultralytics is required for VisionNavigationSystem. "
        "Install with `pip install ultralytics`."
    ) from exc

try:
    from .tello import Tello
except ImportError:  # pragma: no cover - allow docs/tests without full SDK
    Tello = "Tello"  # type: ignore[assignment]


@dataclass
class DetectionResult:
    """Container for YOLO detections with derived depth estimates."""

    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    distance: float  # metres (relative)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))


class ErrorCorrector:
    """
    Simple PID controller with output clamping and integral wind-up protection.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        *,
        integrator_limit: float = 1.0,
        output_limit: float = 1.0,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integrator_limit = integrator_limit
        self.output_limit = output_limit

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()

    def update(self, error: float, *, current_time: Optional[float] = None) -> float:
        now = current_time or time.time()
        dt = max(1e-3, now - self._prev_time)

        # Integral term with clamping
        self._integral += error * dt
        self._integral = float(np.clip(self._integral, -self.integrator_limit, self.integrator_limit))

        derivative = (error - self._prev_error) / dt
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        output = float(np.clip(output, -self.output_limit, self.output_limit))

        self._prev_error = error
        self._prev_time = now
        return output


class VisionNavigationSystem:
    """
    Vision-driven collision avoidance helper for DJI Tello.

    Parameters
    ----------
    tello:
        Connected :class:`Tello` instance.
    model_path:
        Path or model identifier for the YOLO weights (default: ``yolov8n.pt``).
    known_object_widths:
        Optional mapping from class label to real-world width in metres.  Used
        to refine depth estimates.  When absent, a fallback heuristic is used.
    safe_distance:
        Minimum desired distance to the closest detected obstacle (metres).
    target_height:
        Desired altitude in centimetres.
    """

    def __init__(
        self,
        tello: Tello,
        *,
        model_path: str = "yolov8n.pt",
        known_object_widths: Optional[Dict[str, float]] = None,
        safe_distance: float = 1.5,
        target_height: float = 120.0,
        height_pid: Tuple[float, float, float] = (0.5, 0.02, 0.1),
        distance_pid: Tuple[float, float, float] = (0.8, 0.0, 0.2),
    ) -> None:
        self._tello = tello
        self._model = YOLO(model_path)
        self._safe_distance = safe_distance
        self._target_height = target_height
        self._known_widths = known_object_widths or {}

        self._height_corrector = ErrorCorrector(*height_pid, integrator_limit=0.5, output_limit=0.8)
        self._distance_corrector = ErrorCorrector(*distance_pid, integrator_limit=0.3, output_limit=1.0)

        self._height_estimate = target_height / 100.0  # metres
        self._focal_pixels = 920.0  # approximate Tello focal length (px)

    # --------------------------------------------------------------------- #
    # Perception
    # --------------------------------------------------------------------- #

    def estimate_height(self, alpha: float = 0.7) -> float:
        """
        Fuse Tello ToF telemetry with exponential smoothing.

        Parameters
        ----------
        alpha:
            Smoothing factor (0..1).  Higher values weigh the latest reading
            more heavily.

        Returns
        -------
        float
            Estimated altitude in metres.
        """
        try:
            raw_cm = self._tello.get_distance_tof()
        except Exception:
            # Fallback to state dictionary if available
            try:
                state = self._tello.get_current_state()
                raw_cm = float(state.get("tof", self._height_estimate * 100))
            except Exception:
                raw_cm = self._height_estimate * 100

        raw_m = max(0.0, raw_cm / 100.0)
        self._height_estimate = alpha * raw_m + (1 - alpha) * self._height_estimate
        return self._height_estimate

    def _estimate_depth_from_detection(self, detection: DetectionResult) -> float:
        """
        Estimate object distance from its apparent width using the pinhole model.
        """
        x1, _, x2, _ = detection.bbox
        pixel_width = max(1.0, x2 - x1)
        real_width = self._known_widths.get(detection.label, 0.45)  # default to ~45cm
        depth = (real_width * self._focal_pixels) / pixel_width
        return max(depth, 0.1)

    def detect_obstacles(
        self,
        frame: np.ndarray,
        *,
        conf_threshold: float = 0.35,
    ) -> List[DetectionResult]:
        """
        Run YOLO inference and return detections with depth estimates.
        """
        results = self._model(frame, conf=conf_threshold, verbose=False)
        detections: List[DetectionResult] = []

        for result in results:
            names = result.names if hasattr(result, "names") else {}
            for box in getattr(result, "boxes", []):
                cls_idx = int(box.cls.item())
                label = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                detection = DetectionResult(
                    label=label,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    distance=0.0,
                )
                detection.distance = self._estimate_depth_from_detection(detection)
                detections.append(detection)
        return detections

    # --------------------------------------------------------------------- #
    # Control
    # --------------------------------------------------------------------- #

    def avoid_obstacles(
        self,
        frame: np.ndarray,
        *,
        display: bool = False,
        conf_threshold: float = 0.35,
        min_confidence: float = 0.2,
        dt: float = 0.1,
    ) -> Dict[str, float]:
        """
        Analyse the frame, update corrections, and optionally issue RC commands.

        Returns a dictionary describing the latest control outputs.
        """
        height_m = self.estimate_height()
        detections = self.detect_obstacles(frame, conf_threshold=conf_threshold)

        closest = min((d for d in detections if d.confidence >= min_confidence), default=None, key=lambda d: d.distance)
        distance_error = 0.0
        forward_velocity = 0.0

        if closest is not None:
            distance_error = self._safe_distance - closest.distance
            correction = self._distance_corrector.update(distance_error, current_time=time.time())
            forward_velocity = -int(round(correction * 60))  # negative -> move backward when too close
        else:
            self._distance_corrector.reset()

        height_error = (self._target_height / 100.0) - height_m
        vertical_correction = self._height_corrector.update(height_error, current_time=time.time())
        vertical_velocity = int(round(vertical_correction * 60))  # cm/s approx

        # Saturate velocities to safe bounds
        forward_velocity = int(np.clip(forward_velocity, -60, 60))
        vertical_velocity = int(np.clip(vertical_velocity, -60, 60))

        # Issue RC commands if the tello object supports it and is in flight
        try:
            if getattr(self._tello, "is_flying", False):
                self._tello.send_rc_control(0, forward_velocity, vertical_velocity, 0)
        except AttributeError:
            # Developer edition may not expose full RC API
            pass

        if display:
            annotated = frame.copy()
            self._draw_annotations(annotated, detections, height_m)
            cv2.imshow("Tello Vision Navigation", annotated)
            cv2.waitKey(int(dt * 1000))

        return {
            "height_m": height_m,
            "height_error": height_error,
            "distance_error": distance_error,
            "forward_velocity": forward_velocity,
            "vertical_velocity": vertical_velocity,
            "detections": detections,
        }

    # --------------------------------------------------------------------- #
    # Visualisation
    # --------------------------------------------------------------------- #

    def _draw_annotations(
        self,
        frame: np.ndarray,
        detections: Iterable[DetectionResult],
        height_m: float,
    ) -> None:
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"{det.label} {det.confidence:.2f} {det.distance:.2f}m",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"Altitude: {height_m:.2f} m (target {self._target_height/100:.2f} m)",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def shutdown(self) -> None:
        """Cleanup any OpenCV windows."""
        cv2.destroyAllWindows()


