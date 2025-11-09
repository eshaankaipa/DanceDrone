from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - validated at runtime
    raise ImportError(
        "ultralytics is required for YoloSLAM. "
        "Install with `pip install ultralytics`."
    ) from exc

try:
    from .tello import Tello, BackgroundFrameRead
except ImportError:  # pragma: no cover - fallback for partial installations
    # The developer version of this repository might not ship the full Tello
    # implementation.  Defer the import error until the class is actually used.
    Tello = "Tello"  # type: ignore[assignment]
    BackgroundFrameRead = "BackgroundFrameRead"  # type: ignore[assignment]


@dataclass
class Detection:
    """Container for YOLO detections."""

    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)

    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))


@dataclass
class FrameState:
    """Pose, keypoints and semantic detections for a processed frame."""

    timestamp: float
    rotation: np.ndarray
    translation: np.ndarray
    keypoints: List[cv2.KeyPoint] = field(default_factory=list)
    descriptors: Optional[np.ndarray] = None
    detections: List[Detection] = field(default_factory=list)


class YoloSLAM:
    """
    Basic SLAM front-end for the DJI Tello video stream.

    The pipeline:

    1. Acquire frames from `Tello.get_frame_read()` or any RGB source.
    2. Run YOLO inference to obtain semantic detections.
    3. Detect ORB keypoints/descriptors and match them to the previous frame.
    4. Estimate relative pose using the essential matrix.
    5. Update a simple trajectory estimate (with unknown global scale).

    Parameters
    ----------
    tello:
        Connected `Tello` instance.  If `None`, the caller must feed frames
        manually via :meth:`process_frame`.
    model_path:
        Weights passed to `ultralytics.YOLO`. Defaults to `yolov8n.pt`.
    confidence:
        Minimum detection confidence for YOLO outputs.
    max_features:
        Maximum number of ORB features per frame.
    match_keep:
        Maximum number of descriptor matches retained for pose estimation.
    camera_matrix:
        3x3 intrinsic calibration.  If omitted, a heuristic for the Tello camera
        is used (approximate focal length of 920 px on 960x720 images).
    use_gpu:
        If True/False, forces YOLO onto CUDA/CPU.  If `None`, the default
        behaviour of `ultralytics` is used.
    """

    def __init__(
        self,
        tello: Optional[Tello] = None,
        *,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.3,
        max_features: int = 500,
        match_keep: int = 150,
        camera_matrix: Optional[np.ndarray] = None,
        use_gpu: Optional[bool] = None,
    ) -> None:
        self._tello = tello
        self._model = YOLO(model_path)
        if use_gpu is True:
            self._model.to("cuda")
        elif use_gpu is False:
            self._model.to("cpu")

        self._confidence = confidence
        self._orb = cv2.ORB_create(nfeatures=max_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._match_keep = match_keep

        if camera_matrix is None:
            self._K = self._default_intrinsics()
        else:
            if camera_matrix.shape != (3, 3):
                raise ValueError("camera_matrix must be 3x3.")
            self._K = camera_matrix.astype(np.float64)

        self._states: List[FrameState] = []
        self._last_keypoints: Optional[List[cv2.KeyPoint]] = None
        self._last_descriptors: Optional[np.ndarray] = None
        self._last_detections: List[Detection] = []

        self._R = np.eye(3, dtype=np.float64)
        self._t = np.zeros((3, 1), dtype=np.float64)
        self._trajectory: List[np.ndarray] = [self._t.copy()]

    @staticmethod
    def _default_intrinsics() -> np.ndarray:
        """Approximate intrinsics for 960x720 Tello frames."""
        fx = fy = 920.0
        cx = 480.0
        cy = 360.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def trajectory(self) -> np.ndarray:
        if not self._trajectory:
            return np.zeros((0, 3), dtype=np.float64)
        return np.vstack([pose.ravel() for pose in self._trajectory])

    @property
    def last_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._R.copy(), self._t.copy()

    @property
    def states(self) -> List[FrameState]:
        return list(self._states)

    def _ensure_stream(self) -> BackgroundFrameRead:
        if self._tello is None:
            raise RuntimeError("Tello instance is required for streaming.")

        self._tello.connect()
        self._tello.streamon()
        return self._tello.get_frame_read()

    def run(
        self,
        *,
        duration: Optional[float] = None,
        display: bool = False,
        annotate: bool = True,
        sleep_interval: float = 0.03,
    ) -> None:
        """
        Start SLAM on the drone video stream.

        Parameters
        ----------
        duration:
            Duration in seconds.  Runs indefinitely when `None`.
        display:
            If True, opens an OpenCV window showing the annotated frames.
        annotate:
            If True, draw detections and keypoints on frames when `display` is
            enabled.
        sleep_interval:
            Delay between frames to avoid hogging the CPU.
        """
        frame_reader = self._ensure_stream()
        start = time.time()

        try:
            while True:
                frame = frame_reader.frame
                if frame is None:
                    time.sleep(0.01)
                    continue

                state = self.process_frame(frame, timestamp=time.time())
                if display and state is not None:
                    output = frame.copy()
                    if annotate:
                        self._draw_annotations(output, state)
                    cv2.imshow("Tello YOLO SLAM", output)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break

                if duration is not None and (time.time() - start) >= duration:
                    break

                time.sleep(sleep_interval)
        finally:
            if display:
                cv2.destroyAllWindows()

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        timestamp: Optional[float] = None,
    ) -> Optional[FrameState]:
        """
        Process a single RGB frame.

        Returns a :class:`FrameState` with the current pose estimate, or `None`
        when pose estimation was not possible (e.g. insufficient matches).
        """
        if frame is None or frame.size == 0:
            return None

        ts = timestamp if timestamp is not None else time.time()
        detections = self._run_yolo(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        keypoints = list(keypoints or [])

        if descriptors is None or len(keypoints) < 8 or self._last_descriptors is None:
            # Not enough information for VO – store the state and continue.
            state = FrameState(
                timestamp=ts,
                rotation=self._R.copy(),
                translation=self._t.copy(),
                keypoints=keypoints,
                descriptors=descriptors,
                detections=detections,
            )
            self._store_state(state)
            self._last_keypoints = keypoints
            self._last_descriptors = descriptors
            self._last_detections = detections
            return state

        matches = self._match_descriptors(descriptors, self._last_descriptors)
        if len(matches) < 8:
            return None

        pts_curr = np.float32([keypoints[m.queryIdx].pt for m in matches])
        pts_prev = np.float32([self._last_keypoints[m.trainIdx].pt for m in matches])

        pose = self._estimate_pose(pts_curr, pts_prev)
        if pose is None:
            return None
        R_rel, t_rel = pose

        scale = self._estimate_scale(detections, self._last_detections)
        self._t += self._R @ (t_rel * scale)
        self._R = R_rel @ self._R
        self._trajectory.append(self._t.copy())

        state = FrameState(
            timestamp=ts,
            rotation=self._R.copy(),
            translation=self._t.copy(),
            keypoints=keypoints,
            descriptors=descriptors,
            detections=detections,
        )
        self._store_state(state)

        self._last_keypoints = keypoints
        self._last_descriptors = descriptors
        self._last_detections = detections
        return state

    def _match_descriptors(
        self,
        descriptors_curr: np.ndarray,
        descriptors_prev: np.ndarray,
    ) -> List[cv2.DMatch]:
        matches = self._bf.match(descriptors_curr, descriptors_prev)
        matches.sort(key=lambda m: m.distance)
        return matches[: self._match_keep]

    def _estimate_pose(
        self,
        pts_curr: np.ndarray,
        pts_prev: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        E, mask = cv2.findEssentialMat(
            pts_curr,
            pts_prev,
            self._K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        if E is None:
            return None

        inlier_count = int(mask.ravel().sum()) if mask is not None else 0
        if inlier_count < 8:
            return None

        _, R, t, _ = cv2.recoverPose(E, pts_curr, pts_prev, self._K)
        return R, t

    def _estimate_scale(
        self,
        curr: Iterable[Detection],
        prev: Iterable[Detection],
    ) -> float:
        """
        Infer a relative scale factor using detection areas.

        The heuristic assumes that objects retain similar real-world dimensions,
        so changes in bounding box area correlate with relative depth changes.
        """
        prev_by_label: Dict[str, Detection] = {det.label: det for det in prev}
        ratios: List[float] = []
        for detection in curr:
            if detection.label not in prev_by_label:
                continue
            prev_det = prev_by_label[detection.label]
            if prev_det.area() <= 0 or detection.area() <= 0:
                continue
            ratio = np.sqrt(prev_det.area() / detection.area())
            ratios.append(float(ratio))

        if not ratios:
            return 1.0
        return float(np.clip(np.median(ratios), 0.25, 4.0))

    def _run_yolo(self, frame: np.ndarray) -> List[Detection]:
        results = self._model(frame, conf=self._confidence, verbose=False)
        detections: List[Detection] = []

        for result in results:
            if not hasattr(result, "boxes"):
                continue
            names = result.names
            for box in result.boxes:
                cls_idx = int(box.cls.item())
                label = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                detections.append(Detection(label=label, confidence=confidence, bbox=(x1, y1, x2, y2)))
        return detections

    def _draw_annotations(self, frame: np.ndarray, state: FrameState) -> None:
        for det in state.detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                frame,
                f"{det.label} {det.confidence:.2f}",
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        for kp in state.keypoints[:200]:
            x, y = map(int, kp.pt)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        pose_text = "Pos: [{:.2f}, {:.2f}, {:.2f}]".format(*self._t.ravel())
        cv2.putText(
            frame,
            pose_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _store_state(self, state: FrameState) -> None:
        self._states.append(state)
        if len(self._states) > 500:
            self._states.pop(0)

