from __future__ import annotations

import json
import logging
import math
import pathlib
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import librosa
except ImportError as exc:
    raise ImportError(
        "librosa is required for music-based drone choreography. "
        "Install it with `pip install librosa`."
    ) from exc

try:
    from .tello import Tello
except ImportError:
    Tello = "Tello"

logger = logging.getLogger(__name__)


@dataclass
class BeatEvent:
    """Represents a single detected beat in an audio track."""

    time: float
    tempo: float
    strength: float


@dataclass
class ChoreographyStep:
    """Represents a planned drone command at a specific timestamp."""

    time: float
    command: str
    args: Tuple[Union[int, float, str], ...] = ()
    kwargs: dict = None

    def __post_init__(self) -> None:
        if self.kwargs is None:
            self.kwargs = {}


def normalise_strengths(strengths: np.ndarray) -> np.ndarray:
    if strengths.size == 0:
        return strengths
    min_val = float(np.min(strengths))
    max_val = float(np.max(strengths))
    if math.isclose(max_val, min_val):
        return np.ones_like(strengths)
    return (strengths - min_val) / (max_val - min_val)


class InvalidAudioError(RuntimeError):
    """Raised when audio loading or analysis fails."""


class MusicChoreographer:
    """
    Analyse music and choreograph drone movements.

    Parameters
    ----------
    tello:
        Optional connected `Tello` instance. One can also pass frames to `play_schedule`
        later by setting `preview_only=True`.
    preload_audio:
        When True, audio data is loaded during initialisation; otherwise `load_audio`
        must be called before scheduling.
    command_latency:
        Seconds to send commands in advance to compensate for network latency.
    default_step_distance:
        Distance in centimetres used for simple move commands.
    default_vertical_distance:
        Vertical distance (cm) used in up/down commands.
    """

    def __init__(
        self,
        tello: Optional[Tello] = None,
        *,
        audio_path: Optional[Union[str, pathlib.Path]] = None,
        preload_audio: bool = True,
        command_latency: float = 0.25,
        default_step_distance: int = 40,
        default_vertical_distance: int = 30,
    ) -> None:
        self._tello = tello
        self._audio: Optional[np.ndarray] = None
        self._sr: Optional[int] = None
        self._beats: List[BeatEvent] = []
        self._schedule: List[ChoreographyStep] = []
        self.command_latency = command_latency
        self.default_step_distance = default_step_distance
        self.default_vertical_distance = default_vertical_distance
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if audio_path is not None and preload_audio:
            self.load_audio(audio_path)

    def load_audio(
        self,
        source: Union[str, pathlib.Path, np.ndarray],
        *,
        sample_rate: Optional[int] = None,
        mono: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio into memory for subsequent analysis.
        """
        try:
            if isinstance(source, np.ndarray):
                audio = np.copy(source)
                sr = sample_rate or 22050
            else:
                audio, sr = librosa.load(path=source, sr=sample_rate, mono=mono)
        except Exception as exc:
            raise InvalidAudioError(f"Failed to load audio: {exc}") from exc

        self._audio = audio
        self._sr = int(sr)
        logger.debug("Loaded audio with %d samples @ %d Hz", len(audio), sr)
        return audio, sr

    def detect_beats(
        self,
        *,
        tightness: float = 0.8,
        hop_length: int = 512,
        start_bpm: Optional[float] = None,
    ) -> List[BeatEvent]:
        """
        Perform beat tracking and compute relative strengths.
        """
        if self._audio is None or self._sr is None:
            raise InvalidAudioError("Audio must be loaded before beat detection.")

        audio = self._audio
        sr = self._sr

        tempo, beats = librosa.beat.beat_track(
            y=audio,
            sr=sr,
            tightness=tightness,
            hop_length=hop_length,
            start_bpm=start_bpm,
        )

        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
        beat_strengths = onset_env[beats] if beats.size else np.array([])
        beat_strengths = normalise_strengths(beat_strengths)

        times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
        bpm = float(tempo)

        events = [
            BeatEvent(time=float(t), tempo=bpm, strength=float(s)) for t, s in zip(times, beat_strengths)
        ]
        self._beats = events
        logger.info("Detected %d beats @ %.2f BPM", len(events), bpm)
        return events

    
    BEAT_COMMANDS: Sequence[str] = (
        "move_forward",
        "move_right",
        "move_left",
        "move_back",
        "move_up",
        "move_down",
        "rotate_clockwise",
        "rotate_counter_clockwise",
    )

    def build_schedule(
        self,
        *,
        include_takeoff_land: bool = True,
        power_moves: Sequence[str] = ("flip_forward", "flip_left", "flip_right"),
        strength_threshold: float = 0.75,
        tempo_divider: int = 1,
    ) -> List[ChoreographyStep]:
        """
        Map beat events to drone commands.
        """
        if not self._beats:
            raise RuntimeError("No beats detected. Call detect_beats() first.")

        schedule: List[ChoreographyStep] = []
        beats = self._beats[:: max(1, tempo_divider)]

        if include_takeoff_land:
            schedule.append(ChoreographyStep(time=0.0, command="takeoff"))

        axis = 0
        for event in beats:
            command = self.BEAT_COMMANDS[axis % len(self.BEAT_COMMANDS)]
            axis += 1

            args: Tuple[Union[int, float, str], ...]
            if "move" in command:
                distance = self.default_step_distance
                args = (distance,)
            elif "rotate" in command:
                args = (90,)
            else:
                args = ()

            if event.strength >= strength_threshold and power_moves:
                power_command = power_moves[axis % len(power_moves)]
                schedule.append(ChoreographyStep(time=event.time, command=power_command))
            else:
                schedule.append(ChoreographyStep(time=event.time, command=command, args=args))

        if include_takeoff_land:
            last_time = schedule[-1].time + 2.0 if schedule else 4.0
            schedule.append(ChoreographyStep(time=last_time, command="land"))

        self._schedule = schedule
        logger.info("Built schedule with %d steps", len(schedule))
        return schedule

    def play_schedule(
        self,
        *,
        preview_only: bool = False,
        on_step: Optional[Callable[[ChoreographyStep], None]] = None,
    ) -> None:
        """
        Execute the planned choreography.
        """
        if not self._schedule:
            raise RuntimeError("No choreography steps available. Call build_schedule().")

        if preview_only:
            for step in self._schedule:
                logger.info("[Preview] %s at %.2fs -> %s%s", step.command, step.time, step.args, step.kwargs)
            return

        if self._tello is None:
            raise RuntimeError("Tello instance required to play choreography.")

        self._stop_event.clear()
        self._playback_thread = threading.Thread(
            target=self._run_schedule, kwargs={"on_step": on_step}, daemon=True
        )
        self._playback_thread.start()

    def stop(self) -> None:
        """
        Request the scheduler to stop. The current command may still finish.
        """
        self._stop_event.set()
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=3.0)

    def export_schedule(self) -> List[dict]:
        """
        Serialize the schedule to a list of dictionaries.
        """
        return [
            {"time": step.time, "command": step.command, "args": list(step.args), "kwargs": step.kwargs}
            for step in self._schedule
        ]

    def save_schedule(self, path: Union[str, pathlib.Path]) -> None:
        """
        Save schedule to JSON for later replay.
        """
        data = self.export_schedule()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _run_schedule(self, *, on_step: Optional[Callable[[ChoreographyStep], None]] = None) -> None:
        start_time = time.time()
        idx = 0
        schedule = list(self._schedule)

        while idx < len(schedule) and not self._stop_event.is_set():
            step = schedule[idx]
            target_time = start_time + max(0.0, step.time - self.command_latency)
            now = time.time()
            sleep_time = target_time - now

            if sleep_time > 0:
                time.sleep(sleep_time)

            if self._stop_event.is_set():
                break

            if on_step is not None:
                try:
                    on_step(step)
                except Exception as exc:
                    logger.warning("on_step callback failed: %s", exc)

            self._execute_step(step)
            idx += 1

        logger.info("Finished playing choreography with %d steps", idx)

    def _execute_step(self, step: ChoreographyStep) -> None:
        tello = self._tello
        if tello is None:
            return

        command_fn = getattr(tello, step.command, None)
        if command_fn is None:
            logger.warning("Unknown Tello command: %s", step.command)
            return

        try:
            command_fn(*step.args, **step.kwargs)
        except Exception as exc:
            logger.error("Failed to execute %s: %s", step.command, exc)


