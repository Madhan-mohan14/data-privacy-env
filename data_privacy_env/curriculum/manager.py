import os
import threading


class CurriculumManager:
    ESCALATE_THRESHOLD = 0.75
    DEESCALATE_THRESHOLD = 0.35
    WINDOW = 10
    MAX_LEVEL = 4
    # (start_episode, end_episode, level)
    TRAINING_SCHEDULE = [(0, 200, 1), (200, 600, 2), (600, 1400, 3), (1400, 9999, 4)]

    def __init__(self, demo: bool | None = None):
        if demo is None:
            demo = os.environ.get("CURRICULUM_TRAINING", "0") != "1"
        self.demo = demo
        self._history: list[float] = []
        self.current_level = 1
        self._episode_count = 0
        self._lock = threading.Lock()

    def record_episode(self, reward: float) -> None:
        with self._lock:
            self._episode_count += 1
            if self.demo:
                self._history.append(reward)
                recent = self._history[-self.WINDOW :]
                if len(recent) >= self.WINDOW:
                    avg = sum(recent) / len(recent)
                    if avg > self.ESCALATE_THRESHOLD and self.current_level < self.MAX_LEVEL:
                        self.current_level += 1
                    elif avg < self.DEESCALATE_THRESHOLD and self.current_level > 1:
                        self.current_level -= 1

    def get_level(self) -> int:
        with self._lock:
            if self.demo:
                return self.current_level
            for start, end, level in self.TRAINING_SCHEDULE:
                if start <= self._episode_count < end:
                    return level
            return self.MAX_LEVEL
