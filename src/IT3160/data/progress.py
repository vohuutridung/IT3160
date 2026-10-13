"""Small, flushed console messages; periodic updates are limited to once per 30s."""

from datetime import datetime
import sys
from time import monotonic


def log(message: str) -> None:
    """Keep progress on stderr so command results on stdout remain valid JSON."""
    print(f"[{datetime.now():%H:%M:%S}] {message}", file=sys.stderr, flush=True)


class Progress:
    def __init__(self, label: str, unit: str):
        self.label = label
        self.unit = unit
        self.count = 0
        self.started = self.last_update = monotonic()
        log(f"{label}: started")

    def update(self, count: int) -> None:
        self.count = count
        now = monotonic()
        if now - self.last_update >= 30:
            log(f"{self.label}: {count:,} {self.unit}")
            self.last_update = now

    def finish(self, detail: str = "") -> None:
        suffix = f"; {detail}" if detail else ""
        log(f"{self.label}: done, {self.count:,} {self.unit} "
            f"in {monotonic() - self.started:.1f}s{suffix}")
