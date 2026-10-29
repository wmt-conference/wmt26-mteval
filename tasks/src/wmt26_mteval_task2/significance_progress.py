"""Parent-only live progress for parallel, checkpointed permutation draws.

Only committed results count as completed. Throughput uses new draws divided
by this session's wall time, never cached work or summed worker CPU time.
The display uses the standard library and works without extra dependencies.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import TextIO


def atomic_json(path: Path, value: dict) -> None:
    """Replace a JSON snapshot atomically, keeping the old file on write failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def duration(seconds: float | None) -> str:
    """Format an estimated or elapsed duration without claiming false precision."""
    if seconds is None or not math.isfinite(seconds):
        return "estimating"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class PermutationProgress:
    """Render a compact dashboard and write a machine-readable live snapshot."""

    def __init__(self, *, total_draws: int, cached_draws: int, total_pairs: int,
                 completed_pairs: int, total_chunks: int, cached_chunks: int,
                 workers: int, mode: str = "auto", interval: float = 1.0,
                 path: Path | None = None, stream: TextIO | None = None):
        """Initialize counters; interactive mode is selected only for a terminal."""
        if mode not in ("auto", "plain", "off") or not math.isfinite(interval) or interval <= 0:
            raise ValueError("Invalid progress mode or refresh interval")
        if not 0 <= cached_draws <= total_draws or total_draws < 1:
            raise ValueError("Invalid progress draw counts")
        self.total_draws = total_draws
        self.cached_draws = cached_draws
        self.new_draws = 0
        self.total_pairs = total_pairs
        self.completed_pairs = completed_pairs
        self.total_chunks = total_chunks
        self.cached_chunks = cached_chunks
        self.new_chunks = 0
        self.workers = workers
        self.interval = interval
        self.path = path
        self.stream = stream if stream is not None else sys.stderr
        self.interactive = mode == "auto" and self.stream.isatty() and os.environ.get("TERM") != "dumb"
        self.mode = mode
        self.started = time.monotonic()
        self.last_commit = None
        self.last_refresh = -math.inf
        self.last_log = -math.inf
        self.lines = 0

    def advance(self, draws: int, *, pair_completed: bool) -> None:
        """Credit a chunk only after the parent has committed it to SQLite."""
        if draws < 1 or self.cached_draws + self.new_draws + draws > self.total_draws:
            raise ValueError("Progress increment exceeds requested work")
        self.new_draws += draws
        self.new_chunks += 1
        self.completed_pairs += int(pair_completed)
        self.last_commit = time.monotonic()

    def snapshot(self, *, in_flight: int = 0, status: str = "running") -> dict:
        """Return exact saved counts and a wall-clock, worker-aggregated ETA."""
        elapsed = max(0., time.monotonic() - self.started)
        completed = self.cached_draws + self.new_draws
        remaining = self.total_draws - completed
        rate = self.new_draws / elapsed if elapsed > 0 and self.new_draws else None
        eta = remaining / rate if rate else None
        if not remaining:
            eta = 0.
        finish = None
        if eta is not None and eta < 86400 * 365:
            finish = (datetime.now().astimezone() + timedelta(seconds=eta)).isoformat(timespec="seconds")
        return dict(status=status, updated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                    total_draws=self.total_draws, completed_draws=completed,
                    remaining_draws=remaining, cached_draws=self.cached_draws,
                    new_draws=self.new_draws, percent=100 * completed / self.total_draws,
                    completed_pairs=self.completed_pairs, total_pairs=self.total_pairs,
                    completed_chunks=self.cached_chunks + self.new_chunks, total_chunks=self.total_chunks,
                    workers=self.workers, in_flight_chunks=in_flight,
                    elapsed_session_seconds=elapsed, new_draws_per_second=rate,
                    estimated_remaining_seconds=eta, estimated_finish=finish,
                    seconds_since_checkpoint=(time.monotonic()-self.last_commit if self.last_commit else None))

    def render_lines(self, state: dict, width: int) -> list[str]:
        """Build a width-aware dashboard; no worker writes to the terminal."""
        width = max(30, width)
        bar_width = max(8, min(48, width - 28))
        filled = int(bar_width * state["completed_draws"] / state["total_draws"])
        bar = "#" * filled + "." * (bar_width - filled)
        rate = state["new_draws_per_second"]
        speed = f"{rate:,.1f} draws/s" if rate is not None else "estimating speed"
        finish = state["estimated_finish"] or "estimating"
        lines = [f"TASK 2 SIGNIFICANCE | {state['status']} | {self.workers} workers",
                 f"[{bar}] {state['percent']:6.2f}%",
                 f"Saved draws {state['completed_draws']:,}/{self.total_draws:,} | Left {state['remaining_draws']:,}",
                 f"Evaluator pairs {state['completed_pairs']}/{self.total_pairs} | Chunks {state['completed_chunks']:,}/{self.total_chunks:,}",
                 f"Elapsed {duration(state['elapsed_session_seconds'])} | Remaining {duration(state['estimated_remaining_seconds'])}",
                 f"{speed} | Expected finish {finish}",
                 f"Resumed {self.cached_draws:,} | New {self.new_draws:,} | In flight {state['in_flight_chunks']}"]
        return [line if len(line) <= width else line[:width-3]+"..." for line in lines]

    def refresh(self, *, in_flight: int = 0, status: str = "running", force: bool = False) -> dict:
        """Refresh elapsed time even when no worker has completed a chunk yet."""
        now = time.monotonic()
        state = self.snapshot(in_flight=in_flight, status=status)
        if not force and now-self.last_refresh < self.interval:
            return state
        self.last_refresh = now
        if self.path is not None:
            atomic_json(self.path, state)
        if self.mode == "off":
            return state
        if self.interactive:
            width = max(30, shutil.get_terminal_size((100, 24)).columns - 1)
            lines = self.render_lines(state, width)
            if self.lines:
                self.stream.write(f"\033[{self.lines}F")
            color = not os.environ.get("NO_COLOR")
            for index, line in enumerate(lines):
                text = f"\033[1;36m{line}\033[0m" if color and index < 2 else line
                self.stream.write("\033[2K" + text + "\n")
            self.lines = len(lines)
        elif force or now-self.last_log >= max(30., self.interval):
            self.stream.write(
                f"Task 2 {status}: {state['percent']:.2f}% | draws {state['completed_draws']:,}/{self.total_draws:,} "
                f"| pairs {self.completed_pairs}/{self.total_pairs} | cached {self.cached_draws:,} "
                f"| workers {self.workers}, in flight {in_flight} "
                f"| elapsed {duration(state['elapsed_session_seconds'])} "
                f"| remaining {duration(state['estimated_remaining_seconds'])}\n")
            self.last_log = now
        self.stream.flush()
        return state
