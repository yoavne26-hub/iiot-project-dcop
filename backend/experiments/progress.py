"""Terminal progress helpers for experiment runs."""

from __future__ import annotations

from types import TracebackType


class ExperimentProgress:
    """Small wrapper around Rich progress bars with a no-op mode."""

    def __init__(
        self,
        enabled: bool,
        total_problems: int,
        total_iterations: int,
    ) -> None:
        self.enabled = enabled
        self.total_problems = total_problems
        self.total_iterations = total_iterations
        self._progress = None
        self._problems_task = None
        self._run_task = None

    def __enter__(self) -> "ExperimentProgress":
        if not self.enabled:
            return self

        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )
        except ImportError as error:
            raise RuntimeError(
                "Progress display requires rich. Install dependencies with "
                "`pip install -r requirements.txt`, or pass --no-progress."
            ) from error

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self._progress.start()
        self._problems_task = self._progress.add_task(
            "Problems",
            total=self.total_problems,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._progress is not None:
            self._progress.stop()

    def start_run(self, simulator: str, algorithm: str, problem_index: int) -> None:
        """Create or reset the current algorithm/simulator run task."""

        if self._progress is None:
            return

        description = f"{simulator} / {algorithm}  problem {problem_index + 1}"
        if self._run_task is None:
            self._run_task = self._progress.add_task(
                description,
                total=self.total_iterations,
            )
        else:
            self._progress.reset(
                self._run_task,
                total=self.total_iterations,
                completed=0,
                description=description,
            )
            self._progress.start_task(self._run_task)

    def update_run(self, completed_iterations: int) -> None:
        """Update the current run task to an absolute completed iteration count."""

        if self._progress is None or self._run_task is None:
            return

        completed = min(completed_iterations, self.total_iterations)
        self._progress.update(self._run_task, completed=completed)

    def finish_run(self) -> None:
        """Mark the current run task complete."""

        if self._progress is None or self._run_task is None:
            return

        self._progress.update(self._run_task, completed=self.total_iterations)
        self._progress.stop_task(self._run_task)

    def finish_problem(self) -> None:
        """Advance the overall problem progress by one completed problem."""

        if self._progress is None or self._problems_task is None:
            return

        self._progress.advance(self._problems_task, 1)
