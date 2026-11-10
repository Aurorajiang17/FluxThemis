from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from fluxthemis.structures import StepResult, TaskSpec


class EnvironmentCaseRejected(RuntimeError):
    """Signal that an environment case cannot start an episode."""


class NonFiniteObservationError(ValueError):
    """Nonfinite simulator state, distinct from shape errors."""

    def __init__(self, field, lane, shape, invalid_indices):
        super().__init__(f'{field} contains nonfinite values in lane {lane}')
        self.details = dict(
            field=field,
            lane=int(lane),
            shape=list(shape),
            invalid_indices=invalid_indices)
        self.prediction_metadata = []
        self.completed_results = []


class BaseEnvironment(ABC):
    """Contract implemented by simulation and real-robot adapters."""

    supports_native_video_recording = False
    batch_size = 1

    def reset_batch(
        self, task: TaskSpec, episode_indices: Sequence[int], *,
        environment_seeds: Sequence[int | None],
        environment_case_indices: Sequence[int | None]
    ) -> Sequence[tuple[Any, dict]]:
        """Reset all lanes of one native batch, in lane order.

        Batched execution is opt-in. It must preserve separate observations,
        terminal states and recordings for each scheduled episode.
        """
        raise NotImplementedError(
            'Environment does not support native batches')

    def step_batch(self, actions: dict[int, Any]) -> dict[int, StepResult]:
        """Advance active lanes once; return exactly their transitions."""
        raise NotImplementedError(
            'Environment does not support native batches')

    def configure_worker(self, worker_id: str, run_id: str) -> None:
        del worker_id, run_id

    def configure_simulator_device(self, device_id: str) -> None:
        del device_id

    def configure_video_recording(self, enabled: bool,
                                  output_dir: str | None) -> None:
        del enabled, output_dir

    def setup(self) -> None:
        """Allocate simulator or hardware resources."""

    @abstractmethod
    def tasks(self) -> Sequence[TaskSpec]:
        """Return the tasks available for evaluation."""

    @abstractmethod
    def reset(self, task: TaskSpec, episode_index: int, *,
              environment_seed: int | None,
              environment_case_index: int | None) -> tuple[Any, dict]:
        """Reset one episode and return ``(observation, info)``."""

    @abstractmethod
    def step(self, action: Any) -> StepResult:
        """Execute one environment-unit action."""

    def render(self) -> Any | None:
        return None

    def close(self) -> None:
        """Release resources. Implementations should make this idempotent."""
