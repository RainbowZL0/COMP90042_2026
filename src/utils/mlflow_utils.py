"""Small optional MLflow wrapper.

The project should run even when mlflow is not installed, so all helpers degrade
cleanly to no-ops. Install with ``uv add mlflow`` when experiment tracking is
wanted.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class MLflowRun:
    """Thin no-op-safe wrapper around an optional active MLflow run."""

    def __init__(self, enabled: bool, mlflow_module: Any | None = None):
        self.enabled = enabled
        self._mlflow = mlflow_module

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.enabled or self._mlflow is None:
            return
        clean = {k: _clean_value(v) for k, v in params.items() if _clean_value(v) is not None}
        if clean:
            self._mlflow.log_params(clean)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or self._mlflow is None:
            return
        clean: dict[str, float] = {}
        for k, v in metrics.items():
            try:
                clean[k] = float(v)
            except (TypeError, ValueError):
                continue
        if clean:
            self._mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: str | Path) -> None:
        if not self.enabled or self._mlflow is None:
            return
        path = Path(path)
        if path.exists():
            self._mlflow.log_artifact(str(path))

    def set_tags(self, tags: dict[str, Any]) -> None:
        if not self.enabled or self._mlflow is None:
            return
        clean = {k: str(v) for k, v in tags.items() if v is not None}
        if clean:
            self._mlflow.set_tags(clean)


def _clean_value(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@contextmanager
def mlflow_run(
    enabled: bool,
    experiment_name: str,
    run_name: str | None = None,
    tracking_uri: str | Path = "outputs/mlruns",
    tags: dict[str, Any] | None = None,
) -> Iterator[MLflowRun]:
    """Start an MLflow run if enabled and mlflow is installed.

    If mlflow import fails, the context silently becomes a no-op. This avoids
    forcing mlflow in Colab or test environments.
    """
    if not enabled:
        yield MLflowRun(False)
        return

    try:
        import mlflow  # type: ignore
    except ImportError:
        print("[mlflow] mlflow is not installed; run `uv add mlflow` to enable tracking.")
        yield MLflowRun(False)
        return

    uri = str(tracking_uri)
    if not (uri.startswith("http://") or uri.startswith("https://") or uri.startswith("file:")):
        Path(uri).mkdir(parents=True, exist_ok=True)
        uri = Path(uri).resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        wrapper = MLflowRun(True, mlflow)
        if tags:
            wrapper.set_tags(tags)
        yield wrapper
