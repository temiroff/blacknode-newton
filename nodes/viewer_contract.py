"""Small provider contract between Newton sessions and visual front ends."""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Protocol


class ViewerHandle(Protocol):
    """Runtime surface used by the simulation loop."""

    @property
    def url(self) -> str: ...

    def is_running(self) -> bool: ...

    def begin_frame(self, time_seconds: float) -> None: ...

    def log_state(self, state: Any) -> None: ...

    def log_reference_state(self, state: Any | None, options: dict[str, Any]) -> None: ...

    def end_frame(self) -> None: ...

    def set_visibility(self, path: str, visible: bool) -> bool: ...

    def set_grid(self, visible: bool) -> bool: ...

    def set_render_options(self, show_visuals: bool, show_colliders: bool) -> bool: ...

    def set_transform(self, path: str, transform: dict[str, Any]) -> bool: ...

    def set_material(self, path: str, material_path: str, material: dict[str, Any]) -> bool: ...

    def set_environment(self, environment: dict[str, Any]) -> bool: ...

    def close(self) -> None: ...


ViewerFactory = Callable[[Any, Any, dict[str, Any]], ViewerHandle]

_LOCK = threading.RLock()
_FACTORIES: dict[str, ViewerFactory] = {}


def register_viewer(name: str, factory: ViewerFactory) -> None:
    """Register a concrete viewer without coupling it to the physics runtime."""
    key = str(name or "").strip().lower()
    if not key:
        raise ValueError("viewer provider name is required")
    with _LOCK:
        _FACTORIES[key] = factory


def create_viewer(name: str, session: Any, model: Any, config: dict[str, Any]) -> ViewerHandle:
    key = str(name or "").strip().lower()
    with _LOCK:
        factory = _FACTORIES.get(key)
        available = sorted(_FACTORIES)
    if factory is None:
        detail = ", ".join(available) if available else "none"
        raise RuntimeError(f"viewer provider '{key}' is unavailable; loaded providers: {detail}")
    return factory(session, model, config)


def available_viewers() -> list[str]:
    with _LOCK:
        return sorted(_FACTORIES)
