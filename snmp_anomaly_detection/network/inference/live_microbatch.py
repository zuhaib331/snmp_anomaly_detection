from __future__ import annotations

import time
from dataclasses import dataclass

from snmp_anomaly_detection.network.inference.event_processor import (
    EventProcessor,
    PreparedWindow,
    ProcessedWindow,
)
from snmp_anomaly_detection.network.inference.events import NormalizedEvent


@dataclass(frozen=True)
class MicroBatchConfig:
    batch_size: int = 8
    max_wait_ms: int = 50


class LiveMicroBatchProcessor:
    def __init__(
        self,
        event_processor: EventProcessor,
        config: MicroBatchConfig | None = None,
    ):
        self.event_processor = event_processor
        self.config = config or MicroBatchConfig()
        self._pending: list[PreparedWindow] = []
        self._first_pending_at: float | None = None

    def handle_event(
        self,
        event: NormalizedEvent,
        now_monotonic: float | None = None,
    ) -> list[ProcessedWindow]:
        prepared_window = self.event_processor.prepare_event(event)
        if prepared_window is None:
            return self._flush_if_due(now_monotonic=now_monotonic)

        return self.add_prepared_window(
            prepared_window=prepared_window,
            now_monotonic=now_monotonic,
        )

    def add_prepared_window(
        self,
        prepared_window: PreparedWindow,
        now_monotonic: float | None = None,
    ) -> list[ProcessedWindow]:
        now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic

        if not self._pending:
            self._first_pending_at = now_monotonic

        self._pending.append(prepared_window)

        if len(self._pending) >= self.config.batch_size:
            return self.flush()

        return self._flush_if_due(now_monotonic=now_monotonic)

    def poll(self, now_monotonic: float | None = None) -> list[ProcessedWindow]:
        return self._flush_if_due(now_monotonic=now_monotonic)

    def flush(self) -> list[ProcessedWindow]:
        if not self._pending:
            return []

        pending = self._pending
        self._pending = []
        self._first_pending_at = None
        return self.event_processor.process_prepared_windows(pending)

    def _flush_if_due(self, now_monotonic: float | None = None) -> list[ProcessedWindow]:
        if not self._pending or self._first_pending_at is None:
            return []

        now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
        elapsed_ms = (now_monotonic - self._first_pending_at) * 1000.0
        if elapsed_ms >= self.config.max_wait_ms:
            return self.flush()

        return []
