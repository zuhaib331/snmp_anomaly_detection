# Live Micro-Batching

This document describes the reusable micro-batch scoring layer added before Kafka integration.

## Objective

Support low-latency live event scoring without calling the model for every single ready window.

## Behavior

- Events are still ingested one by one.
- Each event is normalized and passed to `EventProcessor`.
- When a device window becomes ready, it is added to a short-lived micro-batch queue.
- The queue flushes when either:
  - `batch_size` is reached
  - `max_wait_ms` has elapsed since the first queued window

## Current Defaults

- `batch_size = 8`
- `max_wait_ms = 50`

## Why This Helps

- keeps latency low for critical live events
- avoids one model call per ready window
- matches the intended Kafka design without implementing Kafka yet

## Current Scope

- reusable processor only
- no Kafka consumer or topic integration yet
- no CLI entry point yet
