# Kafka Test Data Generator

This document describes the `produce-kafka-test-data` command used for Kafka testing.

It is intentionally separate from the Phase 4 live Kafka implementation plan.

## Purpose

Use this command to generate fresh synthetic SNMP-like events continuously and publish them to Kafka.

This helps with:
- testing Kafka connectivity
- testing message schema compatibility
- testing mixed-device live traffic
- testing anomaly injection behavior

## Scope

This command is for test data generation only.

It does:
- generate new events in memory
- simulate multiple devices on one shared topic
- publish messages continuously or for a fixed count
- inject anomalies with a configurable probability

It does not:
- read from CSV
- train the model
- replace the Phase 4 live detection plan
- run anomaly scoring by itself

## Kafka Topic and Schema

The producer sends to the same hardcoded topic used by the current Kafka dry consumer:

- topic: `snmp-live-events`

Each message contains:
- `timestamp`
- `device_id`
- `cpu`
- `memory`
- `in_octets`
- `out_octets`
- `errors`
- `anomaly`

This schema matches what the Kafka consumer normalizes today.

## Current Consumer Compatibility

The generated messages can be consumed by:

- `python3 main.py detect-kafka-dry`

That command will:
- consume the Kafka messages
- decode JSON
- normalize the fields into the shared event shape

Current limitation:
- `detect-kafka-dry` does not yet run full anomaly scoring

## How to Run

Start the dry Kafka consumer in one terminal:

```bash
python3 main.py detect-kafka-dry
```

Start the test-data generator in another terminal:

```bash
python3 main.py produce-kafka-test-data
```

Default behavior:
- continuous streaming
- 15 devices
- 0.1 second delay between messages
- 2% anomaly probability

Stop either process with `Ctrl+C`.

## Main Options

### Control device count

```bash
python3 main.py produce-kafka-test-data --device-count 20
```

This simulates 20 devices publishing into the same topic.

### Control message rate

Use `--sleep-seconds` to control how quickly events are sent.

Examples:
- `--sleep-seconds 1.0` about 1 message per second
- `--sleep-seconds 0.1` about 10 messages per second
- `--sleep-seconds 0.05` about 20 messages per second

Example:

```bash
python3 main.py produce-kafka-test-data --sleep-seconds 0.05
```

### Control anomaly ratio

Use `--anomaly-probability` to control how often anomalous events are generated.

Examples:
- `--anomaly-probability 0.01` about 1%
- `--anomaly-probability 0.05` about 5%
- `--anomaly-probability 0.10` about 10%

Example:

```bash
python3 main.py produce-kafka-test-data --anomaly-probability 0.10
```

### Limit the run

Use `--max-messages` when you want a short controlled test.

Example:

```bash
python3 main.py produce-kafka-test-data --max-messages 100
```

### Reproducible stream

Use `--seed` to generate repeatable random behavior during testing.

Example:

```bash
python3 main.py produce-kafka-test-data --seed 7
```

## Example Test Runs

### Normal live traffic

```bash
python3 main.py produce-kafka-test-data --device-count 15 --sleep-seconds 0.1 --anomaly-probability 0.02
```

### Faster stream with more anomalies

```bash
python3 main.py produce-kafka-test-data --device-count 20 --sleep-seconds 0.05 --anomaly-probability 0.10
```

### Short smoke test

```bash
python3 main.py produce-kafka-test-data --max-messages 50 --sleep-seconds 0.1 --anomaly-probability 0.05
```

## Notes

- The producer is intended for testing and simulation.
- The command currently targets one hardcoded Kafka topic.
- The current live consumer path available in CLI is `detect-kafka-dry`.
- Full Kafka-driven anomaly scoring can be added later as a separate step.
