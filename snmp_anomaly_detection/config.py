from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Power SNMP feature column definitions
# ---------------------------------------------------------------------------

# Aggregated UPS health metrics (baseline model input) — canonical names from OID mapping doc
BASELINE_UPS_FEATURES: tuple[str, ...] = (
    "battery_charge_pct",
    "battery_voltage_v",
    "battery_current_a",
    "battery_temperature_c",
    "runtime_remaining_min",
    "on_battery_status",
    "battery_replace_status",
    "input_voltage_v",
    "input_frequency_hz",
    "output_voltage_v",
    "output_current_a",
    "output_load_pct",
    "output_frequency_hz",
    "output_power_w",
    # Rate-of-change features — capture slow-draining signals within a window
    "runtime_delta",
    "battery_charge_delta",
)

# Per-phase raw metrics (phase-level model input) — superset of BASELINE_UPS_FEATURES
PHASE_LEVEL_FEATURES: tuple[str, ...] = BASELINE_UPS_FEATURES + (
    "input_voltage_l1",
    "input_voltage_l2",
    "input_voltage_l3",
    "input_current_l1",
    "input_current_l2",
    "input_current_l3",
    "output_current_l1",
    "output_current_l2",
    "output_current_l3",
    "voltage_imbalance_pct",
    "current_skew_pct",
)

# Features used for battery RUL forecasting
BATTERY_RUL_FEATURES: tuple[str, ...] = (
    "battery_charge_pct",
    "battery_voltage_v",
    "battery_current_a",
    "battery_temperature_c",
    "runtime_remaining_min",
    "charge_rate",
    "discharge_cycles_approx",
)


@dataclass(frozen=True)
class ProjectPaths:
    package_root: Path = PACKAGE_ROOT
    repo_root: Path = field(default_factory=lambda: PACKAGE_ROOT.parent)
    data_dir: Path = field(default_factory=lambda: PACKAGE_ROOT / "data")
    outputs_dir: Path = field(default_factory=lambda: PACKAGE_ROOT / "outputs")
    utils_dir: Path = field(default_factory=lambda: PACKAGE_ROOT / "utils")
    dataset_file: Path = field(
        default_factory=lambda: PACKAGE_ROOT / "data" / "synthetic_snmp_dataset.csv"
    )
    scaler_file: Path = field(default_factory=lambda: PACKAGE_ROOT / "utils" / "scaler.pkl")
    x_train_file: Path = field(default_factory=lambda: PACKAGE_ROOT / "outputs" / "X_train.npy")
    x_test_file: Path = field(default_factory=lambda: PACKAGE_ROOT / "outputs" / "X_test.npy")
    y_train_file: Path = field(default_factory=lambda: PACKAGE_ROOT / "outputs" / "y_train.npy")
    y_test_file: Path = field(default_factory=lambda: PACKAGE_ROOT / "outputs" / "y_test.npy")
    model_file: Path = field(
        default_factory=lambda: PACKAGE_ROOT / "outputs" / "lstm_autoencoder.pth"
    )
    model_metadata_file: Path = field(
        default_factory=lambda: PACKAGE_ROOT / "outputs" / "model_metadata.json"
    )
    anomaly_results_file: Path = field(
        default_factory=lambda: PACKAGE_ROOT / "outputs" / "anomaly_results.csv"
    )
    anomaly_windows_file: Path = field(
        default_factory=lambda: PACKAGE_ROOT / "outputs" / "anomaly_windows.json"
    )
    kafka_live_results_file: Path = field(
        default_factory=lambda: PACKAGE_ROOT / "outputs" / "kafka_live_results.jsonl"
    )
    kafka_live_anomaly_windows_file: Path = field(
        default_factory=lambda: PACKAGE_ROOT / "outputs" / "kafka_live_anomaly_windows.jsonl"
    )

    power_dataset_file: Path = field(
        default_factory=lambda: PACKAGE_ROOT / "data" / "synthetic_power_snmp_dataset.csv"
    )
    power_outputs_dir: Path = field(
        default_factory=lambda: PACKAGE_ROOT / "outputs" / "power_baseline"
    )
    power_phase_outputs_dir: Path = field(
        default_factory=lambda: PACKAGE_ROOT / "outputs" / "power_phase"
    )
    battery_rul_outputs_dir: Path = field(
        default_factory=lambda: PACKAGE_ROOT / "outputs" / "battery_rul"
    )
    power_dual_outputs_dir: Path = field(
        default_factory=lambda: PACKAGE_ROOT / "outputs" / "power_dual"
    )

    def ensure_directories(self) -> None:
        for directory in (self.data_dir, self.outputs_dir, self.utils_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def ensure_power_directories(self) -> None:
        for directory in (
            self.data_dir,
            self.power_outputs_dir,
            self.power_phase_outputs_dir,
            self.battery_rul_outputs_dir,
            self.power_dual_outputs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    sequence_length: int = 10
    feature_columns: tuple[str, ...] = (
        "cpu",
        "memory",
        "in_octets",
        "out_octets",
        "errors",
    )
    train_split: float = 0.8
    normal_label: int = 0
    save_scaler: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    hidden_size: int = 64
    latent_size: int = 32
    threshold_std_multiplier: float = 2.0


@dataclass(frozen=True)
class InferenceConfig:
    save_results: bool = True


@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: tuple[str, ...] = ("localhost:9092",)
    consumer_group_id: str = "snmp-anomaly-detection"
    poll_timeout_ms: int = 1000
    micro_batch_size: int = 8
    micro_batch_max_wait_ms: int = 50
    save_local_results: bool = True


# ---------------------------------------------------------------------------
# Power domain configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PowerDeviceConfig:
    device_category: str = "ups"          # ups | pdu | network | env
    vendor: str = "apc"                   # apc | liebert | raritan | cisco | generic
    phase_count: int = 1                  # 1 (single-phase) or 3 (three-phase)
    rated_capacity_w: float = 3000.0      # nameplate power rating in Watts
    battery_ah: float = 7.2              # battery capacity in Amp-hours (UPS only)
    battery_expected_life_years: float = 3.0

    def __post_init__(self) -> None:
        if self.device_category not in ("ups", "pdu", "network", "env"):
            raise ValueError(f"Unknown device_category: {self.device_category}")
        if self.phase_count not in (1, 3):
            raise ValueError("phase_count must be 1 or 3")


@dataclass(frozen=True)
class PowerTrainingConfig:
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    hidden_size: int = 64
    latent_size: int = 32
    threshold_std_multiplier: float = 3.0
    dropout: float = 0.1              # used for MC Dropout in RUL confidence intervals


@dataclass(frozen=True)
class PowerInferenceConfig:
    alert_policy: str = "or"          # "or" = either model; "and" = both must trigger
    rul_urgent_days: int = 14
    rul_warn_days: int = 30
    compound_alert_window_minutes: int = 10
    save_results: bool = True
