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
    "battery_voltage_ratio",      # battery_voltage_v / rated_battery_v — vendor-agnostic
    "battery_current_ratio",      # battery_current_a / rated_discharge_current — vendor-agnostic
    "battery_temperature_c",
    "runtime_remaining_min",
    "on_battery_status",
    "battery_replace_status",
    "input_voltage_dev_pct",      # (input_voltage_v - nominal_voltage_v) / nominal_voltage_v * 100
    "input_frequency_hz",
    "output_voltage_dev_pct",     # (output_voltage_v - nominal_voltage_v) / nominal_voltage_v * 100
    "output_current_ratio",       # output_current_a / (rated_capacity_w / nominal_voltage_v)
    "output_load_pct",
    "output_frequency_hz",
    # output_power_w dropped — redundant with output_load_pct after normalization
    # Rate-of-change features — capture slow-changing signals within a window
    # battery_charge_delta omitted: charge changes ~0.02%/step (noise after scaling),
    # LSTM reconstructs it with MSE=2.37 on normal data → inflates UPS threshold to 43×
    # battery_charge_pct sequence already gives the LSTM the trend signal implicitly.
    "runtime_delta",
    "temperature_delta",
    "output_load_delta",
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
    # B3: voltage drop rate-of-change — amplifies phase sag signal diluted in global MSE
    "voltage_drop_delta_l1",
    "voltage_drop_delta_l2",
    "voltage_drop_delta_l3",
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

# ---------------------------------------------------------------------------
# Model capability registry — single source of truth for which device
# categories each model applies to.  To enable a model for a new device type,
# add the category string here only — no inference or preprocessing code changes.
# Long-term these will become capability flags on PowerDeviceProfile (see A1).
# ---------------------------------------------------------------------------
BASELINE_MODEL_CATEGORIES: frozenset[str] = frozenset({"ups", "pdu", "network", "env"})
PHASE_MODEL_CATEGORIES: frozenset[str] = frozenset({"ups"})
BATTERY_RUL_CATEGORIES: frozenset[str] = frozenset({"ups"})

# Features excluded from IForest training and scoring per device category.
# Battery features are always 0.0 for non-UPS devices — constant zeros corrupt
# isolation tree splits and shift the score distribution, producing false positives.
# output_frequency_hz is excluded for non-UPS for the same reason the LSTM excludes
# it from MSE: grid noise (~50/60 Hz ± tiny variation) is not a fault signal.
_IF_BATTERY_FEATURES: frozenset[str] = frozenset({
    "battery_voltage_ratio",
    "battery_current_ratio",
    "on_battery_status",
    "battery_replace_status",
})

IF_EXCLUDED_FEATURES: dict[str, frozenset[str]] = {
    "ups":     frozenset(),
    "pdu":     _IF_BATTERY_FEATURES | {"output_frequency_hz"},
    "network": _IF_BATTERY_FEATURES | {"output_frequency_hz"},
    "env":     _IF_BATTERY_FEATURES | {"output_frequency_hz"},
}

# Minimum ratio of (if_score / if_threshold) required to raise an IF flag.
# Filters borderline flags whose ratios cluster at 1.00–1.10 on normal data.
# Revisit with real device data (A1) — auto-calibration per category planned (E4b).
IF_MIN_SCORE_RATIO: float = 1.05


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
    device_stats_file: Path = field(
        default_factory=lambda: PACKAGE_ROOT / "outputs" / "power_dual" / "device_stats.json"
    )
    iforest_model_file: Path = field(
        default_factory=lambda: PACKAGE_ROOT / "outputs" / "power_dual" / "iforest_models.pkl"
    )
    iforest_metadata_file: Path = field(
        default_factory=lambda: PACKAGE_ROOT / "outputs" / "power_dual" / "iforest_metadata.json"
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
    threshold_std_multiplier: float = 3.0


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
    epochs: int = 100
    learning_rate: float = 1e-3
    hidden_size: int = 64
    latent_size: int = 32
    threshold_std_multiplier: float = 2.0  # lowered from 3.0: better recall on soft anomalies
    seq_len: int = 20                       # 20 × 5-min steps = 100-min context window
    dropout: float = 0.3              # MC Dropout for RUL CIs — must be ≥ 0.2 for meaningful variance
    mc_samples: int = 50              # forward passes per MC Dropout prediction


@dataclass(frozen=True)
class PowerInferenceConfig:
    alert_policy: str = "or"          # "or" = either model; "and" = both must trigger
    rul_urgent_days: int = 14
    rul_warn_days: int = 30
    compound_alert_window_minutes: int = 10
    save_results: bool = True
    # B1 — per-device online threshold
    n_calibration_windows: int = 50   # windows observed silently before device threshold locks in
    online_threshold_k: float = 3.0   # mean + k*std defines the device threshold after calibration
    calibration_safety_multiplier: float = 3.0  # during cold-start, fire only if error > N × category threshold
    # E7 — two-stage alert lifecycle
    n_confirmation_windows: int = 3   # LSTM windows to wait before auto-clearing a SUSPECTED alert
    # Overload rule — output_load_pct above this value is treated as a definitive overload fault.
    # 100.0 is physically correct for most vendors; adjust per vendor if their agent reports
    # transient brief spikes above 100 under normal load (e.g., some PDU firmware).
    overload_load_pct_threshold: float = float("inf")  # TODO: restore to 100.0 once vendor OID scaling is verified
