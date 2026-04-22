from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR_NAME: Final[str] = "current"
BASELINE_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "cpu",
    "memory",
    "in_rate",
    "out_rate",
    "error_rate",
)
F2_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "cpu",
    "memory",
    "in_rate",
    "out_rate",
    "error_rate",
    "utilization_in_pct",
    "utilization_out_pct",
    "discard_rate_in",
    "discard_rate_out",
    "packet_rate_in",
    "packet_rate_out",
    "in_out_ratio",
    "interface_oper_status",
    "interface_admin_status",
)
F3_CONTEXT_BASE_FEATURES: Final[tuple[str, ...]] = (
    "in_rate",
    "out_rate",
    "error_rate",
    "utilization_in_pct",
    "utilization_out_pct",
)
F3_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *F2_FEATURE_COLUMNS,
    *tuple(
        f"{feature_name}_{stat_name}"
        for feature_name in F3_CONTEXT_BASE_FEATURES
        for stat_name in ("rolling_mean", "rolling_std", "zscore", "trend")
    ),
    "burst_indicator",
)


@dataclass(frozen=True)
class ProjectPaths:
    package_root: Path = PACKAGE_ROOT
    repo_root: Path = field(default_factory=lambda: PACKAGE_ROOT.parent)
    data_dir: Path = field(default_factory=lambda: PACKAGE_ROOT / "data")
    outputs_dir: Path = field(default_factory=lambda: PACKAGE_ROOT / "outputs")
    utils_dir: Path = field(default_factory=lambda: PACKAGE_ROOT / "utils")
    artifacts_root_dir: Path = field(default_factory=lambda: PACKAGE_ROOT / "artifacts")
    artifact_dir_name: str = DEFAULT_ARTIFACT_DIR_NAME
    artifact_dir_override: str | None = None
    dataset_file: Path = field(
        default_factory=lambda: PACKAGE_ROOT / "data" / "synthetic_snmp_dataset.csv"
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

    @property
    def artifact_dir(self) -> Path:
        if self.artifact_dir_override:
            override_path = Path(self.artifact_dir_override)
            if not override_path.is_absolute():
                override_path = self.repo_root / override_path
            return override_path
        return self.artifacts_root_dir / self.artifact_dir_name

    @property
    def scaler_file(self) -> Path:
        return self.artifact_dir / "scaler.pkl"

    @property
    def x_train_file(self) -> Path:
        return self.artifact_dir / "X_train.npy"

    @property
    def x_test_file(self) -> Path:
        return self.artifact_dir / "X_test.npy"

    @property
    def y_train_file(self) -> Path:
        return self.artifact_dir / "y_train.npy"

    @property
    def y_test_file(self) -> Path:
        return self.artifact_dir / "y_test.npy"

    @property
    def model_file(self) -> Path:
        return self.artifact_dir / "lstm_autoencoder.pth"

    @property
    def model_metadata_file(self) -> Path:
        return self.artifact_dir / "model_metadata.json"

    @property
    def preprocessing_metadata_file(self) -> Path:
        return self.artifact_dir / "preprocessing_metadata.json"

    @property
    def p1_time_split_file(self) -> Path:
        return self.outputs_dir / f"{self.artifact_dir.name}_p1_time_split.json"

    @property
    def p1_baseline_metrics_file(self) -> Path:
        return self.outputs_dir / f"{self.artifact_dir.name}_p1_baseline_metrics.json"

    @property
    def p2_interface_capability_matrix_file(self) -> Path:
        return self.outputs_dir / "p2_interface_capability_matrix.json"

    @property
    def p2_extended_feature_schema_file(self) -> Path:
        return self.outputs_dir / "p2_extended_feature_schema.json"

    @property
    def p3_event_source_selection_file(self) -> Path:
        return self.outputs_dir / "p3_event_source_selection.json"

    @property
    def p3_normalized_event_schema_file(self) -> Path:
        return self.outputs_dir / "p3_normalized_event_schema.json"

    @property
    def p3_timestamp_alignment_notes_file(self) -> Path:
        return self.outputs_dir / "p3_timestamp_alignment_notes.json"

    @property
    def p3_sample_normalized_events_file(self) -> Path:
        return self.outputs_dir / "p3_sample_normalized_events.jsonl"

    @property
    def c1_normalized_events_file(self) -> Path:
        return self.outputs_dir / "c1_normalized_events.jsonl"

    @property
    def c1_rejected_events_file(self) -> Path:
        return self.outputs_dir / "c1_rejected_events.jsonl"

    @property
    def c1_normalization_summary_file(self) -> Path:
        return self.outputs_dir / "c1_normalization_summary.json"

    @property
    def c2_correlated_anomaly_windows_file(self) -> Path:
        return self.outputs_dir / "c2_correlated_anomaly_windows.json"

    @property
    def c2_correlation_summary_file(self) -> Path:
        return self.outputs_dir / "c2_correlation_summary.json"

    @property
    def c3_explained_anomaly_windows_file(self) -> Path:
        return self.outputs_dir / "c3_explained_anomaly_windows.json"

    @property
    def c3_explanation_summary_file(self) -> Path:
        return self.outputs_dir / "c3_explanation_summary.json"

    @property
    def legacy_scaler_file(self) -> Path:
        return self.utils_dir / "scaler.pkl"

    @property
    def legacy_x_train_file(self) -> Path:
        return self.outputs_dir / "X_train.npy"

    @property
    def legacy_x_test_file(self) -> Path:
        return self.outputs_dir / "X_test.npy"

    @property
    def legacy_y_train_file(self) -> Path:
        return self.outputs_dir / "y_train.npy"

    @property
    def legacy_y_test_file(self) -> Path:
        return self.outputs_dir / "y_test.npy"

    @property
    def legacy_model_file(self) -> Path:
        return self.outputs_dir / "lstm_autoencoder.pth"

    @property
    def legacy_model_metadata_file(self) -> Path:
        return self.outputs_dir / "model_metadata.json"

    def ensure_directories(self) -> None:
        for directory in (
            self.data_dir,
            self.outputs_dir,
            self.utils_dir,
            self.artifacts_root_dir,
            self.artifact_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    sequence_length: int = 10
    feature_columns: tuple[str, ...] = BASELINE_FEATURE_COLUMNS
    train_split: float = 0.8
    normal_label: int = 0
    save_scaler: bool = True
    scaler_name: str = "minmax"
    log1p_features: tuple[str, ...] = ()


def feature_columns_for_profile(profile_name: str) -> tuple[str, ...]:
    normalized_name = profile_name.strip().lower()
    if normalized_name == "baseline":
        return BASELINE_FEATURE_COLUMNS
    if normalized_name == "f2":
        return F2_FEATURE_COLUMNS
    if normalized_name == "f3":
        return F3_FEATURE_COLUMNS
    raise ValueError("Unsupported feature profile. Expected `baseline`, `f2`, or `f3`.")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    hidden_size: int = 64
    latent_size: int = 32
    threshold_mode: str = "stddev"
    threshold_std_multiplier: float = 3.0
    threshold_percentile: float = 99.5


@dataclass(frozen=True)
class EvaluationConfig:
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    report_top_n: int = 10


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
