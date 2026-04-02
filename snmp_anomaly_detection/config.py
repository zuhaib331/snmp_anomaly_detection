from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR_NAME: Final[str] = "current"


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
    feature_columns: tuple[str, ...] = (
        "cpu",
        "memory",
        "in_rate",
        "out_rate",
        "error_rate",
    )
    train_split: float = 0.8
    normal_label: int = 0
    save_scaler: bool = True
    scaler_name: str = "minmax"
    log1p_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    hidden_size: int = 64
    latent_size: int = 32
    threshold_std_multiplier: float = 3.0


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
