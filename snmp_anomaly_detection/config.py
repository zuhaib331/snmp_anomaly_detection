from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


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

    def ensure_directories(self) -> None:
        for directory in (self.data_dir, self.outputs_dir, self.utils_dir):
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
