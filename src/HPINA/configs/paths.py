"""
src/HPINA/configs/paths.py
──────────────────────────
Typed Python loader for configs/data_paths.yaml.

Usage
-----
    from src.HPINA.configs.paths import PathConfig

    cfg = PathConfig.from_yaml("configs/data_paths.yaml")

    # Access any path as a pathlib.Path:
    print(cfg.raw.solexs)            # /opt/helioforge/raw/solexs
    print(cfg.features.parquet)      # /opt/helioforge/features/selected_features.parquet
    print(cfg.experiments.baseline_tcn.checkpoints)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Leaf dataclasses — one per YAML section
# ---------------------------------------------------------------------------

@dataclass
class RawPaths:
    root:   Path
    solexs: Path
    hel1os: Path


@dataclass
class PreprocessingPaths:
    validated:    Path
    synchronized: Path
    processed:    Path


@dataclass
class FeaturePaths:
    root:    Path
    csv:     Path
    parquet: Path
    excel:   Path


@dataclass
class WindowPaths:
    root:  Path
    train: Path
    val:   Path
    test:  Path


@dataclass
class MetadataPaths:
    root:            Path
    solexs_metadata: Path
    hel1os_metadata: Path
    sync_report:     Path
    gti_table:       Path


@dataclass
class ModelPaths:
    root:         Path
    baseline_tcn: Path
    multiscale:   Path
    dual_stream:  Path
    full_hpina:   Path


@dataclass
class BaselineTCNExperimentPaths:
    root:        Path
    runs:        Path
    checkpoints: Path
    ablations:   Path


@dataclass
class ExperimentPaths:
    root:         Path
    baseline_tcn: BaselineTCNExperimentPaths


@dataclass
class OutputPaths:
    root:        Path
    predictions: Path
    reports:     Path


@dataclass
class ReportPaths:
    root:    Path
    figures: Path
    tables:  Path


@dataclass
class LogPaths:
    root:       Path
    pipeline:   Path
    training:   Path
    evaluation: Path
    errors:     Path


@dataclass
class NormalisationPaths:
    stats_json: Path


@dataclass
class DatasetBudget:
    """
    Controls how many raw observations are loaded.

    mode = 'all'    — process every file found (full 83.1 GB)
    mode = 'budget' — accumulate observations (sorted by name, reproducible)
                      until the per-instrument GB ceiling is reached.
    """
    mode:       str    # 'all' | 'budget'
    total_gb:   float  # informational: target total GB
    hel1os_gb:  float  # HEL1OS byte ceiling in GB
    solexs_gb:  float  # SoLEXS byte ceiling in GB
    sort_key:   str    # 'name' → sort folders alphabetically

    @property
    def hel1os_bytes(self) -> int:
        return int(self.hel1os_gb * 1024 ** 3)

    @property
    def solexs_bytes(self) -> int:
        return int(self.solexs_gb * 1024 ** 3)

    @property
    def is_budget_mode(self) -> bool:
        return self.mode == "budget"

# ---------------------------------------------------------------------------
# Root config object
# ---------------------------------------------------------------------------

@dataclass
class PathConfig:
    dataset_root:   Path
    dataset:        DatasetBudget
    raw:            RawPaths
    preprocessing:  PreprocessingPaths
    features:       FeaturePaths
    windows:        WindowPaths
    metadata:       MetadataPaths
    models:         ModelPaths
    experiments:    ExperimentPaths
    outputs:        OutputPaths
    reports:        ReportPaths
    logs:           LogPaths
    normalisation:  NormalisationPaths

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "PathConfig":
        """Load and parse configs/data_paths.yaml."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"data_paths.yaml not found at: {yaml_path.resolve()}")

        with open(yaml_path, "r") as fh:
            d = yaml.safe_load(fh)

        def p(val: str) -> Path:
            return Path(val)

        raw_d        = d["raw"]
        pre_d        = d["preprocessing"]
        feat_d       = d["features"]
        win_d        = d["windows"]
        meta_d       = d["metadata"]
        model_d      = d["models"]
        exp_d        = d["experiments"]
        out_d        = d["outputs"]
        rep_d        = d["reports"]
        log_d        = d["logs"]
        norm_d       = d["normalisation"]
        btcn_d       = exp_d["baseline_tcn"]
        bgt_d        = d.get("dataset", {})

        return cls(
            dataset_root  = p(d["dataset_root"]),
            dataset       = DatasetBudget(
                mode      = bgt_d.get("mode",       "all"),
                total_gb  = float(bgt_d.get("total_gb",  83.1)),
                hel1os_gb = float(bgt_d.get("hel1os_gb", 48.9)),
                solexs_gb = float(bgt_d.get("solexs_gb", 34.2)),
                sort_key  = bgt_d.get("sort_key",  "name"),
            ),
            raw           = RawPaths(
                root   = p(raw_d["root"]),
                solexs = p(raw_d["solexs"]),
                hel1os = p(raw_d["hel1os"]),
            ),
            preprocessing = PreprocessingPaths(
                validated    = p(pre_d["validated"]),
                synchronized = p(pre_d["synchronized"]),
                processed    = p(pre_d["processed"]),
            ),
            features      = FeaturePaths(
                root    = p(feat_d["root"]),
                csv     = p(feat_d["csv"]),
                parquet = p(feat_d["parquet"]),
                excel   = p(feat_d["excel"]),
            ),
            windows       = WindowPaths(
                root  = p(win_d["root"]),
                train = p(win_d["train"]),
                val   = p(win_d["val"]),
                test  = p(win_d["test"]),
            ),
            metadata      = MetadataPaths(
                root            = p(meta_d["root"]),
                solexs_metadata = p(meta_d["solexs_metadata"]),
                hel1os_metadata = p(meta_d["hel1os_metadata"]),
                sync_report     = p(meta_d["sync_report"]),
                gti_table       = p(meta_d["gti_table"]),
            ),
            models        = ModelPaths(
                root         = p(model_d["root"]),
                baseline_tcn = p(model_d["baseline_tcn"]),
                multiscale   = p(model_d["multiscale"]),
                dual_stream  = p(model_d["dual_stream"]),
                full_hpina   = p(model_d["full_hpina"]),
            ),
            experiments   = ExperimentPaths(
                root         = p(exp_d["root"]),
                baseline_tcn = BaselineTCNExperimentPaths(
                    root        = p(btcn_d["root"]),
                    runs        = p(btcn_d["runs"]),
                    checkpoints = p(btcn_d["checkpoints"]),
                    ablations   = p(btcn_d["ablations"]),
                ),
            ),
            outputs       = OutputPaths(
                root        = p(out_d["root"]),
                predictions = p(out_d["predictions"]),
                reports     = p(out_d["reports"]),
            ),
            reports       = ReportPaths(
                root    = p(rep_d["root"]),
                figures = p(rep_d["figures"]),
                tables  = p(rep_d["tables"]),
            ),
            logs          = LogPaths(
                root       = p(log_d["root"]),
                pipeline   = p(log_d["pipeline"]),
                training   = p(log_d["training"]),
                evaluation = p(log_d["evaluation"]),
                errors     = p(log_d["errors"]),
            ),
            normalisation = NormalisationPaths(
                stats_json = p(norm_d["stats_json"]),
            ),
        )

    # ------------------------------------------------------------------
    # Convenience: ensure all directories exist on the EC2 instance
    # ------------------------------------------------------------------

    def makedirs(self, exist_ok: bool = True) -> None:
        """
        Create every directory declared in data_paths.yaml.
        Call once during environment setup on EC2:

            from src.HPINA.configs.paths import PathConfig
            PathConfig.from_yaml("configs/data_paths.yaml").makedirs()
        """
        dirs: list[Path] = [
            self.dataset_root,
            self.raw.root, self.raw.solexs, self.raw.hel1os,
            self.preprocessing.validated, self.preprocessing.synchronized, self.preprocessing.processed,
            self.features.root,
            self.windows.root,
            self.metadata.root,
            self.models.root, self.models.baseline_tcn, self.models.multiscale,
            self.models.dual_stream, self.models.full_hpina,
            self.experiments.root,
            self.experiments.baseline_tcn.root,
            self.experiments.baseline_tcn.runs,
            self.experiments.baseline_tcn.checkpoints,
            self.experiments.baseline_tcn.ablations,
            self.outputs.root, self.outputs.predictions, self.outputs.reports,
            self.reports.root, self.reports.figures, self.reports.tables,
            self.logs.root,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=exist_ok)
        print(f"[PathConfig] Created {len(dirs)} directories under {self.dataset_root}")
