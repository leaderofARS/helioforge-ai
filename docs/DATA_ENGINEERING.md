# HELIO-FORGE AI
# Phase 1 – Data Engineering Handover

# Table of Contents

1. Introduction
2. Phase 1 Overview
3. Complete Pipeline Architecture
4. Folder Structure
5. Module-by-Module Summary
6. Feature Engineering
7. Dataset Pipeline
8. Feature Selection
9. Visualization
10. Generated Outputs
11. Current Project Status
12. Known Limitations
13. TODO List
14. Before Starting ML
15. Phase 2 Roadmap

---

# 1. Introduction

This document serves as the complete handover documentation for the Data Engineering phase of HELIO-FORGE AI.

The purpose of this document is to explain:

- what has been implemented
- why it exists
- how data flows
- which files are responsible
- remaining improvements
- what should happen before Machine Learning begins

This document should be sufficient for continuing development without reading every source file individually.

---

# 2. Phase 1 Overview

Phase 1 converts raw FITS observations from HEL1OS and SoLEXS into an ML-ready feature dataset.

Pipeline:

Raw FITS

↓

Preprocessing

↓

Synchronization

↓

Observation Loader

↓

Dataset Builder

↓

Feature Engineering

↓

Dataset Exporter

↓

Feature Selection

↓

Visualization

↓

ML Ready Dataset

---

# 3. Folder Structure

Helio-forge/

├── preprocessing/

├── dataset_pipeline/

├── feature_engineering/

├── feature_selection/

├── visualization/

├── reports/

├── data/

├── notebooks/

├── docs/

└── main.py

Explain every folder in detail.

---

# 4. preprocessing/

Purpose

Responsible for converting raw FITS files into processed CSV observations.

Files

## hel1os.py

Responsibilities

- read HEL1OS FITS
- validate
- extract Event
- extract GTI
- extract Housekeeping

Outputs

processed/hel1os/

---

## solexs.py

Responsibilities

- load lightcurves
- extract GTI
- clean observations

Outputs

processed/solexs/

---

## synchronization.py

Responsibilities

- synchronize HEL1OS and SoLEXS observations
- generate synchronization report

Output

reports/synchronization_report.csv

Current Status

✔ Working

Future

ObservationLoader should consume this report.

---

# 5. dataset_pipeline/

Contains three major modules.

## observation_loader.py

Purpose

Loads synchronized observations.

Responsibilities

- locate processed observations
- read CSV files
- return NumPy arrays

Returns

timestamps

soft_signal

hard_signal

solexs_id

hel1os_id

Current Limitation

Currently uses

solexs_folders[0]

instead of synchronization_report.csv.

TODO

Read reports/synchronization_report.csv and automatically pair observations.

---

## dataset_builder.py

Purpose

Converts observations into feature dataset.

Responsibilities

Receive observation

↓

Call FeaturePipeline

↓

Generate feature dictionary

↓

Append dataset row

↓

Create DataFrame

Current Status

✔ Stable

TODO

Support optional class labels for supervised learning.

---

## dataset_exporter.py

Purpose

Export datasets.

Supports

CSV

Parquet

Excel

Excel Workbook

Dataset

Statistics

Missing Values

Correlation Matrix

Summary

Current Status

✔ Stable

TODO

- Export selected features automatically
- Include synchronization metadata
- Add observation metadata sheet

---

# 6. feature_engineering/

Contains all scientific feature extraction.

Files

soft_features.py

hard_features.py

temporal_features.py

frequency_features.py

wavelet_features.py

entropy_features.py

correlation_features.py

feature_pipeline.py

Explain every file.

For each include

Purpose

Scientific significance

Input

Output

Generated Features

---

## Feature Counts

Soft Features

...

Hard Features

...

Temporal Features

...

Frequency Features

...

Wavelet Features

...

Entropy Features

...

Correlation Features

...

Total

79 engineered features

---

# 7. Feature Selection

Pipeline

79 Features

↓

Variance Filter

↓

50 Features

↓

Correlation Filter

↓

7 Features

Files

variance_filter.py

correlation_filter.py

feature_importance.py

feature_selector.py

Current Status

✔ Stable

TODO

Support model-driven feature importance after baseline TCN.

---

# 8. Visualization

Modules

feature_count.py

correlation_heatmap.py

feature_distribution.py

missing_values.py

feature_selection_summary.py

pca_analysis.py

visualization_pipeline.py

Generated Reports

Feature Count

Heatmaps

Distributions

Missing Values

PCA

Selection Summary

Current Status

✔ Stable

Future

Interactive Plotly

Observation Viewer

Feature Importance Plots

---

# 9. Generated Outputs

processed/

metadata/

reports/

features/

visualizations/

Explain every generated file.

---

# 10. Current Project Status

Completed

✔ HEL1OS preprocessing

✔ SoLEXS preprocessing

✔ Validation

✔ Metadata extraction

✔ Synchronization

✔ Observation Loader

✔ Dataset Builder

✔ Feature Engineering

✔ Dataset Exporter

✔ Feature Selection

✔ Visualization

Pipeline Status

✔ End-to-end executable

Command

py main.py

---

# 11. Known Limitations

Observation Loader

Uses temporary pairing.

TODO

Use synchronization_report.csv.

Dataset Builder

No class labels yet.

Dataset Exporter

No selected-feature workbook.

Visualization

Static Matplotlib only.

---

# 12. Remaining TODOs

High Priority

☐ ObservationLoader → synchronization report

☐ Remove temporary pairing

Medium Priority

☐ Export selected feature workbook

☐ Add synchronization sheet

☐ Observation metadata sheet

Low Priority

☐ Interactive dashboard

☐ Plotly support

☐ Time-series visualization

---

# 13. Before Phase 2

Verify

✔ Feature schema frozen

✔ Pipeline reproducible

✔ Selected dataset generated

✔ Synchronization report generated

Pending

☐ Loader uses synchronization report

---

# 14. Phase 2

Machine Learning

Sequence Builder

↓

Scaling

↓

Dataset Split

↓

PyTorch Dataset

↓

DataLoader

↓

Baseline TCN

↓

Training

↓

Evaluation

↓

Inference

---

# Final Notes

The Data Engineering pipeline is considered complete.

Future work should focus on:

1. Observation pairing using synchronization report.

2. Machine Learning pipeline.

3. Model deployment.

No major architectural changes are expected for Phase 1 modules.