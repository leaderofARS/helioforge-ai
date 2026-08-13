#HelioForge AI

> AI-powered Solar Flare Prediction and Analysis using Aditya-L1 (SoLEXS & HEL1OS) scientific data.

---

## Dashboard deployment (EC2)

The dashboard, FastAPI service, and reverse proxy run as one Compose stack.
Install Docker Engine with the Compose plugin, then from the repository root:

```bash
docker compose up --build -d
```

Open `http://<EC2-public-IP>/`. The API is proxied under `/api`; no separate
port needs to be exposed. Check startup and model loading with:

```bash
docker compose ps
curl http://localhost/api/health
docker compose logs -f backend
```

All data paths are read from `configs/data_paths.yaml` and used unchanged.
The Compose stack mounts `/opt/helioforge-ai/data`, `/opt/helioforge-ai/models`,
and `/opt/helioforge-ai/experiments` at those same paths. The trained
`best_macro_f1.pt` must be in the configured baseline checkpoint/model path;
the configured `windows.test` file powers the live demo endpoint.

---

## Overview

HelioForge AI is a machine learning platform designed to preprocess, analyze, visualize, and predict solar flare activity using scientific observations from **ISRO's Aditya-L1 mission**.

The project combines scientific data processing, feature engineering, machine learning, deep learning, and interactive visualization into a unified pipeline.

---

## Objectives

- Read and process Aditya-L1 scientific datasets
- Build robust preprocessing pipelines
- Engineer meaningful solar physics features
- Train Machine Learning and Deep Learning models
- Predict solar flare events
- Provide an interactive Streamlit dashboard
- Build a scalable research platform

---

# Dataset

Primary Instruments

- HEL1OS
- SoLEXS

Supported File Types

- `.fits`
- `.lc`
- `.pi`
- `.gti`

Dataset Storage

```
AWS S3
        ↓
EC2
        ↓
/opt/helioforge-ai/data/raw
```

Datasets are **NOT stored inside GitHub**.

---

# Tech Stack

## Programming

- Python 3.11

## Scientific Computing

- NumPy
- Pandas
- SciPy
- Astropy
- SunPy

## Machine Learning

- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost

## Deep Learning

- PyTorch

## Visualization

- Plotly
- Streamlit
- Matplotlib

## Cloud

- AWS EC2
- AWS S3
- IAM

## Version Control

- Git
- GitHub

---

# Project Structure

```
helioforge-ai/
│
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
│
├── configs/
├── docs/
├── notebooks/
├── scripts/
├── tests/
│
├── artifacts/
│   ├── checkpoints/
│   ├── figures/
│   ├── models/
│   └── results/
│
├── data/
│   ├── external/
│   ├── interim/
│   └── processed/
│
├── logs/
│
├── README.md
├── requirements.txt
├── environment.yml
├── pyproject.toml
└── setup.py
```

---

# Shared Infrastructure

The project uses a shared AWS development server.

Shared Resources

```
EC2 Ubuntu Server

Shared Conda Environment
/opt/conda-envs/helioforge

Shared Dataset
/opt/helioforge-ai/data/raw

Shared Models
/opt/helioforge-ai/models

Shared Outputs
/opt/helioforge-ai/outputs

Shared Logs
/opt/helioforge-ai/logs
```

Each developer maintains their **own Git clone** while sharing the same dataset and Python environment.

---

# Development Workflow

Every developer works on a separate branch.

Example

```
main

├── prakul/preprocessing
├── raghuveer/features
├── balaji/dashboard
└── abhay/integration
```

Never commit directly to `main`.

All changes must be submitted through Pull Requests.

---

# Getting Started

## Clone Repository

```bash
git clone git@github.com:leaderofARS/helioforge-ai.git

cd helioforge-ai
```

---

## Activate Environment

```bash
source /opt/helioforge-ai/bootstrap.sh
```

---

## Verify Installation

```bash
python -c "import numpy, pandas, astropy, torch"
```

---

## Verify Dataset

```bash
ls /opt/helioforge-ai/data/raw
```

Expected

```
hel1os
solexs
```

---

# Development Rules

- Never commit datasets.
- Never modify another developer's branch.
- Commit frequently with meaningful messages.
- Use Pull Requests for merging.
- Keep `main` stable.
- Write tests for new functionality.
- Document significant changes.

---

# Team

| Member | Responsibility |
|----------|---------------|
| Abhay R S | Project Lead, Architecture, Integration, AWS Infrastructure |
| Prakul P Shetty | Data Ingestion & Scientific File Parsers |
| Raghuveer | Feature Engineering & Machine Learning |
| Balaji | Dashboard, Visualization & UI |

---

# Roadmap

## Phase 1

- Infrastructure
- Dataset Management
- Development Environment

**Status:** ✅ Completed

---

## Phase 2

- FITS Reader
- SoLEXS Parser
- HEL1OS Parser
- Metadata Extraction

**Status:** 🚧 In Progress

---

## Phase 3

- Data Cleaning
- Preprocessing
- Feature Engineering

---

## Phase 4

- Machine Learning Models
- Deep Learning Models
- Training Pipeline

---

## Phase 5

- Evaluation
- Explainability
- Model Optimization

---

## Phase 6

- Streamlit Dashboard
- Deployment
- Documentation

---

# License

This project is licensed under the MIT License.

See the `LICENSE` file for details.

---

# Repository Status

Current Version

```
v0.1.0
```

Project Status

```
Active Development
```

---

# Contact

Repository Owner

**leaderofARS**

For project discussions, use GitHub Issues and Pull Requests.
