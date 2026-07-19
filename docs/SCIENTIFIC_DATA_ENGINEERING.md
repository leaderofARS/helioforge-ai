# SOLAR PRELUDE – Scientific Data Engineering Documentation

## 1. Overview

This document summarizes the Scientific Data Engineering work completed for the SOLAR PRELUDE project under the ISRO Bharatiya Antariksh Hackathon 2026.

The objective of this phase was to preprocess scientific observations obtained from the Aditya-L1 mission and prepare them for downstream machine learning tasks.

The preprocessing pipeline focuses on reading FITS files, validating scientific datasets, extracting metadata, organizing processed data, and generating machine-learning-ready features.

---

# 2. Instruments Used

The project uses scientific observations from two payloads onboard Aditya-L1.

## HEL1OS

HEL1OS provides event-based solar X-ray observations together with supporting scientific datasets such as:

- Event Dataset
- Good Time Interval (GTI)
- Housekeeping Dataset
- Light Curves
- Spectra

## SoLEXS

SoLEXS provides solar X-ray observations through:

- Light Curve Dataset
- Spectrum Dataset
- Good Time Interval (GTI)

---

# 3. Technologies Used

The preprocessing pipeline was developed using:

- Python 3
- Astropy
- NumPy
- Pandas
- Pathlib

Development Environment

- Visual Studio Code
- Jupyter Notebook
- Git
- GitHub

---

# 4. Scientific Data Engineering Workflow

The preprocessing workflow consists of the following stages.

1. FITS File Exploration
2. Scientific Dataset Inspection
3. Metadata Extraction
4. Dataset Validation
5. Timestamp Analysis
6. Scientific Preprocessing
7. Observation Synchronization
8. Feature Engineering
9. Processed Data Generation

---

# 5. HEL1OS Processing

The following preprocessing tasks were completed for HEL1OS.

- FITS file exploration
- Event dataset loading
- GTI dataset loading
- Housekeeping dataset loading
- Metadata extraction
- Dataset validation
- Timestamp validation
- Scientific summary generation
- CSV generation

---

# 6. SoLEXS Processing

The following preprocessing tasks were completed for SoLEXS.

- FITS file exploration
- Light Curve loading
- GTI loading
- Spectrum inspection
- Metadata extraction
- Dataset validation
- Scientific summary generation
- CSV generation

---

# 7. Metadata Extraction

Metadata was extracted from every scientific file.

The extracted information includes:

- File Name
- File Type
- File Size
- Number of HDUs
- Instrument
- Telescope
- Observation Date
- Object
- Creator

Metadata reports are automatically generated during preprocessing.

---

# 8. Dataset Validation

Several validation checks were performed.

These include:

- Missing value detection
- Empty dataset validation
- Timestamp validation
- Numeric value validation
- Negative value detection
- Scientific dataset verification

The validation confirmed that the datasets are suitable for preprocessing.

---

# 9. Synchronization

A synchronization module was developed to compare the observation windows of HEL1OS and SoLEXS datasets.

The module generates a synchronization report containing:

- HEL1OS observation interval
- SoLEXS observation interval
- GTI observation interval

This prepares the datasets for future cross-instrument analysis.

---

# 10. Feature Engineering

Initial machine learning features were extracted from both datasets.

HEL1OS Features

- Event records
- Average energy
- Maximum energy
- Minimum energy
- Energy standard deviation
- Average detector temperature
- Observation interval

SoLEXS Features

- Light curve records
- Average counts
- Peak counts
- Count statistics
- Observation interval
- Spectrum records

The extracted features are combined into a single machine-learning-ready dataset.

---

# 11. Python Package Structure

The preprocessing package contains the following modules.

- config.py
- utils.py
- fits_reader.py
- metadata.py
- validation.py
- hel1os.py
- solexs.py
- synchronization.py
- feature_engineering.py
- main.py

Each module performs a dedicated preprocessing task.

---

# 12. Generated Outputs

Running the complete preprocessing pipeline automatically generates the following outputs.

Processed Data

processed/

- hel1os/
  - event.csv
  - gti.csv
  - housekeeping.csv

- solexs/
  - lightcurve.csv
  - gti.csv

- features/
  - ml_features.csv

Reports

reports/

- hel1os_metadata.csv
- solexs_metadata.csv
- synchronization_report.csv

---

# 13. Pipeline Execution

The complete preprocessing workflow is executed using a single command.

```bash
python main.py